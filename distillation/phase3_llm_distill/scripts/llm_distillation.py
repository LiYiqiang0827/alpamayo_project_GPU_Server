#!/usr/bin/env python3
"""
LLM蒸馏训练脚本
基于CosmosReason2-2B扩展模型，蒸馏Alpamayo1.5-10B的CoT生成能力
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transformers import (
    AutoTokenizer, 
    AutoProcessor,
    AutoModelForCausalLM,
    Qwen3VLForConditionalGeneration,
    Qwen3VLConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

# 路径配置
TOKENIZER_PATH = "/data01/mikelee/weight/alpamayo2B/tokenizer_final"
MODEL_PATH = "/data01/mikelee/weight/alpamayo2B"
DATASET_PATH = "/gpfs-data/mikelee/llm_distillation_data/distillation_dataset.jsonl"
OUTPUT_DIR = "/gpfs-data/mikelee/llm_distillation_output"

# 训练配置
DEFAULT_CONFIG = {
    "batch_size": 2,  # 每GPU，根据显存调整
    "num_gpus": 3,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "num_epochs": 3,
    "warmup_steps": 500,
    "max_grad_norm": 1.0,
    "save_steps": 1000,
    "eval_steps": 500,
    "logging_steps": 100,
    "max_seq_length": 2048,  # 最大序列长度
    "gradient_accumulation_steps": 4,
    "fp16": True,
    "bf16": False,
}


class LLMDistillationDataset(Dataset):
    """
    LLM蒸馏数据集
    
    从JSON Lines文件加载，每条记录包含：
    - image_paths: 16张图片的路径
    - history_traj_path: 历史轨迹npy文件路径
    - cot_text: Teacher的CoT输出文本
    """
    
    def __init__(self, dataset_path, tokenizer, processor, max_samples=None):
        self.tokenizer = tokenizer
        self.processor = processor
        self.samples = []
        
        # 加载数据集
        with open(dataset_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))
                if max_samples and len(self.samples) >= max_samples:
                    break
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图片
        images = []
        for img_path in sample['image_paths']:
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            images.append(img)
        
        # 加载历史轨迹
        if sample['history_traj_path'] and os.path.exists(sample['history_traj_path']):
            history_traj = np.load(sample['history_traj_path'])  # (16, 7)
        else:
            history_traj = np.zeros((16, 7), dtype=np.float32)
        
        # 构建prompt
        cot_text = sample['cot_text']
        
        # 使用processor处理图片和文本
        # 注意：这里需要构建完整的对话格式
        messages = self._build_messages(cot_text)
        
        # 应用chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # 处理图片和文本
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        
        # 移除batch维度（DataLoader会添加）
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # 添加labels（用于计算loss）
        # labels = input_ids，但只计算assistant部分的loss
        inputs['labels'] = inputs['input_ids'].clone()
        
        return inputs
    
    def _build_messages(self, cot_text):
        """构建Alpamayo格式的对话消息"""
        # System
        system_msg = {
            "role": "system",
            "content": "You are a driving assistant that generates safe and accurate actions."
        }
        
        # User: 图片 + 轨迹 + prompt
        user_content = []
        
        # 添加16张图片（在processor中会被处理）
        # 图片已经在__getitem__中加载，这里只需要文本部分
        
        # 轨迹占位符（48个token）
        traj_text = "<|traj_history_start|>" + "<|traj_history|>" * 48 + "<|traj_history_end|>"
        
        # Prompt
        prompt_text = f"{traj_text}output the chain-of-thought reasoning of the driving process, then output the future trajectory."
        
        user_content.append({"type": "text", "text": prompt_text})
        
        user_msg = {
            "role": "user",
            "content": user_content
        }
        
        # Assistant: CoT（训练目标）
        assistant_msg = {
            "role": "assistant",
            "content": f"<|cot_start|>{cot_text}<|cot_end|>"
        }
        
        return [system_msg, user_msg, assistant_msg]


class LLMDistillationTrainer:
    """LLM蒸馏训练器"""
    
    def __init__(self, config):
        self.config = {**DEFAULT_CONFIG, **config}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载tokenizer和processor
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        
        # 加载模型
        self.model = self._load_model()
        
        # 冻结ViT，只训练LLM
        self._freeze_vit()
        
        # 优化器
        self.optimizer = self._build_optimizer()
        
        # 学习率调度
        self.scheduler = self._build_scheduler()
        
        # 输出目录
        self.output_dir = Path(self.config.get("output_dir", OUTPUT_DIR))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / "logs")
        
        # 训练状态
        self.global_step = 0
        self.best_loss = float('inf')
    
    def _load_model(self):
        """加载扩展后的Cosmos-2B模型"""
        print(f"Loading model from: {MODEL_PATH}")
        
        # 加载配置
        config = Qwen3VLConfig.from_pretrained(MODEL_PATH)
        print(f"Model config: vocab_size={config.vocab_size}, hidden_size={config.text_config.hidden_size}")
        
        # 加载模型
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if self.config['fp16'] else torch.float32,
            device_map="auto",  # 自动分配层到各GPU
            trust_remote_code=True,
        )
        
        print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")
        
        return model
    
    def _freeze_vit(self):
        """冻结Vision Encoder，只训练LLM部分"""
        print("Freezing Vision Encoder...")
        
        # 冻结视觉编码器
        for param in self.model.visual.parameters():
            param.requires_grad = False
        
        # 冻结视觉相关的投影层
        if hasattr(self.model, 'visual_merger'):
            for param in self.model.visual_merger.parameters():
                param.requires_grad = False
        
        # 统计可训练参数
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable / 1e6:.0f}M / {total / 1e6:.0f}M ({trainable/total*100:.1f}%)")
    
    def _build_optimizer(self):
        """构建优化器"""
        # 只优化可训练参数
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        return optimizer
    
    def _build_scheduler(self):
        """构建学习率调度器"""
        from transformers import get_linear_schedule_with_warmup
        
        # 估算总步数
        total_steps = self.config['num_epochs'] * self.config['steps_per_epoch']
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_steps,
        )
        
        return scheduler
    
    def compute_loss(self, model_outputs, labels):
        """
        计算蒸馏损失
        
        简化版本：只使用CE Loss（硬标签）
        后续可以加入KL Divergence（软标签，需要Teacher logits）
        """
        logits = model_outputs.logits  # [batch, seq_len, vocab_size]
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss
    
    def train_step(self, batch):
        """单步训练"""
        self.model.train()
        
        # 将数据移到设备
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        pixel_values = batch.get('pixel_values', None)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        image_grid_thw = batch.get('image_grid_thw', None)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=None,  # 我们自己计算loss
        )
        
        # 计算loss
        loss = self.compute_loss(outputs, labels)
        
        # Backward
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config['max_grad_norm']
        )
        
        # 更新参数
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def eval_step(self, batch):
        """单步评估"""
        self.model.eval()
        
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            pixel_values = batch.get('pixel_values', None)
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device)
            image_grid_thw = batch.get('image_grid_thw', None)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=None,
            )
            
            loss = self.compute_loss(outputs, labels)
        
        return loss.item()
    
    def train(self, train_dataloader, eval_dataloader=None):
        """主训练循环"""
        print("=" * 70)
        print("Starting LLM Distillation Training")
        print("=" * 70)
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Training
            train_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc="Training")
            
            for step, batch in enumerate(progress_bar):
                loss = self.train_step(batch)
                train_loss += loss
                self.global_step += 1
                
                # 日志
                if self.global_step % self.config['logging_steps'] == 0:
                    avg_loss = train_loss / self.config['logging_steps']
                    self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                    self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})
                    train_loss = 0.0
                
                # 保存
                if self.global_step % self.config['save_steps'] == 0:
                    self.save_checkpoint()
                
                # 评估
                if eval_dataloader and self.global_step % self.config['eval_steps'] == 0:
                    eval_loss = self.evaluate(eval_dataloader)
                    self.writer.add_scalar("eval/loss", eval_loss, self.global_step)
                    print(f"\nEval loss: {eval_loss:.4f}")
                    
                    # 保存最佳模型
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.save_checkpoint(is_best=True)
            
            # Epoch结束保存
            self.save_checkpoint()
        
        print("\nTraining completed!")
        self.save_checkpoint(is_final=True)
    
    def evaluate(self, eval_dataloader):
        """评估"""
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            loss = self.eval_step(batch)
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, is_best=False, is_final=False):
        """保存检查点"""
        if is_best:
            save_dir = self.output_dir / "best"
        elif is_final:
            save_dir = self.output_dir / "final"
        else:
            save_dir = self.output_dir / f"checkpoint-{self.global_step}"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # 保存训练状态
        state = {
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "config": self.config,
        }
        with open(save_dir / "trainer_state.json", 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Checkpoint saved: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="LLM Distillation Training")
    parser.add_argument("--dataset", default=DATASET_PATH, help="Path to distillation dataset")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--num-gpus", type=int, default=3, help="Number of GPUs")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples for debugging")
    
    args = parser.parse_args()
    
    # 构建配置
    config = {
        "batch_size": args.batch_size,
        "num_gpus": args.num_gpus,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "output_dir": args.output_dir,
    }
    
    # 创建数据集
    print("Loading dataset...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    
    dataset = LLMDistillationDataset(
        dataset_path=args.dataset,
        tokenizer=tokenizer,
        processor=processor,
        max_samples=args.max_samples,
    )
    
    # 划分训练/验证集
    train_size = int(0.95 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    config['steps_per_epoch'] = len(train_dataloader)
    
    # 创建训练器
    trainer = LLMDistillationTrainer(config)
    
    # 开始训练
    trainer.train(train_dataloader, eval_dataloader)


if __name__ == "__main__":
    main()
