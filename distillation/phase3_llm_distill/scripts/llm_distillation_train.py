#!/usr/bin/env python3
"""
LLM蒸馏训练脚本 - 完整版本
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
    Qwen3VLForConditionalGeneration,
    Qwen3VLConfig,
    get_linear_schedule_with_warmup,
)

# 路径配置
TOKENIZER_PATH = "/data01/mikelee/weight/alpamayo2B/tokenizer_final"
MODEL_PATH = "/data01/mikelee/weight/alpamayo2B"
DATASET_PATH = "/gpfs-data/mikelee/llm_distillation_data/distillation_dataset.jsonl"
OUTPUT_DIR = "/gpfs-data/mikelee/llm_distillation_output"

# 训练配置
DEFAULT_CONFIG = {
    "batch_size": 1,  # 每GPU，根据显存调整
    "num_gpus": 3,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "num_epochs": 3,
    "warmup_steps": 1000,
    "max_grad_norm": 1.0,
    "save_steps": 5000,
    "eval_steps": 2500,
    "logging_steps": 100,
    "gradient_accumulation_steps": 4,
    "max_samples": None,  # None表示使用全部数据
}


class LLMDistillationDataset(Dataset):
    """LLM蒸馏数据集"""
    
    def __init__(self, dataset_path, tokenizer, processor, max_samples=None):
        self.tokenizer = tokenizer
        self.processor = processor
        self.samples = []
        
        # 加载数据集
        with open(dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图片
        from PIL import Image
        images = []
        for img_path in sample['image_paths']:
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}")
                # 使用空白图占位
                images.append(Image.new('RGB', (576, 320), color='black'))
        
        # 构建prompt
        cot_text = sample['cot_text']
        
        # 构建包含图片占位符的prompt
        image_placeholders = ""
        for _ in images:
            image_placeholders += "<|vision_start|><|image_pad|><|vision_end|>"
        
        messages = [
            {
                "role": "system",
                "content": "You are a driving assistant that generates safe and accurate actions."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{image_placeholders}<|traj_history_start|><|traj_history|>*48<|traj_history_end|>output the chain-of-thought reasoning of the driving process, then output the future trajectory."}
                ]
            },
            {
                "role": "assistant",
                "content": f"<|cot_start|>{cot_text}<|cot_end|>"
            }
        ]
        
        # 应用chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # 处理图片和文本
        try:
            inputs = self.processor(
                text=[text],
                images=images,
                return_tensors="pt",
                padding=True,
            )
            
            # 移除batch维度
            inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # 确保image_grid_thw格式正确
            if 'image_grid_thw' in inputs:
                image_grid_thw = inputs['image_grid_thw']
                if isinstance(image_grid_thw, torch.Tensor):
                    if image_grid_thw.dim() == 3 and image_grid_thw.shape[0] == 1:
                        image_grid_thw = image_grid_thw.squeeze(0)
                    inputs['image_grid_thw'] = image_grid_thw
            
            inputs['labels'] = inputs['input_ids'].clone()
            
            return inputs
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # 返回空样本
            return {
                'input_ids': torch.tensor([0]),
                'attention_mask': torch.tensor([1]),
                'labels': torch.tensor([0]),
            }


def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    # 找到最大长度
    max_len = max(item['input_ids'].shape[0] for item in batch)
    
    # 填充
    input_ids = []
    attention_masks = []
    labels = []
    
    for item in batch:
        seq_len = item['input_ids'].shape[0]
        pad_len = max_len - seq_len
        
        input_ids.append(torch.cat([item['input_ids'], torch.full((pad_len,), 0, dtype=torch.long)]))
        attention_masks.append(torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([item['labels'], torch.full((pad_len,), -100, dtype=torch.long)]))
    
    result = {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.stack(labels),
    }
    
    # 处理pixel_values和image_grid_thw（假设所有样本相同）
    if 'pixel_values' in batch[0]:
        result['pixel_values'] = torch.stack([item['pixel_values'] for item in batch])
    
    if 'image_grid_thw' in batch[0]:
        image_grid_thw = batch[0]['image_grid_thw']
        if isinstance(image_grid_thw, torch.Tensor):
            result['image_grid_thw'] = image_grid_thw.unsqueeze(0).repeat(len(batch), 1, 1)
        else:
            result['image_grid_thw'] = [item['image_grid_thw'] for item in batch]
    
    return result


class LLMDistillationTrainer:
    """LLM蒸馏训练器"""
    
    def __init__(self, config):
        self.config = {**DEFAULT_CONFIG, **config}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载tokenizer和processor
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
        # 加载模型
        self.model = self._load_model()
        
        # 冻结ViT，只训练LLM
        self._freeze_vit()
        
        # 优化器
        self.optimizer = self._build_optimizer()
        
        # 学习率调度
        self.scheduler = None  # 在train中初始化
        
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
        print(f"Model config: vocab_size={config.text_config.vocab_size}, hidden_size={config.text_config.hidden_size}")
        
        # 加载模型
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {total_params / 1e6:.0f}M parameters")
        
        return model
    
    def _freeze_vit(self):
        """冻结Vision Encoder，只训练LLM部分"""
        print("Freezing Vision Encoder...")
        
        # 冻结视觉编码器
        for param in self.model.visual.parameters():
            param.requires_grad = False
        
        # 统计可训练参数
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable / 1e6:.0f}M / {total / 1e6:.0f}M ({trainable/total*100:.1f}%)")
    
    def _build_optimizer(self):
        """构建优化器"""
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        return optimizer
    
    def train_step(self, batch):
        """单步训练"""
        self.model.train()
        
        # 将数据移到设备
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        pixel_values = batch.get('pixel_values', None)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        
        image_grid_thw = batch.get('image_grid_thw', None)
        if image_grid_thw is not None:
            if isinstance(image_grid_thw, torch.Tensor):
                if image_grid_thw.dim() == 3 and image_grid_thw.shape[0] == 1:
                    image_grid_thw = image_grid_thw.squeeze(0)
                image_grid_thw = image_grid_thw.to(self.device).long()
        
        # Forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )
        
        loss = outputs.loss
        
        # 梯度累积
        if self.config['gradient_accumulation_steps'] > 1:
            loss = loss / self.config['gradient_accumulation_steps']
        
        # Backward
        loss.backward()
        
        return loss.item()
    
    def train(self, train_dataloader, eval_dataloader=None):
        """主训练循环"""
        print("=" * 70)
        print("Starting LLM Distillation Training")
        print("=" * 70)
        
        # 初始化学习率调度器
        total_steps = len(train_dataloader) * self.config['num_epochs'] // self.config['gradient_accumulation_steps']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_steps,
        )
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Training
            train_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc="Training")
            
            for step, batch in enumerate(progress_bar):
                loss = self.train_step(batch)
                train_loss += loss
                
                # 梯度更新
                if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['max_grad_norm']
                    )
                    
                    # 更新参数
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # 日志
                    if self.global_step % self.config['logging_steps'] == 0:
                        avg_loss = train_loss / self.config['logging_steps']
                        self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                        self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                        })
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
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                pixel_values = batch.get('pixel_values', None)
                if pixel_values is not None:
                    pixel_values = pixel_values.to(self.device)
                
                image_grid_thw = batch.get('image_grid_thw', None)
                if image_grid_thw is not None:
                    if isinstance(image_grid_thw, torch.Tensor):
                        if image_grid_thw.dim() == 3 and image_grid_thw.shape[0] == 1:
                            image_grid_thw = image_grid_thw.squeeze(0)
                        image_grid_thw = image_grid_thw.to(self.device).long()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels,
                )
                
                loss = outputs.loss
                total_loss += loss.item()
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
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
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
        "max_samples": args.max_samples,
    }
    
    # 创建数据集
    print("Loading dataset...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    dataset = LLMDistillationDataset(
        dataset_path=args.dataset,
        tokenizer=tokenizer,
        processor=processor,
        max_samples=args.max_samples,
    )
    
    # 划分训练/验证集 (95/5)
    train_size = int(0.95 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    print(f"Train batches: {len(train_dataloader)}, Eval batches: {len(eval_dataloader)}")
    
    # 创建训练器
    trainer = LLMDistillationTrainer(config)
    
    # 开始训练
    trainer.train(train_dataloader, eval_dataloader)


if __name__ == "__main__":
    main()
