#!/usr/bin/env python3
"""
LLM蒸馏训练脚本 - 完整版本 (v2)
基于CosmosReason2-2B扩展模型，蒸馏Alpamayo1.5-10B的CoT生成能力

改进点:
1. 加载教师模型 (Alpamayo1.5-10B) 并冻结所有参数
2. 加载学生模型 (Alpamayo2B) 并冻结ViT和基础词表(0-151642)
3. 实现KL散度损失函数，支持温度调度和多阶段训练
4. 支持预计算teacher logits以节省显存
5. 三阶段训练: CE预热 -> CE+KL联合 -> 降低temperature微调
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

# ==================== 路径配置 ====================
TEACHER_PATH = "/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B"
STUDENT_PATH = "/data01/mikelee/weight/alpamayo2B"
TOKENIZER_PATH = "/data01/mikelee/weight/alpamayo2B/tokenizer_final"
DATASET_PATH = "/gpfs-data/mikelee/llm_distillation_data/distillation_dataset.jsonl"
OUTPUT_DIR = "/gpfs-data/mikelee/llm_distillation_output"
TEACHER_LOGITS_DIR = "/gpfs-data/mikelee/llm_distillation_data/teacher_logits"  # 预计算teacher logits保存路径

# ==================== 词表范围配置 ====================
BASE_VOCAB_START = 0
BASE_VOCAB_END = 151642       # 基础词表范围 [0, 151642]
EXTENDED_VOCAB_START = 151669  # 扩展词表起始
EXTENDED_VOCAB_END = 155696    # 扩展词表结束 [151669, 155696]

# ==================== 蒸馏配置 ====================
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
    
    # === 蒸馏特有配置 ===
    "temperature": 2.0,           # 默认温度
    "alpha": 0.7,                 # KL loss权重
    "beta": 0.3,                  # CE loss权重
    "precompute_teacher_logits": False,  # 是否预计算teacher logits
    "use_cached_logits": False,   # 是否使用缓存的teacher logits
    
    # === 三阶段训练配置 ===
    "phase1_warmup_epochs": 1,    # 阶段1: CE预热epoch数
    "phase1_enabled": True,       # 是否启用阶段1
    "phase2_epochs": 2,           # 阶段2: CE+KL联合训练epoch数
    "phase3_finetune_epochs": 1,  # 阶段3: 降低temperature微调epoch数
    "phase3_temperature": 1.0,    # 阶段3温度
    "phase3_lr": 1e-5,            # 阶段3学习率
}


# ==================== 数据集 ====================
class LLMDistillationDataset(Dataset):
    """LLM蒸馏数据集 - 支持预计算teacher logits"""
    
    def __init__(self, dataset_path, tokenizer, processor, max_samples=None, 
                 teacher_logits_dir=None, use_cached_logits=False):
        self.tokenizer = tokenizer
        self.processor = processor
        self.samples = []
        self.teacher_logits_dir = teacher_logits_dir
        self.use_cached_logits = use_cached_logits
        
        # 加载数据集
        with open(dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))
        
        print(f"Loaded {len(self.samples)} samples")
        
        # 如果启用缓存，检查缓存文件
        if self.use_cached_logits and self.teacher_logits_dir:
            missing = []
            for i in range(len(self.samples)):
                cache_path = Path(self.teacher_logits_dir) / f"sample_{i}.pt"
                if not cache_path.exists():
                    missing.append(i)
            if missing:
                print(f"Warning: {len(missing)}/{len(self.samples)} samples missing teacher logits cache")
    
    def __len__(self):
        return len(self.samples)
    
    def get_teacher_logits(self, idx):
        """获取预计算的teacher logits"""
        if not self.use_cached_logits or not self.teacher_logits_dir:
            return None
        cache_path = Path(self.teacher_logits_dir) / f"sample_{idx}.pt"
        if cache_path.exists():
            return torch.load(cache_path, map_location='cpu')
        return None
    
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
            
            # 添加teacher logits (如果缓存)
            if self.use_cached_logits:
                teacher_logits = self.get_teacher_logits(idx)
                if teacher_logits is not None:
                    inputs['teacher_logits'] = teacher_logits
            
            return inputs
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
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
    teacher_logits_list = []
    has_teacher_logits = 'teacher_logits' in batch[0]
    
    for item in batch:
        seq_len = item['input_ids'].shape[0]
        pad_len = max_len - seq_len
        
        input_ids.append(torch.cat([item['input_ids'], torch.full((pad_len,), 0, dtype=torch.long)]))
        attention_masks.append(torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([item['labels'], torch.full((pad_len,), -100, dtype=torch.long)]))
        
        # 处理teacher logits
        if has_teacher_logits and 'teacher_logits' in item:
            t_logits = item['teacher_logits']
            if isinstance(t_logits, torch.Tensor):
                # pad teacher logits to max_len
                vocab_size = t_logits.shape[-1]
                if t_logits.dim() == 2:  # (seq_len, vocab_size)
                    pad_logits = torch.full((pad_len, vocab_size), 0.0, dtype=t_logits.dtype)
                    teacher_logits_list.append(torch.cat([t_logits, pad_logits]))
                elif t_logits.dim() == 3:  # (1, seq_len, vocab_size)
                    pad_logits = torch.full((1, pad_len, vocab_size), 0.0, dtype=t_logits.dtype)
                    teacher_logits_list.append(torch.cat([t_logits, pad_logits], dim=1))
    
    result = {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.stack(labels),
    }
    
    if has_teacher_logits and teacher_logits_list:
        result['teacher_logits'] = torch.stack(teacher_logits_list)
    
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


# ==================== KL散度损失 ====================
class DistillationLoss(nn.Module):
    """知识蒸馏损失函数 - KL散度 + CE"""
    
    def __init__(self, temperature=2.0, alpha=0.7, beta=0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # KL loss权重
        self.beta = beta    # CE loss权重
        self.kl_div = nn.KLDivLoss(reduction='batchmean', log_target=False)
    
    def forward(self, student_logits, teacher_logits, labels, 
                mask=None, student_model=None):
        """
        Args:
            student_logits: (batch, seq_len, vocab_size)
            teacher_logits: (batch, seq_len, vocab_size) 或 None
            labels: (batch, seq_len) - 真实标签
            mask: (batch, seq_len) - 有效位置mask
            student_model: 学生模型，用于计算CE loss
        Returns:
            total_loss, kl_loss, ce_loss
        """
        batch_size, seq_len, vocab_size = student_logits.shape
        
        # 1. CE Loss (使用原始logits)
        ce_loss = 0.0
        if student_model is not None and labels is not None:
            # 使用transformers的loss计算
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            ce_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # 2. KL Loss (使用temperature缩放)
        kl_loss = 0.0
        if teacher_logits is not None:
            # 确保teacher和student的vocab大小一致
            min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
            s_logits = student_logits[..., :min_vocab]
            t_logits = teacher_logits[..., :min_vocab]
            
            # 应用temperature
            s_logits_scaled = s_logits / self.temperature
            t_logits_scaled = t_logits / self.temperature
            
            # 计算soft labels
            student_probs = F.log_softmax(s_logits_scaled, dim=-1)
            teacher_probs = F.softmax(t_logits_scaled, dim=-1)
            
            # 只在有效token位置计算KL (忽略pad和-100)
            if mask is not None:
                # 扩展mask到vocab维度
                mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
                
                # 应用mask
                student_probs_masked = student_probs * mask_expanded
                teacher_probs_masked = teacher_probs * mask_expanded
                
                # 计算KL散度 (手动实现以支持mask)
                kl_per_token = F.kl_div(
                    student_probs_masked, 
                    teacher_probs_masked, 
                    reduction='none',
                    log_target=False
                ).sum(dim=-1)  # (batch, seq_len)
                
                # 只对有效位置求和
                kl_loss = (kl_per_token * mask).sum() / mask.sum()
            else:
                # 无mask，对整个序列计算
                kl_loss = self.kl_div(
                    student_probs.view(-1, min_vocab),
                    teacher_probs.view(-1, min_vocab)
                )
            
            # 缩放回原始尺度 (temperature^2)
            kl_loss = kl_loss * (self.temperature ** 2)
        
        # 3. 总损失
        if teacher_logits is not None:
            total_loss = self.alpha * kl_loss + self.beta * ce_loss
        else:
            total_loss = ce_loss  # 只有CE loss (预热阶段)
        
        return total_loss, kl_loss, ce_loss


# ==================== 训练器 ====================
class LLMDistillationTrainer:
    """LLM蒸馏训练器 - 支持教师-学生联合训练"""
    
    def __init__(self, config):
        self.config = {**DEFAULT_CONFIG, **config}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载tokenizer和processor (教师和学生共享)
        print("Loading tokenizer and processor...")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(STUDENT_PATH, trust_remote_code=True)
        
        # 加载教师模型 (冻结)
        self.teacher_model = self._load_teacher_model()
        
        # 加载学生模型 (部分冻结)
        self.student_model = self._load_student_model()
        
        # 冻结指定部分
        self._freeze_parameters()
        
        # 蒸馏损失函数
        self.distillation_loss = DistillationLoss(
            temperature=self.config['temperature'],
            alpha=self.config['alpha'],
            beta=self.config['beta'],
        )
        
        # 优化器
        self.optimizer = self._build_optimizer()
        
        # 学习率调度
        self.scheduler = None
        
        # 输出目录
        self.output_dir = Path(self.config.get("output_dir", OUTPUT_DIR))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / "logs")
        
        # 训练状态
        self.global_step = 0
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.current_phase = 1  # 当前训练阶段
    
    def _load_teacher_model(self):
        """加载教师模型 (Alpamayo1.5-10B)"""
        print(f"\n{'='*70}")
        print(f"Loading TEACHER model from: {TEACHER_PATH}")
        print(f"{'='*70}")
        
        # 检查路径是否存在
        if not Path(TEACHER_PATH).exists():
            print(f"WARNING: Teacher path not found: {TEACHER_PATH}")
            print("Teacher model will not be loaded. Set precompute_teacher_logits=True")
            return None
        
        try:
            # 加载配置
            config = Qwen3VLConfig.from_pretrained(TEACHER_PATH)
            print(f"Teacher config: vocab_size={config.text_config.vocab_size}, "
                  f"hidden_size={config.text_config.hidden_size}, "
                  f"num_layers={config.text_config.num_hidden_layers}")
            
            # 加载模型 (使用float16和自动设备映射)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                TEACHER_PATH,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # 冻结所有参数
            for param in model.parameters():
                param.requires_grad = False
            
            model.eval()  # 设置为评估模式
            
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Teacher model loaded: {total_params / 1e9:.2f}B parameters")
            print(f"All teacher parameters FROZEN")
            
            return model
            
        except Exception as e:
            print(f"ERROR loading teacher model: {e}")
            print("Teacher model will not be loaded.")
            return None
    
    def _load_student_model(self):
        """加载学生模型 (Alpamayo2B)"""
        print(f"\n{'='*70}")
        print(f"Loading STUDENT model from: {STUDENT_PATH}")
        print(f"{'='*70}")
        
        # 加载配置
        config = Qwen3VLConfig.from_pretrained(STUDENT_PATH)
        print(f"Student config: vocab_size={config.text_config.vocab_size}, "
              f"hidden_size={config.text_config.hidden_size}, "
              f"num_layers={config.text_config.num_hidden_layers}")
        
        # 加载模型
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            STUDENT_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Student model loaded: {total_params / 1e6:.0f}M parameters")
        
        return model
    
    def _freeze_parameters(self):
        """冻结参数:
        - ViT: 冻结
        - 基础词表 (0-151642): 冻结
        - 扩展词表 (151669-155696): 训练
        - LLM部分: 训练
        """
        print(f"\n{'='*70}")
        print("Freezing parameters...")
        print(f"{'='*70}")
        
        # 1. 冻结ViT
        if hasattr(self.student_model, 'visual'):
            for param in self.student_model.visual.parameters():
                param.requires_grad = False
            print("✓ Vision Encoder (ViT) FROZEN")
        
        # 2. 处理词表embedding
        if hasattr(self.student_model, 'model') and hasattr(self.student_model.model, 'embed_tokens'):
            embed_tokens = self.student_model.model.embed_tokens
            
            # 冻结基础词表 (0-151642)
            base_vocab_size = BASE_VOCAB_END + 1  # 151643
            
            # 对embedding层进行部分冻结
            # 注意: nn.Embedding不支持直接对单个token的embedding进行冻结
            # 我们通过hook或自定义方式实现
            
            # 方法: 在训练时mask掉基础词表的梯度
            def freeze_base_vocab_hook(grad):
                """冻结基础词表梯度的hook"""
                grad_clone = grad.clone()
                # 冻结基础词表范围
                if grad_clone.shape[0] > BASE_VOCAB_END:
                    grad_clone[:BASE_VOCAB_END+1] = 0
                return grad_clone
            
            # 注册hook到embedding权重
            if hasattr(embed_tokens, 'weight'):
                embed_tokens.weight.register_hook(freeze_base_vocab_hook)
                print(f"✓ Base vocab (0-{BASE_VOCAB_END}) FROZEN via gradient hook")
                print(f"✓ Extended vocab ({EXTENDED_VOCAB_START}-{EXTENDED_VOCAB_END}) TRAINABLE")
        
        # 3. 冻结LLM的某些层 (可选，根据显存调整)
        # 默认不冻结LLM层，全部训练
        
        # 统计可训练参数
        trainable = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.student_model.parameters())
        
        # 计算扩展词表参数
        extended_vocab_size = EXTENDED_VOCAB_END - EXTENDED_VOCAB_START + 1
        embed_dim = self.student_model.config.text_config.hidden_size
        extended_embed_params = extended_vocab_size * embed_dim
        
        print(f"\nParameter summary:")
        print(f"  Total parameters: {total / 1e6:.0f}M")
        print(f"  Trainable parameters: {trainable / 1e6:.0f}M")
        print(f"  Frozen parameters: {(total - trainable) / 1e6:.0f}M")
        print(f"  Extended vocab embed params: {extended_embed_params / 1e6:.2f}M")
        print(f"  Trainable ratio: {trainable/total*100:.1f}%")
    
    def _build_optimizer(self):
        """构建优化器 - 只优化可训练参数"""
        params = [p for p in self.student_model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        print(f"\nOptimizer: AdamW")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print(f"  Weight decay: {self.config['weight_decay']}")
        print(f"  Trainable param groups: {len(params)}")
        
        return optimizer
    
    def get_teacher_logits(self, batch):
        """获取教师模型的logits"""
        # 如果使用缓存的teacher logits
        if self.config.get('use_cached_logits', False) and 'teacher_logits' in batch:
            return batch['teacher_logits'].to(self.device)
        
        # 否则实时计算 (需要教师模型)
        if self.teacher_model is None:
            return None
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        pixel_values = batch.get('pixel_values', None)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        
        image_grid_thw = batch.get('image_grid_thw', None)
        if image_grid_thw is not None:
            if isinstance(image_grid_thw, torch.Tensor):
                if image_grid_thw.dim() == 3 and image_grid_thw.shape[0] == 1:
                    image_grid_thw = image_grid_thw.squeeze(0)
                image_grid_thw = image_grid_thw.to(self.device).long()
        
        with torch.no_grad():
            outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        
        return outputs.logits
    
    def train_step(self, batch, phase=2):
        """单步训练
        
        Args:
            batch: 数据批次
            phase: 训练阶段 (1=CE预热, 2=CE+KL, 3=微调)
        """
        self.student_model.train()
        
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
        
        # 学生模型forward
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        student_logits = student_outputs.logits
        
        # 获取教师logits
        teacher_logits = None
        if phase >= 2:  # 阶段2和3使用KL loss
            teacher_logits = self.get_teacher_logits(batch)
        
        # 构建mask (忽略pad和-100的位置)
        # 使用attention_mask，但还需要考虑labels中的-100
        mask = (labels != -100).float()  # (batch, seq_len)
        # 对齐student_logits和labels (student_logits比labels多一个位置)
        if student_logits.shape[1] > labels.shape[1]:
            student_logits = student_logits[:, :-1, :]
        if teacher_logits is not None and teacher_logits.shape[1] > labels.shape[1]:
            teacher_logits = teacher_logits[:, :-1, :]
        
        # 计算蒸馏损失
        total_loss, kl_loss, ce_loss = self.distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            mask=mask,
            student_model=None,  # 我们自己计算了CE
        )
        
        # 如果distillation_loss没有计算CE，使用student_outputs的loss
        if ce_loss == 0.0 and hasattr(student_outputs, 'loss') and student_outputs.loss is not None:
            ce_loss = student_outputs.loss
            if phase >= 2 and teacher_logits is not None:
                total_loss = self.config['alpha'] * kl_loss + self.config['beta'] * ce_loss
            else:
                total_loss = ce_loss
        
        # 梯度累积
        if self.config['gradient_accumulation_steps'] > 1:
            total_loss = total_loss / self.config['gradient_accumulation_steps']
            if kl_loss != 0:
                kl_loss = kl_loss / self.config['gradient_accumulation_steps']
            if ce_loss != 0:
                ce_loss = ce_loss / self.config['gradient_accumulation_steps']
        
        # Backward
        total_loss.backward()
        
        return {
            'total_loss': total_loss.item(),
            'kl_loss': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            'ce_loss': ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss,
        }
    
    def train(self, train_dataloader, eval_dataloader=None):
        """主训练循环 - 三阶段训练"""
        print("=" * 70)
        print("Starting LLM Distillation Training")
        print("=" * 70)
        
        # 计算总步数
        steps_per_epoch = len(train_dataloader) // self.config['gradient_accumulation_steps']
        
        # ========== 阶段1: CE Loss 预热 ==========
        if self.config.get('phase1_enabled', True):
            phase1_epochs = self.config.get('phase1_warmup_epochs', 1)
            print(f"\n{'='*70}")
            print(f"PHASE 1: CE Loss Warmup ({phase1_epochs} epochs)")
            print(f"{'='*70}")
            
            self._train_phase(
                train_dataloader, 
                eval_dataloader,
                phase=1,
                epochs=phase1_epochs,
                desc="Phase1-Warmup"
            )
        
        # ========== 阶段2: CE + KL Loss 联合训练 ==========
        phase2_epochs = self.config.get('phase2_epochs', 2)
        print(f"\n{'='*70}")
        print(f"PHASE 2: CE + KL Distillation ({phase2_epochs} epochs)")
        print(f"  Temperature: {self.config['temperature']}")
        print(f"  Alpha (KL weight): {self.config['alpha']}")
        print(f"  Beta (CE weight): {self.config['beta']}")
        print(f"{'='*70}")
        
        self._train_phase(
            train_dataloader,
            eval_dataloader,
            phase=2,
            epochs=phase2_epochs,
            desc="Phase2-Distill"
        )
        
        # ========== 阶段3: 降低Temperature微调 ==========
        phase3_epochs = self.config.get('phase3_finetune_epochs', 1)
        phase3_temp = self.config.get('phase3_temperature', 1.0)
        phase3_lr = self.config.get('phase3_lr', 1e-5)
        
        print(f"\n{'='*70}")
        print(f"PHASE 3: Fine-tuning with lower temperature ({phase3_epochs} epochs)")
        print(f"  Temperature: {phase3_temp} (was {self.config['temperature']})")
        print(f"  Learning rate: {phase3_lr} (was {self.config['learning_rate']})")
        print(f"{'='*70}")
        
        # 更新蒸馏损失的温度
        self.distillation_loss.temperature = phase3_temp
        
        # 更新学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = phase3_lr
        
        self._train_phase(
            train_dataloader,
            eval_dataloader,
            phase=3,
            epochs=phase3_epochs,
            desc="Phase3-Finetune"
        )
        
        print("\n" + "=" * 70)
        print("Training completed!")
        print("=" * 70)
        self.save_checkpoint(is_final=True)
    
    def _train_phase(self, train_dataloader, eval_dataloader, phase, epochs, desc):
        """训练一个阶段"""
        # 初始化学习率调度器
        total_steps = len(train_dataloader) * epochs // self.config['gradient_accumulation_steps']
        if self.scheduler is None:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config['warmup_steps'],
                num_training_steps=total_steps,
            )
        
        for epoch in range(epochs):
            print(f"\n{desc} Epoch {epoch + 1}/{epochs}")
            
            # Training
            train_metrics = {
                'total_loss': 0.0,
                'kl_loss': 0.0,
                'ce_loss': 0.0,
            }
            progress_bar = tqdm(train_dataloader, desc=f"{desc}-E{epoch+1}")
            
            for step, batch in enumerate(progress_bar):
                metrics = self.train_step(batch, phase=phase)
                
                train_metrics['total_loss'] += metrics['total_loss']
                train_metrics['kl_loss'] += metrics['kl_loss']
                train_metrics['ce_loss'] += metrics['ce_loss']
                
                # 梯度更新
                if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        self.config['max_grad_norm']
                    )
                    
                    # 更新参数
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # 日志
                    if self.global_step % self.config['logging_steps'] == 0:
                        # 计算平均loss
                        avg_total = train_metrics['total_loss'] / self.config['logging_steps']
                        avg_kl = train_metrics['kl_loss'] / self.config['logging_steps']
                        avg_ce = train_metrics['ce_loss'] / self.config['logging_steps']
                        
                        # TensorBoard
                        self.writer.add_scalar(f"{desc}/loss", avg_total, self.global_step)
                        self.writer.add_scalar(f"{desc}/kl_loss", avg_kl, self.global_step)
                        self.writer.add_scalar(f"{desc}/ce_loss", avg_ce, self.global_step)
                        self.writer.add_scalar(f"{desc}/lr", self.scheduler.get_last_lr()[0], self.global_step)
                        self.writer.add_scalar(f"{desc}/temperature", self.distillation_loss.temperature, self.global_step)
                        
                        progress_bar.set_postfix({
                            "loss": f"{avg_total:.4f}",
                            "kl": f"{avg_kl:.4f}",
                            "ce": f"{avg_ce:.4f}",
                            "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                            "T": f"{self.distillation_loss.temperature:.1f}",
                        })
                        
                        # 重置累计
                        train_metrics = {k: 0.0 for k in train_metrics}
                    
                    # 保存
                    if self.global_step % self.config['save_steps'] == 0:
                        self.save_checkpoint()
                    
                    # 评估
                    if eval_dataloader and self.global_step % self.config['eval_steps'] == 0:
                        eval_loss = self.evaluate(eval_dataloader, phase=phase)
                        self.writer.add_scalar(f"{desc}/eval_loss", eval_loss, self.global_step)
                        print(f"\nEval loss: {eval_loss:.4f}")
                        
                        # 保存最佳模型
                        if eval_loss < self.best_loss:
                            self.best_loss = eval_loss
                            self.save_checkpoint(is_best=True)
            
            # Epoch结束保存
            self.save_checkpoint()
    
    def evaluate(self, eval_dataloader, phase=2):
        """评估"""
        self.student_model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # 获取教师logits
                teacher_logits = None
                if phase >= 2:
                    teacher_logits = self.get_teacher_logits(batch)
                
                # 学生模型forward
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
                
                student_outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
                
                # 计算损失
                student_logits = student_outputs.logits
                mask = (labels != -100).float()
                
                if student_logits.shape[1] > labels.shape[1]:
                    student_logits = student_logits[:, :-1, :]
                if teacher_logits is not None and teacher_logits.shape[1] > labels.shape[1]:
                    teacher_logits = teacher_logits[:, :-1, :]
                
                total_loss_batch, _, _ = self.distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    mask=mask,
                )
                
                total_loss += total_loss_batch.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def precompute_teacher_logits(self, dataloader, output_dir):
        """预计算教师模型的logits并保存到磁盘"""
        if self.teacher_model is None:
            print("ERROR: Teacher model not loaded. Cannot precompute logits.")
            return
        
        print(f"\n{'='*70}")
        print("Precomputing teacher logits...")
        print(f"{'='*70}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.teacher_model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Precomputing")):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                pixel_values = batch.get('pixel_values', None)
                if pixel_values is not None:
                    pixel_values = pixel_values.to(self.device)
                
                image_grid_thw = batch.get('image_grid_thw', None)
                if image_grid_thw is not None:
                    if isinstance(image_grid_thw, torch.Tensor):
                        if image_grid_thw.dim() == 3 and image_grid_thw.shape[0] == 1:
                            image_grid_thw = image_grid_thw.squeeze(0)
                        image_grid_thw = image_grid_thw.to(self.device).long()
                
                outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
                
                # 保存logits (在CPU上)
                logits = outputs.logits.cpu()
                
                # 逐个样本保存 (支持不同长度)
                batch_size = logits.shape[0]
                for j in range(batch_size):
                    sample_idx = i * dataloader.batch_size + j
                    sample_logits = logits[j]  # (seq_len, vocab_size)
                    
                    # 保存为fp16节省空间
                    torch.save(sample_logits.half(), output_path / f"sample_{sample_idx}.pt")
        
        print(f"Teacher logits saved to: {output_dir}")
        print(f"Total samples: {len(dataloader.dataset)}")
    
    def save_checkpoint(self, is_best=False, is_final=False):
        """保存检查点"""
        if is_best:
            save_dir = self.output_dir / "best"
        elif is_final:
            save_dir = self.output_dir / "final"
        else:
            save_dir = self.output_dir / f"checkpoint-{self.global_step}"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存学生模型
        self.student_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # 保存训练状态
        state = {
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "config": self.config,
            "current_phase": self.current_phase,
            "current_epoch": self.current_epoch,
        }
        with open(save_dir / "trainer_state.json", 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Checkpoint saved: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="LLM Distillation Training v2")
    
    # 基本参数
    parser.add_argument("--dataset", default=DATASET_PATH, help="Path to distillation dataset")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--num-gpus", type=int, default=3, help="Number of GPUs")
    parser.add_argument("--epochs", type=int, default=3, help="Total number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples for debugging")
    
    # 蒸馏参数
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.7, help="KL loss weight")
    parser.add_argument("--beta", type=float, default=0.3, help="CE loss weight")
    parser.add_argument("--phase3-temp", type=float, default=1.0, help="Phase 3 temperature")
    parser.add_argument("--phase3-lr", type=float, default=1e-5, help="Phase 3 learning rate")
    
    # 阶段控制
    parser.add_argument("--phase1-epochs", type=int, default=1, help="Phase 1 warmup epochs")
    parser.add_argument("--phase2-epochs", type=int, default=2, help="Phase 2 distillation epochs")
    parser.add_argument("--phase3-epochs", type=int, default=1, help="Phase 3 finetune epochs")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip phase 1 warmup")
    
    # 预计算teacher logits
    parser.add_argument("--precompute-teacher", action="store_true", help="Precompute teacher logits")
    parser.add_argument("--use-cached-logits", action="store_true", help="Use cached teacher logits")
    parser.add_argument("--teacher-logits-dir", default=TEACHER_LOGITS_DIR, help="Teacher logits cache dir")
    
    args = parser.parse_args()
    
    # 构建配置
    config = {
        "batch_size": args.batch_size,
        "num_gpus": args.num_gpus,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "output_dir": args.output_dir,
        "max_samples": args.max_samples,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "beta": args.beta,
        "phase3_temperature": args.phase3_temp,
        "phase3_lr": args.phase3_lr,
        "phase1_warmup_epochs": args.phase1_epochs,
        "phase2_epochs": args.phase2_epochs,
        "phase3_finetune_epochs": args.phase3_epochs,
        "phase1_enabled": not args.skip_phase1,
        "precompute_teacher_logits": args.precompute_teacher,
        "use_cached_logits": args.use_cached_logits,
        "teacher_logits_dir": args.teacher_logits_dir,
    }
    
    # 创建数据集
    print("Loading dataset...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(STUDENT_PATH, trust_remote_code=True)
    
    dataset = LLMDistillationDataset(
        dataset_path=args.dataset,
        tokenizer=tokenizer,
        processor=processor,
        max_samples=args.max_samples,
        teacher_logits_dir=args.teacher_logits_dir,
        use_cached_logits=args.use_cached_logits,
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
    
    # 预计算teacher logits (如果需要)
    if args.precompute_teacher:
        trainer.precompute_teacher_logits(train_dataloader, args.teacher_logits_dir)
        print("Precomputation done. Exiting.")
        return
    
    # 开始训练
    trainer.train(train_dataloader, eval_dataloader)


if __name__ == "__main__":
    main()
