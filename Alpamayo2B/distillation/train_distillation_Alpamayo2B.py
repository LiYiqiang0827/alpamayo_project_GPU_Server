"""
Alpamayo2B 蒸馏训练脚本
=======================
用于 Alpamayo1.5-10B → Alpamayo2B 的 LLM 知识蒸馏训练

主要功能：
1. 蒸馏损失函数（KL散度 + 交叉熵）
2. 训练循环（支持多GPU、梯度累积、学习率调度）
3. 检查点管理和训练监控

作者: 小胖龟
创建时间: 2026-05-12
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

# 导入自定义模块
from dataloader_Alpamayo2B import (
    AlpamayoDistillationDataset,
    collate_fn,
    create_distillation_dataloader,
)
from model_setup_Alpamayo2B import (
    setup_model_for_distillation,
    verify_freeze_status,
)

logger = logging.getLogger(__name__)

# ==================== 配置 ====================
MODEL_PATH = "/data01/mikelee/weight/alpamayo2B"
TOKENIZER_PATH = "/data01/mikelee/weight/alpamayo2B/tokenizer_final"
TEACHER_MODEL_PATH = "/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B"

# 词表范围配置
BASE_VOCAB_START = 0
BASE_VOCAB_END = 151642
EXTENDED_VOCAB_START = 151669
EXTENDED_VOCAB_END = 155696

# 默认训练配置
DEFAULT_CONFIG = {
    # 数据配置
    "infer_result_csv": "/gpfs-data/mikelee/infer_result/infer_result_20260507_195923/infer_results_all.csv",
    "teacher_logits_dir": "/gpfs-data/mikelee/infer_result/infer_result_20260507_195923/logits",
    
    # 模型配置
    "model_path": MODEL_PATH,
    "tokenizer_path": TOKENIZER_PATH,
    
    # 训练配置
    "output_dir": "/data02/mikelee/Alpamayo2B_distill_output",
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "num_epochs": 3,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    
    # 蒸馏配置
    "temperature": 2.0,
    "alpha": 0.7,  # KL loss权重
    "beta": 0.3,   # CE loss权重
    
    # 保存配置
    "save_steps": 500,
    "eval_steps": 250,
    "logging_steps": 10,
    
    # 其他
    "seed": 42,
    "max_samples": None,  # None表示使用全部数据
    "num_workers": 4,
    "device": "cuda:1",
    "dtype": "bfloat16",
}


# ==================== 蒸馏损失函数 ====================
class DistillationLoss(nn.Module):
    """
    知识蒸馏损失函数 - KL散度 + CE
    
    支持：
    - Soft Loss: KL散度（学习教师模型的概率分布）
    - Hard Loss: 交叉熵（学习硬标签）
    - 温度缩放
    - Mask支持（忽略pad和-100）
    """
    
    def __init__(self, temperature: float = 2.0, alpha: float = 0.7, beta: float = 0.3):
        """
        Args:
            temperature: 蒸馏温度
            alpha: KL loss权重
            beta: CE loss权重
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.ce_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: Optional[torch.Tensor],
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算蒸馏损失
        
        Args:
            student_logits: (batch, seq_len, vocab_size) 学生模型logits
            teacher_logits: (batch, seq_len, vocab_size) 教师模型logits
            labels: (batch, seq_len) 真实标签
            attention_mask: (batch, seq_len) 注意力mask
            
        Returns:
            total_loss: 总损失
            kl_loss: KL散度损失
            ce_loss: 交叉熵损失
        """
        batch_size, seq_len, vocab_size = student_logits.shape
        
        # 1. CE Loss (使用原始logits)
        # Shift for next token prediction
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        ce_loss = self.ce_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # 2. KL Loss (使用temperature缩放)
        kl_loss = torch.tensor(0.0, device=student_logits.device)
        
        if teacher_logits is not None:
            # 确保teacher和student的seq_len一致
            # student_logits: (batch, seq_len_student, vocab)
            # teacher_logits: (batch, seq_len_teacher, vocab)
            # 取两者中较短的seq_len
            student_seq_len = student_logits.size(1)
            teacher_seq_len = teacher_logits.size(1)
            min_seq_len = min(student_seq_len, teacher_seq_len)
            
            # 截断到相同长度
            s_logits = student_logits[:, :min_seq_len, :]
            t_logits = teacher_logits[:, :min_seq_len, :]
            
            # 确保teacher和student的vocab大小一致
            min_vocab = min(s_logits.size(-1), t_logits.size(-1))
            s_logits = s_logits[..., :min_vocab]
            t_logits = t_logits[..., :min_vocab]
            
            # 应用temperature
            s_logits_scaled = s_logits / self.temperature
            t_logits_scaled = t_logits / self.temperature
            
            # 计算soft labels
            student_probs = F.log_softmax(s_logits_scaled, dim=-1)
            teacher_probs = F.softmax(t_logits_scaled, dim=-1)
            
            # 创建mask（忽略pad和-100）
            if attention_mask is not None:
                # 截断mask到相同长度
                mask_truncated = attention_mask[:, :min_seq_len]
                # 扩展mask到vocab维度
                mask_expanded = mask_truncated.unsqueeze(-1)  # (batch, min_seq_len, 1)
                
                # 应用mask
                student_probs_masked = student_probs * mask_expanded
                teacher_probs_masked = teacher_probs * mask_expanded
                
                # 计算KL散度
                kl_per_token = F.kl_div(
                    student_probs_masked,
                    teacher_probs_masked,
                    reduction='none',
                    log_target=False,
                ).sum(dim=-1)  # (batch, min_seq_len)
                
                # 只对有效位置求和
                kl_loss = (kl_per_token * mask_truncated).sum() / mask_truncated.sum().clamp(min=1)
            else:
                # 无mask，对整个序列计算
                kl_loss = F.kl_div(
                    student_probs.view(-1, min_vocab),
                    teacher_probs.view(-1, min_vocab),
                    reduction='batchmean',
                    log_target=False,
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
class DistillationTrainer:
    """
    蒸馏训练器
    
    支持：
    - 单GPU训练
    - 梯度累积
    - 学习率调度（warmup + linear decay）
    - 混合精度训练
    - TensorBoard日志
    - 检查点保存
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 训练配置字典
        """
        self.config = {**DEFAULT_CONFIG, **config}
        self.device = torch.device(self.config["device"])
        
        # 设置随机种子
        self._set_seed(self.config["seed"])
        
        # 创建输出目录（带时间戳子文件夹）
        base_output_dir = Path(self.config["output_dir"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = base_output_dir / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {self.output_dir}")
        
        # 保存配置
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / "logs")
        
        # 日志文件设置
        log_dir = Path("/home/user/mikelee/alpamayo_project/Alpamayo2B/distillation/log_distill")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"logDistill_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger = logging.getLogger("distillation")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        
        self.logger.info("=" * 70)
        self.logger.info("Alpamayo2B Distillation Training Started")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info("=" * 70)
        
        # 验证集最佳损失
        self.best_val_loss = float("inf")
        
        # 加载模型
        self._setup_models()
        
        # 加载数据
        self._setup_dataloader()
        
        # 设置优化器和学习率调度
        self._setup_optimizer()
        
        # 损失函数
        self.distillation_loss = DistillationLoss(
            temperature=self.config["temperature"],
            alpha=self.config["alpha"],
            beta=self.config["beta"],
        )
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _setup_models(self):
        """加载和设置模型"""
        print("=" * 70)
        print("Setting up models...")
        print("=" * 70)
        
        # 加载tokenizer和processor
        print("\nLoading tokenizer and processor...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["tokenizer_path"],
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.config["model_path"],
            trust_remote_code=True,
        )
        
        # 转换dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map[self.config["dtype"]]
        
        # 加载学生模型（冻结ViT和基础词表）
        print("\nLoading student model...")
        self.student_model, _, _ = setup_model_for_distillation(
            model_path=self.config["model_path"],
            tokenizer_path=self.config["tokenizer_path"],
            device=self.config["device"],
            dtype=dtype,
        )
        
        # 不加载教师模型，直接使用预计算的logits
        print("\nUsing pre-computed teacher logits (no teacher model loaded)")
        self.teacher_model = None
        
        print("\n✓ Models setup complete!")
    
    def _setup_dataloader(self):
        """设置数据加载器（训练集 + 验证集）"""
        print("\nSetting up dataloaders...")
        
        # 训练集
        self.train_dataloader = create_distillation_dataloader(
            infer_result_csv=self.config["infer_result_csv"],
            teacher_logits_dir=self.config["teacher_logits_dir"],
            tokenizer=self.tokenizer,
            processor=self.processor,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            temperature=self.config["temperature"],
            shuffle=True,
            split="train",
        )
        
        # 验证集
        self.val_dataloader = create_distillation_dataloader(
            infer_result_csv=self.config["infer_result_csv"],
            teacher_logits_dir=self.config["teacher_logits_dir"],
            tokenizer=self.tokenizer,
            processor=self.processor,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            temperature=self.config["temperature"],
            shuffle=False,  # 验证集不打乱
            split="val",
        )
        
        print(f"✓ Train dataloader: {len(self.train_dataloader)} batches")
        print(f"✓ Val dataloader: {len(self.val_dataloader)} batches")
    
    def _setup_optimizer(self):
        """设置优化器和学习率调度"""
        # 只优化可训练参数
        trainable_params = [p for p in self.student_model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        
        # 计算总步数
        total_steps = len(self.train_dataloader) * self.config["num_epochs"]
        warmup_steps = int(total_steps * self.config["warmup_ratio"])
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        print(f"\nOptimizer setup:")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print(f"  Weight decay: {self.config['weight_decay']}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
    
    def train(self):
        """主训练循环"""
        # 检查是否是DDP模式
        is_ddp = self.config.get("multi_gpu", False) and self.config.get("world_size", 1) > 1
        rank = self.config.get("rank", 0)
        world_size = self.config.get("world_size", 1)
        
        if rank == 0:
            print("\n" + "=" * 70)
            print("Starting training...")
            print("=" * 70)
        
        self.student_model.train()
        
        for epoch in range(self.config["num_epochs"]):
            self.epoch = epoch
            if rank == 0:
                print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
                print("-" * 70)
            
            epoch_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_ce_loss = 0.0
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # 将数据移到设备
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                pixel_values = batch["pixel_values"].to(self.device) if "pixel_values" in batch else None
                image_grid_thw = batch["image_grid_thw"].to(self.device) if "image_grid_thw" in batch else None
                teacher_logits = batch["teacher_logits"].to(self.device) if "teacher_logits" in batch else None
                
                # 前向传播
                outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=None,  # 我们自己计算loss
                )
                
                student_logits = outputs.logits
                
                # 计算损失
                loss, kl_loss, ce_loss = self.distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    attention_mask=attention_mask,
                )
                
                # 梯度累积
                loss = loss / self.config["gradient_accumulation_steps"]
                loss.backward()
                
                # 更新权重
                if (step + 1) % self.config["gradient_accumulation_steps"] == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        self.config["max_grad_norm"],
                    )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # 记录日志
                    if self.global_step % self.config["logging_steps"] == 0:
                        self._log_metrics(loss, kl_loss, ce_loss)
                    
                    # 保存检查点
                    if self.global_step % self.config["save_steps"] == 0:
                        self._save_checkpoint()
                
                # 累加epoch损失
                epoch_loss += loss.item() * self.config["gradient_accumulation_steps"]
                epoch_kl_loss += kl_loss.item()
                epoch_ce_loss += ce_loss.item()
                
                # 更新进度条
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "kl": f"{kl_loss.item():.4f}",
                    "ce": f"{ce_loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                })
            
            # Epoch结束
            avg_loss = epoch_loss / len(self.train_dataloader)
            avg_kl_loss = epoch_kl_loss / len(self.train_dataloader)
            avg_ce_loss = epoch_ce_loss / len(self.train_dataloader)
            
            if rank == 0:
                print(f"\nEpoch {epoch + 1} Train Summary:")
                print(f"  Average loss: {avg_loss:.4f}")
                print(f"  Average KL loss: {avg_kl_loss:.4f}")
                print(f"  Average CE loss: {avg_ce_loss:.4f}")
            
            # 记录到 TensorBoard
            self.writer.add_scalar("Epoch/Train_Loss", avg_loss, epoch + 1)
            self.writer.add_scalar("Epoch/Train_KL", avg_kl_loss, epoch + 1)
            self.writer.add_scalar("Epoch/Train_CE", avg_ce_loss, epoch + 1)
            
            # 验证
            val_loss, val_kl, val_ce = self._evaluate()
            if rank == 0:
                print(f"\nEpoch {epoch + 1} Val Summary:")
                print(f"  Average loss: {val_loss:.4f}")
                print(f"  Average KL loss: {val_kl:.4f}")
                print(f"  Average CE loss: {val_ce:.4f}")
            
            # 记录到 TensorBoard
            self.writer.add_scalar("Epoch/Val_Loss", val_loss, epoch + 1)
            self.writer.add_scalar("Epoch/Val_KL", val_kl, epoch + 1)
            self.writer.add_scalar("Epoch/Val_CE", val_ce, epoch + 1)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(suffix="best")
                print(f"  ✓ New best model! Val loss: {val_loss:.4f}")
            
            # 保存epoch检查点
            self._save_checkpoint(suffix=f"epoch_{epoch + 1}")
        
        print("\n" + "=" * 70)
        print("Training complete!")
        print("=" * 70)
        
        # 保存最终模型
        self._save_final_model()
    
    def _log_metrics(self, loss: torch.Tensor, kl_loss: torch.Tensor, ce_loss: torch.Tensor):
        """记录指标到TensorBoard和日志文件"""
        lr = self.scheduler.get_last_lr()[0]
        
        self.writer.add_scalar("Loss/total", loss.item(), self.global_step)
        self.writer.add_scalar("Loss/kl", kl_loss.item(), self.global_step)
        self.writer.add_scalar("Loss/ce", ce_loss.item(), self.global_step)
        self.writer.add_scalar("LR", lr, self.global_step)
        
        self.logger.info(
            f"Step {self.global_step} | Epoch {self.epoch + 1} | "
            f"Loss: {loss.item():.6f} | KL: {kl_loss.item():.6f} | "
            f"CE: {ce_loss.item():.6f} | LR: {lr:.2e}"
        )
    
    def _save_checkpoint(self, suffix: str = ""):
        """保存检查点"""
        # 只在主进程保存
        rank = self.config.get("rank", 0)
        if rank != 0:
            return
            
        checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
        if suffix:
            checkpoint_dir = self.output_dir / f"checkpoint-{suffix}"
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        self.student_model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # 保存训练状态
        checkpoint_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        torch.save(checkpoint_state, checkpoint_dir / "training_state.pt")
        
        print(f"\n✓ Checkpoint saved to {checkpoint_dir}")
    
    @torch.no_grad()
    def _evaluate(self):
        """在验证集上评估模型"""
        self.student_model.eval()
        
        total_loss = 0.0
        total_kl_loss = 0.0
        total_ce_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.val_dataloader, desc="Validation", leave=False)
        
        for batch in progress_bar:
            # 将数据移到设备
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            pixel_values = batch["pixel_values"].to(self.device) if "pixel_values" in batch else None
            image_grid_thw = batch["image_grid_thw"].to(self.device) if "image_grid_thw" in batch else None
            teacher_logits = batch["teacher_logits"].to(self.device) if "teacher_logits" in batch else None
            
            # 前向传播
            outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=None,
            )
            
            student_logits = outputs.logits
            
            # 计算损失
            loss, kl_loss, ce_loss = self.distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                attention_mask=attention_mask,
            )
            
            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
            total_ce_loss += ce_loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
            })
        
        self.student_model.train()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_kl = total_kl_loss / num_batches if num_batches > 0 else 0
        avg_ce = total_ce_loss / num_batches if num_batches > 0 else 0
        
        return avg_loss, avg_kl, avg_ce
    
    def _save_final_model(self):
        """保存最终模型"""
        rank = self.config.get("rank", 0)
        if rank != 0:
            return
            
        final_dir = self.output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        self.student_model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        
        print(f"\n✓ Final model saved to {final_dir}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="Train Alpamayo2B distillation")
    
    # 数据配置
    parser.add_argument("--infer-csv", default=DEFAULT_CONFIG["infer_result_csv"])
    parser.add_argument("--logits-dir", default=DEFAULT_CONFIG["teacher_logits_dir"])
    
    # 模型配置
    parser.add_argument("--model-path", default=DEFAULT_CONFIG["model_path"])
    parser.add_argument("--tokenizer-path", default=DEFAULT_CONFIG["tokenizer_path"])
    parser.add_argument("--teacher-path", default=DEFAULT_CONFIG["teacher_model_path"])
    
    # 训练配置
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_CONFIG["gradient_accumulation_steps"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_CONFIG["warmup_ratio"])
    
    # 蒸馏配置
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"])
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG["alpha"])
    parser.add_argument("--beta", type=float, default=DEFAULT_CONFIG["beta"])
    
    # 其他
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--max-samples", type=int, default=DEFAULT_CONFIG["max_samples"])
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"])
    parser.add_argument("--dtype", default=DEFAULT_CONFIG["dtype"], choices=["float16", "bfloat16", "float32"])
    
    args = parser.parse_args()
    
    # 构建配置
    config = {
        "infer_result_csv": args.infer_csv,
        "teacher_logits_dir": args.logits_dir,
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path,
        "teacher_model_path": args.teacher_path,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "beta": args.beta,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "num_workers": args.num_workers,
        "device": args.device,
        "dtype": args.dtype,
    }
    
    # 创建训练器并开始训练
    trainer = DistillationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
