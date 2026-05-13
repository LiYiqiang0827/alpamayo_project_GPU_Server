#!/usr/bin/env python3
"""
Alpamayo2B 蒸馏训练启动脚本（正式版）

使用方法:
    python3 run_distillation_training.py

配置说明:
    - 使用 GPU1 (cuda:1)
    - 不加载教师模型（使用预计算 logits）
    - 冻结 ViT 和基础词表
    - 训练扩展词表和 LLM
"""

import sys
sys.path.insert(0, '/home/user/mikelee/alpamayo_project/Alpamayo2B/distillation')

from train_distillation_Alpamayo2B import DistillationTrainer

# 训练配置（正式版）
config = {
    # 数据路径
    "infer_result_csv": "/gpfs-data/mikelee/infer_result/infer_result_20260507_195923/infer_results_all.csv",
    "teacher_logits_dir": "/gpfs-data/mikelee/infer_result/infer_result_20260507_195923/logits",
    
    # 模型路径
    "model_path": "/data01/mikelee/weight/alpamayo2B",
    "tokenizer_path": "/data01/mikelee/weight/alpamayo2B/tokenizer_final",
    
    # 输出路径
    "output_dir": "/data02/mikelee/Alpamayo2B_distill_output",
    
    # 训练参数
    "batch_size": 1,
    "gradient_accumulation_steps": 4,  # 等效 batch_size=4
    "num_epochs": 3,  # 训练 3 个 epoch
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    
    # 蒸馏参数
    "temperature": 2.0,
    "alpha": 0.7,  # KL 损失权重
    "beta": 0.3,   # CE 损失权重
    
    # 日志和保存
    "save_steps": 5000,   # 每 5000 步保存检查点
    "eval_steps": 10000,  # 每 10000 步评估
    "logging_steps": 100, # 每 100 步记录日志
    
    # 其他
    "seed": 42,
    "max_samples": None,  # None 表示使用所有样本
    "num_workers": 4,
    "device": "cuda:1",  # 使用 GPU1
    "dtype": "bfloat16",
}

if __name__ == "__main__":
    print("=" * 70)
    print("Alpamayo2B Distillation Training (Production)")
    print("=" * 70)
    print(f"Device: {config['device']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']} (accumulation: {config['gradient_accumulation_steps']})")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Temperature: {config['temperature']}")
    print(f"Output: {config['output_dir']}")
    print("=" * 70)
    
    # 创建训练器
    trainer = DistillationTrainer(config)
    
    # 开始训练
    trainer.train()
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
