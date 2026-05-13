#!/usr/bin/env python3
"""
Alpamayo2B 蒸馏训练启动脚本（多GPU版本）

使用方法:
    python3 run_distillation_training_multiGPU.py --gpu-list 3,4,5,7
    python3 run_distillation_training_multiGPU.py --gpu-list 0,1,2,3,4,5,6,7

配置说明:
    - 支持多GPU并行训练（DataParallel）
    - 不加载教师模型（使用预计算 logits）
    - 冻结 ViT 和基础词表
    - 训练扩展词表和 LLM
"""

import sys
import argparse
import os

sys.path.insert(0, '/home/user/mikelee/alpamayo_project/Alpamayo2B/distillation')

from train_distillation_Alpamayo2B import DistillationTrainer
import torch

def parse_gpu_list(gpu_string):
    """解析GPU列表字符串，如 '3,4,5,7' -> [3, 4, 5, 7]"""
    try:
        gpu_list = [int(x.strip()) for x in gpu_string.split(',')]
        # 验证GPU是否可用
        available_gpus = list(range(torch.cuda.device_count()))
        for gpu in gpu_list:
            if gpu not in available_gpus:
                raise ValueError(f"GPU {gpu} 不可用。可用GPU: {available_gpus}")
        return gpu_list
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"无效的GPU列表: {e}")

def main():
    parser = argparse.ArgumentParser(description='Alpamayo2B 多GPU蒸馏训练')
    parser.add_argument(
        '--gpu-list',
        type=parse_gpu_list,
        required=True,
        help='使用的GPU列表，如 "3,4,5,7" 或 "0,1,2,3"'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='每个GPU的batch size（默认: 1）'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='训练轮数（默认: 3）'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-5,
        help='学习率（默认: 5e-5）'
    )
    
    args = parser.parse_args()
    
    gpu_list = args.gpu_list
    primary_gpu = gpu_list[0]  # 主GPU用于日志等操作
    
    # 设置可见GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_list))
    
    print("=" * 70)
    print("Alpamayo2B Distillation Training (Multi-GPU)")
    print("=" * 70)
    print(f"GPU List: {gpu_list}")
    print(f"Primary GPU: cuda:{primary_gpu}")
    print(f"Total GPUs: {len(gpu_list)}")
    print(f"Per-GPU Batch size: {args.batch_size}")
    print(f"Total Effective Batch size: {args.batch_size * len(gpu_list)}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print("=" * 70)
    
    # 训练配置
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
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": 4,  # 等效 batch_size=4 per GPU
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
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
        "device": f"cuda:{primary_gpu}",  # 使用主GPU
        "dtype": "bfloat16",
        
        # 多GPU配置
        "multi_gpu": True,
        "gpu_list": gpu_list,
    }
    
    # 创建训练器
    trainer = DistillationTrainer(config)
    
    # 使用DataParallel包装模型
    if len(gpu_list) > 1:
        print(f"\nWrapping model with DataParallel across GPUs: {gpu_list}")
        trainer.student_model = torch.nn.DataParallel(
            trainer.student_model,
            device_ids=list(range(len(gpu_list)))  # 使用相对索引 0,1,2...
        )
        print("✓ Model wrapped with DataParallel")
    
    # 开始训练
    trainer.train()
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
