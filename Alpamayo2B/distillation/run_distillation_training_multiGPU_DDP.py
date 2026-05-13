#!/usr/bin/env python3
"""
Alpamayo2B 蒸馏训练启动脚本（多GPU DDP版本）

使用方法:
    # 使用4张GPU (3,4,5,7)
    torchrun --nproc_per_node=4 --master_addr=localhost --master_port=29500 \
        run_distillation_training_multiGPU_DDP.py --gpu-list 3,4,5,7
    
    # 使用所有8张GPU
    torchrun --nproc_per_node=8 --master_addr=localhost --master_port=29500 \
        run_distillation_training_multiGPU_DDP.py --gpu-list 0,1,2,3,4,5,6,7

配置说明:
    - 使用 DistributedDataParallel (DDP) 实现真正的多GPU并行
    - 不加载教师模型（使用预计算 logits）
    - 冻结 ViT 和基础词表
    - 训练扩展词表和 LLM
"""

import sys
import argparse
import os

sys.path.insert(0, '/home/user/mikelee/alpamayo_project/Alpamayo2B/distillation')

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def parse_gpu_list(gpu_string):
    """解析GPU列表字符串，如 '3,4,5,7' -> [3, 4, 5, 7]"""
    try:
        gpu_list = [int(x.strip()) for x in gpu_string.split(',')]
        available_gpus = list(range(torch.cuda.device_count()))
        for gpu in gpu_list:
            if gpu not in available_gpus:
                raise ValueError(f"GPU {gpu} 不可用。可用GPU: {available_gpus}")
        return gpu_list
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"无效的GPU列表: {e}")

def setup_distributed(rank, world_size, gpu_list):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # 设置当前进程使用的GPU
    torch.cuda.set_device(gpu_list[rank])
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    print(f"[Rank {rank}] Initialized on GPU {gpu_list[rank]}")

def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='Alpamayo2B 多GPU DDP蒸馏训练')
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
    
    # 获取分布式信息
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    gpu_list = args.gpu_list
    
    # 验证GPU数量匹配
    if world_size != len(gpu_list):
        raise ValueError(
            f"进程数 ({world_size}) 与GPU列表长度 ({len(gpu_list)}) 不匹配。\n"
            f"请使用: torchrun --nproc_per_node={len(gpu_list)} ..."
        )
    
    # 设置当前GPU
    current_gpu = gpu_list[local_rank]
    torch.cuda.set_device(current_gpu)
    
    # 只在主进程打印信息
    if rank == 0:
        print("=" * 70)
        print("Alpamayo2B Distillation Training (Multi-GPU DDP)")
        print("=" * 70)
        print(f"GPU List: {gpu_list}")
        print(f"World Size: {world_size}")
        print(f"Per-GPU Batch size: {args.batch_size}")
        print(f"Total Effective Batch size: {args.batch_size * world_size}")
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
        "gradient_accumulation_steps": 4,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        
        # 蒸馏参数
        "temperature": 2.0,
        "alpha": 0.7,
        "beta": 0.3,
        
        # 日志和保存
        "save_steps": 5000,
        "eval_steps": 10000,
        "logging_steps": 100,
        
        # 其他
        "seed": 42,
        "max_samples": None,
        "num_workers": 4,
        "device": f"cuda:{current_gpu}",
        "dtype": "bfloat16",
        
        # DDP配置
        "multi_gpu": True,
        "gpu_list": gpu_list,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
    }
    
    # 导入训练器
    from train_distillation_Alpamayo2B import DistillationTrainer
    
    # 创建训练器
    trainer = DistillationTrainer(config)
    
    # 使用DDP包装模型
    if world_size > 1:
        if rank == 0:
            print(f"\nWrapping model with DistributedDataParallel...")
        
        trainer.student_model = DDP(
            trainer.student_model,
            device_ids=[current_gpu],
            output_device=current_gpu,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        
        if rank == 0:
            print("✓ Model wrapped with DDP")
    
    # 开始训练
    trainer.train()
    
    # 清理
    cleanup_distributed()
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("Training completed!")
        print("=" * 70)

if __name__ == "__main__":
    main()
