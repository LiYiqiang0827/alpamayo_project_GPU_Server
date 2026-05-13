"""
Alpamayo2B 模型准备脚本
======================
用于蒸馏任务的模型加载和参数冻结

主要功能：
1. 加载 Alpamayo2B 模型
2. 冻结 ViT 部分
3. 冻结基础词表 (0-151642)
4. 确认扩展词表和 LLM 可训练
5. 验证冻结状态

作者: 小胖龟
创建时间: 2026-05-12
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

logger = logging.getLogger(__name__)

# ==================== 配置 ====================
MODEL_PATH = "/data01/mikelee/weight/alpamayo2B"
TOKENIZER_PATH = "/data01/mikelee/weight/alpamayo2B/tokenizer_final"

# 词表范围配置
BASE_VOCAB_START = 0
BASE_VOCAB_END = 151642       # 基础词表范围 [0, 151642]
EXTENDED_VOCAB_START = 151669  # 扩展词表起始
EXTENDED_VOCAB_END = 155696    # 扩展词表结束 [151669, 155696]


def load_model_and_tokenizer(
    model_path: str = MODEL_PATH,
    tokenizer_path: str = TOKENIZER_PATH,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor]:
    """
    加载 Alpamayo2B 模型、tokenizer 和 processor
    
    Args:
        model_path: 模型权重路径
        tokenizer_path: tokenizer 路径
        device: 设备 (cuda/cpu)
        dtype: 数据类型
        
    Returns:
        model: 加载的模型
        tokenizer: tokenizer
        processor: processor
    """
    print(f"Loading model from {model_path}...")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    print(f"Tokenizer loaded: {len(tokenizer)} tokens")
    
    # 加载 processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    print("Processor loaded")
    
    # 加载模型
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params / 1e6:.0f}M parameters")
    
    return model, tokenizer, processor


def freeze_vit(model: Qwen3VLForConditionalGeneration) -> None:
    """
    冻结 ViT (视觉编码器) 部分
    
    Args:
        model: Alpamayo2B 模型
    """
    if hasattr(model, 'visual'):
        for param in model.visual.parameters():
            param.requires_grad = False
        print("✓ Vision Encoder (ViT) FROZEN")
    else:
        print("⚠ Warning: model.visual not found")


def freeze_base_vocab(model: Qwen3VLForConditionalGeneration) -> None:
    """
    冻结基础词表 (0-151642)
    通过梯度 hook 实现部分冻结
    
    Args:
        model: Alpamayo2B 模型
    """
    # 获取 embedding 层
    embed_tokens = None
    if hasattr(model, 'model'):
        if hasattr(model.model, 'embed_tokens'):
            embed_tokens = model.model.embed_tokens
        elif hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'embed_tokens'):
            embed_tokens = model.model.language_model.embed_tokens
        
        # 创建梯度 mask
        base_vocab_size = BASE_VOCAB_END + 1  # 151643
        
        def freeze_base_vocab_hook(grad):
            """冻结基础词表梯度的 hook"""
            grad_clone = grad.clone()
            if grad_clone.shape[0] > BASE_VOCAB_END:
                grad_clone[:BASE_VOCAB_END + 1] = 0
            return grad_clone
        
        # 注册 hook
        if embed_tokens is not None and hasattr(embed_tokens, 'weight'):
            embed_tokens.weight.register_hook(freeze_base_vocab_hook)
            print(f"✓ Base vocab (0-{BASE_VOCAB_END}) FROZEN via gradient hook")
            print(f"✓ Extended vocab ({EXTENDED_VOCAB_START}-{EXTENDED_VOCAB_END}) TRAINABLE")
        else:
            print("⚠ Warning: embed_tokens weight not found")
    else:
        print("⚠ Warning: embed_tokens not found")


def setup_model_for_distillation(
    model_path: str = MODEL_PATH,
    tokenizer_path: str = TOKENIZER_PATH,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor]:
    """
    完整的模型准备流程
    
    1. 加载模型
    2. 冻结 ViT
    3. 冻结基础词表
    4. 验证冻结状态
    
    Args:
        model_path: 模型权重路径
        tokenizer_path: tokenizer 路径
        device: 设备
        dtype: 数据类型
        
    Returns:
        model: 准备好的模型
        tokenizer: tokenizer
        processor: processor
    """
    print("=" * 70)
    print("Setting up model for distillation")
    print("=" * 70)
    
    # 1. 加载模型
    model, tokenizer, processor = load_model_and_tokenizer(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device=device,
        dtype=dtype,
    )
    
    # 2. 冻结 ViT
    print("\n" + "-" * 70)
    print("Freezing ViT...")
    freeze_vit(model)
    
    # 3. 冻结基础词表
    print("\n" + "-" * 70)
    print("Freezing base vocabulary...")
    freeze_base_vocab(model)
    
    # 4. 验证冻结状态
    print("\n" + "-" * 70)
    print("Verifying freeze status...")
    verify_freeze_status(model)
    
    print("\n" + "=" * 70)
    print("Model setup complete!")
    print("=" * 70)
    
    return model, tokenizer, processor


def verify_freeze_status(model: Qwen3VLForConditionalGeneration) -> Dict[str, Any]:
    """
    验证参数冻结状态
    
    Args:
        model: 模型
        
    Returns:
        统计信息字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nParameter Statistics:")
    print(f"  Total parameters: {total_params / 1e6:.0f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.0f}M")
    print(f"  Frozen parameters: {frozen_params / 1e6:.0f}M")
    print(f"  Trainable ratio: {trainable_params / total_params * 100:.1f}%")
    
    # 检查各模块状态
    print(f"\nModule Status:")
    
    # ViT
    if hasattr(model, 'visual'):
        vit_trainable = sum(p.numel() for p in model.visual.parameters() if p.requires_grad)
        vit_total = sum(p.numel() for p in model.visual.parameters())
        print(f"  Visual (ViT): {vit_trainable / 1e6:.0f}M / {vit_total / 1e6:.0f}M trainable")
    
    # LLM
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        llm_trainable = sum(p.numel() for p in model.model.layers.parameters() if p.requires_grad)
        llm_total = sum(p.numel() for p in model.model.layers.parameters())
        print(f"  LLM Layers: {llm_trainable / 1e6:.0f}M / {llm_total / 1e6:.0f}M trainable")
    
    # Embeddings
    embed = None
    if hasattr(model, 'model'):
        if hasattr(model.model, 'embed_tokens'):
            embed = model.model.embed_tokens
        elif hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'embed_tokens'):
            embed = model.model.language_model.embed_tokens
    
    if embed is not None:
        embed_total = embed.weight.numel()
        print(f"  Embeddings: {embed_total / 1e6:.2f}M parameters")
        print(f"    Base vocab (0-{BASE_VOCAB_END}): FROZEN via hook")
        print(f"    Extended vocab ({EXTENDED_VOCAB_START}-{EXTENDED_VOCAB_END}): TRAINABLE")
    
    # LM Head
    if hasattr(model, 'lm_head'):
        lm_head_trainable = sum(p.numel() for p in model.lm_head.parameters() if p.requires_grad)
        lm_head_total = sum(p.numel() for p in model.lm_head.parameters())
        print(f"  LM Head: {lm_head_trainable / 1e6:.0f}M / {lm_head_total / 1e6:.0f}M trainable")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'trainable_ratio': trainable_params / total_params,
    }


def print_model_structure(model: Qwen3VLForConditionalGeneration) -> None:
    """
    打印模型结构概览
    
    Args:
        model: 模型
    """
    print("\nModel Structure:")
    print("-" * 70)
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {module.__class__.__name__} ({params / 1e6:.0f}M params)")
        
        # 打印子模块
        if name in ['visual', 'model']:
            for sub_name, sub_module in module.named_children():
                sub_params = sum(p.numel() for p in sub_module.parameters())
                print(f"  {sub_name}: {sub_module.__class__.__name__} ({sub_params / 1e6:.0f}M params)")


# ==================== 测试 ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Alpamayo2B model for distillation")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Model path")
    parser.add_argument("--tokenizer-path", default=TOKENIZER_PATH, help="Tokenizer path")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Data type")
    args = parser.parse_args()
    
    # 转换 dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    # 设置模型
    model, tokenizer, processor = setup_model_for_distillation(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        dtype=dtype,
    )
    
    # 打印模型结构
    print_model_structure(model)
    
    print("\n✓ Model is ready for distillation!")
