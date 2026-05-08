#!/usr/bin/env python3
"""
Alpamayo1.5-2B 模型构建脚本
基于Alpamayo1.5-10B源码，使用CosmosReason2-2B作为VLM backbone
"""

import os
import sys
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Qwen3VLConfig,
)

# 路径配置
COSMOS_2B_PATH = "/data01/mikelee/weight/alpamayo2B"
ALPAMAYO_10B_PATH = "/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B"
OUTPUT_PATH = "/gpfs-data/mikelee/alpamayo1_5_2b_init"


def build_alpamayo1_5_2b():
    """构建Alpamayo1.5-2B模型"""
    print("=" * 70)
    print("Building Alpamayo1.5-2B Model")
    print("=" * 70)
    
    # 1. 加载CosmosReason2-2B配置
    print("\n1. Loading CosmosReason2-2B config...")
    config = Qwen3VLConfig.from_pretrained(COSMOS_2B_PATH)
    print(f"   Original vocab_size: {config.text_config.vocab_size}")
    
    # 2. 修改配置为Alpamayo1.5格式
    print("\n2. Modifying config for Alpamayo1.5...")
    config.text_config.vocab_size = 155697
    config.vocab_size = 155697
    
    # 添加Alpamayo1.5的特殊配置
    config.traj_vocab_size = 4000
    config.traj_token_start_idx = 151669
    config.tokens_per_history_traj = 48
    config.tokens_per_future_traj = 128
    
    print(f"   New vocab_size: {config.text_config.vocab_size}")
    print(f"   Trajectory tokens: {config.traj_vocab_size}")
    
    # 3. 创建输出目录
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # 4. 保存配置
    config.save_pretrained(OUTPUT_PATH)
    print(f"\n3. Config saved to: {OUTPUT_PATH}")
    
    # 5. 复制tokenizer
    print("\n4. Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(COSMOS_2B_PATH, trust_remote_code=True)
    tokenizer.save_pretrained(OUTPUT_PATH)
    processor = AutoProcessor.from_pretrained(COSMOS_2B_PATH, trust_remote_code=True)
    processor.save_pretrained(OUTPUT_PATH)
    
    # 6. 处理权重
    print("\n5. Processing weights...")
    process_weights()
    
    print("\n" + "=" * 70)
    print("Alpamayo1.5-2B initialization complete!")
    print("=" * 70)


def process_weights():
    """处理权重：从CosmosReason2-2B加载，扩展embedding"""
    
    # 加载CosmosReason2-2B权重
    print("   Loading CosmosReason2-2B weights...")
    cosmos_state = load_file(f"{COSMOS_2B_PATH}/model-expanded.safetensors")
    
    # 加载Alpamayo1.5-10B的embedding（用于复制新增token）
    print("   Loading Alpamayo1.5-10B embedding...")
    alpamayo_embed_path = f"{ALPAMAYO_10B_PATH}/model-00001-of-00005.safetensors"
    if os.path.exists(alpamayo_embed_path):
        alpamayo_state = load_file(alpamayo_embed_path)
        alpamayo_embed = alpamayo_state.get("model.embed_tokens.weight", None)
    else:
        print(f"   Warning: Alpamayo embedding not found at {alpamayo_embed_path}")
        alpamayo_embed = None
    
    # 扩展embedding
    if alpamayo_embed is not None:
        print("   Expanding embedding with Alpamayo trajectory tokens...")
        
        # Cosmos原始embedding [151936, 2048]
        cosmos_embed = cosmos_state["model.embed_tokens.weight"]
        
        # 从Alpamayo复制新增token的embedding并投影降维
        # Alpamayo: [155697, 4096] -> 取前2048维
        new_tokens = alpamayo_embed[151669:155697, :2048].clone()
        
        # 拼接：保留Cosmos原始部分 + 新增Alpamayo部分
        expanded_embed = torch.cat([
            cosmos_embed[:151669],  # 保留原始vocab
            new_tokens  # 新增轨迹token和特殊token
        ], dim=0)
        
        # 更新权重字典
        cosmos_state["model.embed_tokens.weight"] = expanded_embed
        
        # 如果lm_head独立，也需要更新（但通常共享）
        if "lm_head.weight" in cosmos_state:
            cosmos_state["lm_head.weight"] = expanded_embed
        
        print(f"   Expanded embedding: {expanded_embed.shape}")
    
    # 保存初始化权重
    output_weights_path = f"{OUTPUT_PATH}/model.safetensors"
    save_file(cosmos_state, output_weights_path)
    print(f"   Weights saved to: {output_weights_path}")
    
    # 创建权重索引文件
    index = {
        "metadata": {
            "total_size": sum(v.numel() * v.element_size() for v in cosmos_state.values())
        },
        "weight_map": {k: "model.safetensors" for k in cosmos_state.keys()}
    }
    
    with open(f"{OUTPUT_PATH}/model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    build_alpamayo1_5_2b()
