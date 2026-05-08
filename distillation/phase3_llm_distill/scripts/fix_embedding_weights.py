#!/usr/bin/env python3
"""
修复Alpamayo1.5-2B的embedding权重，匹配新的vocab size
"""

import torch
from safetensors.torch import load_file, save_file

MODEL_PATH = "/gpfs-data/mikelee/alpamayo1_5_2b_init"
OLD_VOCAB = 155697
NEW_VOCAB = 155714  # 添加了17个特殊token
HIDDEN_SIZE = 2048

def fix_embedding_weights():
    print("=" * 70)
    print("Fixing Embedding Weights")
    print("=" * 70)
    
    # 加载现有权重
    print("\n1. Loading existing weights...")
    state_dict = load_file(f"{MODEL_PATH}/model.safetensors")
    
    # 获取当前embedding
    embed_key = "model.language_model.embed_tokens.weight"
    embed = state_dict[embed_key]
    print(f"   Current embedding shape: {embed.shape}")
    print(f"   Expected: [{NEW_VOCAB}, {HIDDEN_SIZE}]")
    
    # 扩展embedding
    if embed.shape[0] < NEW_VOCAB:
        print(f"\n2. Expanding embedding from {embed.shape[0]} to {NEW_VOCAB}...")
        
        # 创建新的embedding矩阵
        new_embed = torch.zeros(NEW_VOCAB, HIDDEN_SIZE, dtype=embed.dtype, device=embed.device)
        
        # 复制原有权重
        new_embed[:OLD_VOCAB] = embed[:OLD_VOCAB]
        
        # 新增token使用随机初始化（或从Alpamayo复制）
        # 这里使用Xavier初始化
        torch.nn.init.xavier_uniform_(new_embed[OLD_VOCAB:])
        
        # 更新权重字典
        state_dict[embed_key] = new_embed
        
        # 如果lm_head独立，也需要更新
        if "lm_head.weight" in state_dict:
            state_dict["lm_head.weight"] = new_embed
        
        print(f"   New embedding shape: {new_embed.shape}")
    
    # 保存修复后的权重
    print("\n3. Saving fixed weights...")
    
    # 处理共享权重：safetensors不允许共享内存的tensor
    # 需要复制共享的tensor
    state_dict_to_save = {}
    seen_data_ptr = {}
    
    for key, tensor in state_dict.items():
        data_ptr = tensor.data_ptr()
        if data_ptr in seen_data_ptr:
            # 共享内存的tensor，需要复制
            state_dict_to_save[key] = tensor.clone()
        else:
            seen_data_ptr[data_ptr] = key
            state_dict_to_save[key] = tensor
    
    save_file(state_dict_to_save, f"{MODEL_PATH}/model.safetensors")
    
    # 更新索引文件
    import json
    index = {
        "metadata": {
            "total_size": sum(v.numel() * v.element_size() for v in state_dict.values())
        },
        "weight_map": {k: "model.safetensors" for k in state_dict.keys()}
    }
    
    with open(f"{MODEL_PATH}/model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Embedding weights fixed!")
    print("=" * 70)

if __name__ == "__main__":
    fix_embedding_weights()
