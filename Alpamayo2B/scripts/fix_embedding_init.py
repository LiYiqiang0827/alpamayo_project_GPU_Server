#!/usr/bin/env python3
"""
修复Alpamayo1.5-2B的embedding初始化
确保新token的统计特性与旧token一致
"""

import torch
from safetensors.torch import load_file, save_file

MODEL_PATH = "/gpfs-data/mikelee/alpamayo1_5_2b_init"
OLD_VOCAB = 151669  # Cosmos原始vocab
NEW_VOCAB = 155714  # 扩展后总vocab
HIDDEN_SIZE = 2048

def fix_embedding_initialization():
    print("=" * 70)
    print("Fixing Embedding Initialization")
    print("=" * 70)
    
    # 加载现有权重
    print("\n1. Loading existing weights...")
    state_dict = load_file(f"{MODEL_PATH}/model.safetensors")
    
    embed_key = "model.language_model.embed_tokens.weight"
    embed = state_dict[embed_key]
    
    print(f"   Embedding shape: {embed.shape}")
    
    # 分析旧token的统计特性
    old_embed = embed[:OLD_VOCAB]
    old_mean = old_embed.mean()
    old_std = old_embed.std()
    
    print(f"\n2. Analyzing old tokens (0-{OLD_VOCAB-1}):")
    print(f"   Mean: {old_mean:.4f}")
    print(f"   Std:  {old_std:.4f}")
    print(f"   Min:  {old_embed.min():.4f}")
    print(f"   Max:  {old_embed.max():.4f}")
    
    # 分析当前新token的统计特性
    new_embed_current = embed[OLD_VOCAB:]
    print(f"\n3. Current new tokens ({OLD_VOCAB}-{NEW_VOCAB-1}):")
    print(f"   Mean: {new_embed_current.mean():.4f}")
    print(f"   Std:  {new_embed_current.std():.4f}")
    
    # 重新初始化新token
    print(f"\n4. Re-initializing new tokens with matching statistics...")
    
    # 方法：使用与旧token相同的正态分布
    torch.manual_seed(42)  # 可复现
    new_embed_fixed = torch.normal(
        mean=old_mean,
        std=old_std,
        size=(NEW_VOCAB - OLD_VOCAB, HIDDEN_SIZE),
        dtype=embed.dtype,
        device=embed.device,
    )
    
    # 替换新token
    embed_fixed = embed.clone()
    embed_fixed[OLD_VOCAB:] = new_embed_fixed
    
    # 验证
    print(f"\n5. Verification:")
    print(f"   Old tokens std: {embed_fixed[:OLD_VOCAB].std():.4f}")
    print(f"   New tokens std: {embed_fixed[OLD_VOCAB:].std():.4f}")
    print(f"   Overall std:    {embed_fixed.std():.4f}")
    
    # 更新权重字典
    state_dict[embed_key] = embed_fixed
    
    # 如果lm_head独立，也需要更新
    if "lm_head.weight" in state_dict:
        state_dict["lm_head.weight"] = embed_fixed
    
    # 保存修复后的权重
    print("\n6. Saving fixed weights...")
    
    # 处理共享权重
    state_dict_to_save = {}
    seen_data_ptr = {}
    
    for key, tensor in state_dict.items():
        data_ptr = tensor.data_ptr()
        if data_ptr in seen_data_ptr:
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
    print("Embedding initialization fixed!")
    print("=" * 70)

if __name__ == "__main__":
    fix_embedding_initialization()
