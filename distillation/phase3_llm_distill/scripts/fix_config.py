#!/usr/bin/env python3
"""
修复Alpamayo1.5-2B的config，匹配新的vocab size
"""

import json
from transformers import Qwen3VLConfig

MODEL_PATH = "/gpfs-data/mikelee/alpamayo1_5_2b_init"
NEW_VOCAB = 155714

def fix_config():
    print("=" * 70)
    print("Fixing Config")
    print("=" * 70)
    
    # 加载现有config
    print("\n1. Loading existing config...")
    config = Qwen3VLConfig.from_pretrained(MODEL_PATH)
    print(f"   Current vocab_size: {config.text_config.vocab_size}")
    
    # 修改vocab_size
    print(f"\n2. Updating vocab_size to {NEW_VOCAB}...")
    config.text_config.vocab_size = NEW_VOCAB
    config.vocab_size = NEW_VOCAB
    
    # 保存config
    config.save_pretrained(MODEL_PATH)
    print(f"   Config saved!")
    
    # 验证
    config_new = Qwen3VLConfig.from_pretrained(MODEL_PATH)
    print(f"\n3. Verification:")
    print(f"   New vocab_size: {config_new.text_config.vocab_size}")
    
    print("\n" + "=" * 70)
    print("Config fixed!")
    print("=" * 70)

if __name__ == "__main__":
    fix_config()
