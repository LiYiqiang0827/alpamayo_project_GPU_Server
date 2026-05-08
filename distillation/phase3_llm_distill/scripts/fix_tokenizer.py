#!/usr/bin/env python3
"""
修复Alpamayo1.5-2B的tokenizer，添加特殊token
"""

import os
from transformers import AutoTokenizer, AutoProcessor

MODEL_PATH = "/gpfs-data/mikelee/alpamayo1_5_2b_init"

def fix_tokenizer():
    print("=" * 70)
    print("Fixing Alpamayo1.5-2B Tokenizer")
    print("=" * 70)
    
    # 加载现有tokenizer
    print("\n1. Loading existing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"   Current vocab size: {len(tokenizer)}")
    
    # 添加Alpamayo1.5的特殊token
    print("\n2. Adding special tokens...")
    special_tokens = {
        "cot_start": "<|cot_start|>",
        "cot_end": "<|cot_end|>",
        "traj_history_start": "<|traj_history_start|>",
        "traj_history": "<|traj_history|>",
        "traj_history_end": "<|traj_history_end|>",
        "traj_future_start": "<|traj_future_start|>",
        "traj_future": "<|traj_future|>",
        "traj_future_end": "<|traj_future_end|>",
        "route_start": "<|route_start|>",
        "route_pad": "<|route_pad|>",
        "route_end": "<|route_end|>",
        "question_start": "<|question_start|>",
        "question_end": "<|question_end|>",
        "answer_start": "<|answer_start|>",
        "answer_end": "<|answer_end|>",
        "prompt_start": "<|prompt_start|>",
        "prompt_end": "<|prompt_end|>",
    }
    
    # 添加为special tokens
    tokens_to_add = list(special_tokens.values())
    tokenizer.add_tokens(tokens_to_add, special_tokens=True)
    
    print(f"   Added {len(tokens_to_add)} special tokens")
    print(f"   New vocab size: {len(tokenizer)}")
    
    # 验证token IDs
    print("\n3. Verifying token IDs...")
    for name, token in special_tokens.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"   {name}: {token} -> {token_id}")
    
    # 保存修复后的tokenizer
    print("\n4. Saving fixed tokenizer...")
    tokenizer.save_pretrained(MODEL_PATH)
    
    # 同时保存processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor.save_pretrained(MODEL_PATH)
    
    print("\n" + "=" * 70)
    print("Tokenizer fixed!")
    print("=" * 70)

if __name__ == "__main__":
    fix_tokenizer()
