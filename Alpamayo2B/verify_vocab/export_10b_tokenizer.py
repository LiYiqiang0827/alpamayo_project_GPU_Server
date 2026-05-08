#!/usr/bin/env python3
"""
导出Alpamayo1.5-10B的tokenizer并对比
"""

import sys
sys.path.insert(0, "/home/user/mikelee/alpamayo_project/alpamayo_1_5/alpamayo1.5-main/src")

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from transformers import AutoTokenizer
import json

# 加载10B模型（只加载tokenizer）
print("加载Alpamayo1.5-10B tokenizer...")
tokenizer_10b = AutoTokenizer.from_pretrained(
    "/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B",
    trust_remote_code=True
)

# 保存tokenizer
output_path = "/tmp/alpamayo1_5_10b_tokenizer.json"
with open(output_path, "w") as f:
    json.dump(tokenizer_10b.get_vocab(), f, indent=2)

print(f"10B tokenizer已保存到: {output_path}")
print(f"Vocab size: {len(tokenizer_10b)}")

# 检查特殊token
tokens_to_check = [
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|traj_history_start|>",
    "<|traj_history|>",
    "<|traj_history_end|>",
    "<|traj_future_start|>",
    "<|traj_future|>",
    "<|traj_future_end|>",
    "<|route_start|>",
    "<|route_end|>",
    "<|cot_start|>",
    "<|cot_end|>",
    "<|traj_bin_0|>",
    "<|traj_bin_3999|>",
]

print("\n=== 10B模型特殊Token ID ===")
for token in tokens_to_check:
    token_id = tokenizer_10b.convert_tokens_to_ids(token)
    print(f"{token:30s} ID={token_id:6d}")
