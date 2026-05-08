#!/usr/bin/env python3
"""
修复Alpamayo2B的tokenizer，确保与embedding矩阵匹配
"""

import json
import os

# 加载现有的tokenizer
with open("/gpfs-data/mikelee/alpamayo1_5_2b_init/tokenizer.json", "r") as f:
    tokenizer_data = json.load(f)

vocab = tokenizer_data["model"]["vocab"]
print(f"当前tokenizer vocab size: {len(vocab)}")

# 检查是否已经有特殊token
special_tokens = [
    "<|cot_start|>", "<|cot_end|>",
    "<|traj_history_start|>", "<|traj_history|>", "<|traj_history_end|>",
    "<|traj_future_start|>", "<|traj_future|>", "<|traj_future_end|>",
    "<|route_start|>", "<|route_pad|>", "<|route_end|>",
    "<|question_start|>", "<|question_end|>",
    "<|answer_start|>", "<|answer_end|>",
    "<|prompt_start|>", "<|prompt_end|>",
]

# 检查哪些token已经存在
existing = []
missing = []
for token in special_tokens:
    if token in vocab:
        existing.append((token, vocab[token]))
    else:
        missing.append(token)

print(f"\n已存在的特殊token ({len(existing)}个):")
for token, token_id in existing:
    print(f"  {token}: ID={token_id}")

print(f"\n缺失的特殊token ({len(missing)}个):")
for token in missing:
    print(f"  {token}")

# 检查轨迹token
traj_start = vocab.get("<|traj_bin_0|>", None)
traj_end = vocab.get("<|traj_bin_3999|>", None)

print(f"\n轨迹token:")
print(f"  traj_bin_0: {traj_start}")
print(f"  traj_bin_3999: {traj_end}")

if traj_start is None:
    print("  ❌ 轨迹token未添加！")
else:
    print(f"  ✅ 轨迹token范围: {traj_start} - {traj_end}")

# 计算预期的vocab_size
expected_vocab_size = 151643 + 4000 + len(special_tokens)
print(f"\n预期vocab_size: {expected_vocab_size}")
print(f"实际vocab_size: {len(vocab)}")
print(f"embedding矩阵大小: 155714")

# 检查差异
diff = 155714 - len(vocab)
print(f"\n差异: {diff}")
