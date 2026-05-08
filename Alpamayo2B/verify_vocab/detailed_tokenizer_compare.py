#!/usr/bin/env python3
"""
详细对比Alpamayo1.5-10B和Alpamayo2B的tokenizer差异
"""

import json

# 读取两个tokenizer的词汇表
with open("/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B/tokenizer.json", "r") as f:
    tokenizer_10b = json.load(f)

with open("/gpfs-data/mikelee/alpamayo1_5_2b_init/tokenizer.json", "r") as f:
    tokenizer_2b = json.load(f)

# 获取词汇表
vocab_10b = tokenizer_10b["model"]["vocab"]
vocab_2b = tokenizer_2b["model"]["vocab"]

print(f"10B vocab size: {len(vocab_10b)}")
print(f"2B vocab size: {len(vocab_2b)}")
print(f"差异: {len(vocab_2b) - len(vocab_10b)} 个token")

# 找出差异
only_in_10b = set(vocab_10b.keys()) - set(vocab_2b.keys())
only_in_2b = set(vocab_2b.keys()) - set(vocab_10b.keys())

print(f"\n只在10B中存在的token ({len(only_in_10b)}个):")
for token in sorted(only_in_10b):
    print(f"  {token:40s} ID={vocab_10b[token]:6d}")

print(f"\n只在2B中存在的token ({len(only_in_2b)}个):")
for token in sorted(only_in_2b):
    print(f"  {token:40s} ID={vocab_2b[token]:6d}")

# 检查相同token的ID是否一致
common_tokens = set(vocab_10b.keys()) & set(vocab_2b.keys())
id_mismatch = []
for token in common_tokens:
    if vocab_10b[token] != vocab_2b[token]:
        id_mismatch.append((token, vocab_10b[token], vocab_2b[token]))

if id_mismatch:
    print(f"\n❌ ID不一致的token ({len(id_mismatch)}个):")
    for token, id_10b, id_2b in id_mismatch:
        print(f"  {token}: 10B={id_10b}, 2B={id_2b}")
else:
    print(f"\n✅ 所有{len(common_tokens)}个共同token的ID一致")

# 检查关键token
print("\n=== 关键Token对比 ===")
key_tokens = [
    "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
    "<|traj_history_start|>", "<|traj_history|>", "<|traj_history_end|>",
    "<|traj_future_start|>", "<|traj_future|>", "<|traj_future_end|>",
    "<|route_start|>", "<|route_end|>", "<|cot_start|>", "<|cot_end|>",
    "<|traj_bin_0|>", "<|traj_bin_3999|>",
]

for token in key_tokens:
    id_10b = vocab_10b.get(token, "N/A")
    id_2b = vocab_2b.get(token, "N/A")
    match = "✅" if id_10b == id_2b else "❌"
    print(f"{match} {token:30s} 10B={id_10b:6s} 2B={id_2b:6s}")
