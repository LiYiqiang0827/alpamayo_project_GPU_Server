#!/usr/bin/env python3
"""
对比Alpamayo1.5-10B和Alpamayo2B的tokenizer
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

print("\n=== 特殊Token ID对比 ===")
all_match = True
for token in tokens_to_check:
    id_10b = vocab_10b.get(token, -1)
    id_2b = vocab_2b.get(token, -1)
    match = "✅" if id_10b == id_2b else "❌"
    if id_10b != id_2b:
        all_match = False
    print(f"{match} {token:30s} 10B={id_10b:6d} 2B={id_2b:6d}")

# 检查trajectory bin token范围
print("\n=== 轨迹Bin Token范围 ===")
traj_0_10b = vocab_10b.get("<|traj_bin_0|>", -1)
traj_0_2b = vocab_2b.get("<|traj_bin_0|>", -1)
traj_3999_10b = vocab_10b.get("<|traj_bin_3999|>", -1)
traj_3999_2b = vocab_2b.get("<|traj_bin_3999|>", -1)

print(f"traj_bin_0: 10B={traj_0_10b}, 2B={traj_0_2b}")
print(f"traj_bin_3999: 10B={traj_3999_10b}, 2B={traj_3999_2b}")

# 检查vocab是否完全一致
print("\n=== 词汇表一致性检查 ===")
if vocab_10b == vocab_2b:
    print("✅ 两个tokenizer的词汇表完全一致！")
else:
    print("❌ 词汇表存在差异")
    
    # 找出差异
    only_in_10b = set(vocab_10b.keys()) - set(vocab_2b.keys())
    only_in_2b = set(vocab_2b.keys()) - set(vocab_10b.keys())
    
    if only_in_10b:
        print(f"\n只在10B中存在的token ({len(only_in_10b)}个):")
        for token in list(only_in_10b)[:10]:
            print(f"  {token}")
    
    if only_in_2b:
        print(f"\n只在2B中存在的token ({len(only_in_2b)}个):")
        for token in list(only_in_2b)[:10]:
            print(f"  {token}")
    
    # 检查相同token的ID是否一致
    common_tokens = set(vocab_10b.keys()) & set(vocab_2b.keys())
    id_mismatch = []
    for token in common_tokens:
        if vocab_10b[token] != vocab_2b[token]:
            id_mismatch.append((token, vocab_10b[token], vocab_2b[token]))
    
    if id_mismatch:
        print(f"\n❌ ID不一致的token ({len(id_mismatch)}个):")
        for token, id_10b, id_2b in id_mismatch[:10]:
            print(f"  {token}: 10B={id_10b}, 2B={id_2b}")
    else:
        print("\n✅ 所有共同token的ID一致")

result = "全部一致✅" if all_match else "存在差异❌"
print(f"\n=== 结果: {result} ===")
