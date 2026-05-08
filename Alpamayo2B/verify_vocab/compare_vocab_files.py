#!/usr/bin/env python3
"""
直接对比两个tokenizer的vocab文件
不加载模型，只对比token ID
"""

import json

# 加载2B的tokenizer.json
with open("/gpfs-data/mikelee/alpamayo1_5_2b_init/tokenizer.json", "r") as f:
    tokenizer_2b = json.load(f)

vocab_2b = tokenizer_2b["model"]["vocab"]
print(f"2B vocab size: {len(vocab_2b)}")

# 加载2B的added_tokens_decoder
with open("/gpfs-data/mikelee/alpamayo1_5_2b_init/tokenizer_config.json", "r") as f:
    config_2b = json.load(f)

added_tokens_2b = config_2b.get("added_tokens_decoder", {})
print(f"2B added_tokens: {len(added_tokens_2b)}")

# 合并vocab和added_tokens
full_vocab_2b = {}
for token, token_id in vocab_2b.items():
    full_vocab_2b[token] = token_id
for token_id_str, token_info in added_tokens_2b.items():
    token = token_info["content"]
    token_id = int(token_id_str)
    full_vocab_2b[token] = token_id

print(f"2B full vocab size: {len(full_vocab_2b)}")

# 检查关键token
key_tokens = [
    "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
    "<|traj_history_start|>", "<|traj_history|>", "<|traj_history_end|>",
    "<|traj_future_start|>", "<|traj_future|>", "<|traj_future_end|>",
    "<|route_start|>", "<|route_end|>", "<|cot_start|>", "<|cot_end|>",
    "<|traj_bin_0|>", "<|traj_bin_3999|>",
]

print("\n=== 2B关键Token ID ===")
for token in key_tokens:
    token_id = full_vocab_2b.get(token, "NOT FOUND")
    print(f"  {token:30s} ID={token_id}")

# 检查轨迹token范围
print("\n=== 轨迹Token范围 ===")
traj_start = full_vocab_2b.get("<|traj_bin_0|>")
traj_end = full_vocab_2b.get("<|traj_bin_3999|>")
print(f"  traj_bin_0: {traj_start}")
print(f"  traj_bin_3999: {traj_end}")
print(f"  范围: {traj_start} - {traj_end} ({traj_end - traj_start + 1} tokens)")

# 检查是否有holes（缺失的ID）
print("\n=== 检查Vocab完整性 ===")
all_ids = sorted(full_vocab_2b.values())
holes = []
for i in range(len(all_ids) - 1):
    if all_ids[i+1] - all_ids[i] > 1:
        for missing_id in range(all_ids[i] + 1, all_ids[i+1]):
            holes.append(missing_id)

if holes:
    print(f"❌ 发现 {len(holes)} 个holes:")
    print(f"  缺失的ID: {holes[:20]}...")
else:
    print("✅ 没有holes，所有ID连续")

# 检查10B的config
print("\n=== 10B Config参考 ===")
with open("/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B/config.json", "r") as f:
    config_10b = json.load(f)

print(f"  10B vocab_size: {config_10b.get('vocab_size')}")
print(f"  10B traj_token_start_idx: {config_10b.get('traj_token_start_idx')}")
print(f"  10B traj_vocab_size: {config_10b.get('traj_vocab_size')}")

# 检查10B的traj_token_ids（如果有）
traj_token_ids = config_10b.get("traj_token_ids", {})
if traj_token_ids:
    print(f"\n  10B traj_token_ids:")
    for key, value in traj_token_ids.items():
        print(f"    {key}: {value}")

# 最终对比
print("\n=== 关键对比 ===")
print(f"2B traj_bin_0 ID: {traj_start}")
print(f"10B traj_token_start_idx: {config_10b.get('traj_token_start_idx')}")

if traj_start == config_10b.get('traj_token_start_idx'):
    print("✅ 轨迹token起始位置一致!")
else:
    print("❌ 轨迹token起始位置不一致!")

print(f"\n2B vocab_size (实际): {len(full_vocab_2b)}")
print(f"10B vocab_size (config): {config_10b.get('vocab_size')}")

if len(full_vocab_2b) == config_10b.get('vocab_size'):
    print("✅ vocab_size一致!")
else:
    print(f"⚠️ vocab_size不一致 (差异: {len(full_vocab_2b) - config_10b.get('vocab_size')})")
