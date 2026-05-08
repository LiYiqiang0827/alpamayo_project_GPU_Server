#!/usr/bin/env python3
"""
详细分析2B和10B的token差异
"""

import json

# 加载2B的完整vocab
with open("/gpfs-data/mikelee/alpamayo1_5_2b_init/tokenizer.json", "r") as f:
    tokenizer_2b = json.load(f)

vocab_2b = tokenizer_2b["model"]["vocab"]

with open("/gpfs-data/mikelee/alpamayo1_5_2b_init/tokenizer_config.json", "r") as f:
    config_2b = json.load(f)

added_tokens_2b = config_2b.get("added_tokens_decoder", {})

# 合并
full_vocab_2b = {}
for token, token_id in vocab_2b.items():
    full_vocab_2b[token] = token_id
for token_id_str, token_info in added_tokens_2b.items():
    token = token_info["content"]
    token_id = int(token_id_str)
    full_vocab_2b[token] = token_id

# 加载10B的added_tokens（从config推断）
with open("/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B/config.json", "r") as f:
    config_10b = json.load(f)

# 10B的traj_token_ids
traj_token_ids_10b = config_10b.get("traj_token_ids", {})

print("=== 10B的特殊Token ID（从config推断）===")
for key, value in traj_token_ids_10b.items():
    print(f"  {key}: {value}")

# 对比2B和10B的特殊token
print("\n=== 对比2B和10B的特殊token ===")

# 2B的特殊token（在原始vocab之外的）
special_tokens_2b = {}
for token, token_id in full_vocab_2b.items():
    if token_id >= 151643:  # 原始Qwen3 vocab结束位置
        special_tokens_2b[token] = token_id

print(f"\n2B特殊token数量: {len(special_tokens_2b)}")
print(f"2B特殊token范围: {min(special_tokens_2b.values())} - {max(special_tokens_2b.values())}")

# 按类别分组
print("\n=== 2B特殊token分类 ===")

categories = {
    "基础特殊token": [],
    "轨迹token": [],
    "Alpamayo特殊token": [],
    "其他": []
}

for token, token_id in sorted(special_tokens_2b.items(), key=lambda x: x[1]):
    if "traj_bin_" in token:
        categories["轨迹token"].append((token, token_id))
    elif token in ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", 
                     "<|image_pad|>", "<|video_pad|>", "<|vision_pad|>"]:
        categories["基础特殊token"].append((token, token_id))
    elif "traj_" in token or "route_" in token or "cot_" in token or "question" in token or "answer" in token:
        categories["Alpamayo特殊token"].append((token, token_id))
    else:
        categories["其他"].append((token, token_id))

for category, tokens in categories.items():
    print(f"\n{category} ({len(tokens)}个):")
    for token, token_id in tokens[:10]:
        print(f"  {token:40s} ID={token_id}")
    if len(tokens) > 10:
        print(f"  ... 还有 {len(tokens) - 10} 个")

# 检查哪些token可能是2B独有的
print("\n=== 可能是2B独有的token ===")
# 这些是在原始Qwen3/Cosmos基础上新增的
alpamayo_specific = [
    "<|cot_start|>", "<|cot_end|>",
    "<|traj_history_start|>", "<|traj_history|>", "<|traj_history_end|>",
    "<|traj_future_start|>", "<|traj_future|>", "<|traj_future_end|>",
    "<|route_start|>", "<|route_pad|>", "<|route_end|>",
    "<|question_start|>", "<|question_end|>",
    "<|answer_start|>", "<|answer_end|>",
    "<|prompt_start|>", "<|prompt_end|>",
]

print(f"\nAlpamayo特定token ({len(alpamayo_specific)}个):")
for token in alpamayo_specific:
    if token in full_vocab_2b:
        print(f"  ✅ {token}: ID={full_vocab_2b[token]}")
    else:
        print(f"  ❌ {token}: 未找到")

# 计算差异
print("\n=== 差异分析 ===")
print(f"2B总token数: {len(full_vocab_2b)}")
print(f"10B config vocab_size: {config_10b.get('vocab_size')}")
print(f"差异: {len(full_vocab_2b) - config_10b.get('vocab_size')} 个token")

# 10B的traj_token_start_idx
traj_start_10b = config_10b.get('traj_token_start_idx')
print(f"\n10B traj_token_start_idx: {traj_start_10b}")
print(f"2B traj_bin_0 ID: {full_vocab_2b.get('<|traj_bin_0|>')}")

if full_vocab_2b.get('<|traj_bin_0|>') == traj_start_10b:
    print("✅ 轨迹token起始位置一致!")
else:
    print("❌ 轨迹token起始位置不一致!")
