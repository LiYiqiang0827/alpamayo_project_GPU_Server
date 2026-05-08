#!/usr/bin/env python3
"""
详细分析2B和10B的vocab差异
找出2B多出的43个token是什么
"""

import json

# 加载2B的完整vocab
with open("/gpfs-data/mikelee/alpamayo1_5_2b_init/tokenizer.json", "r") as f:
    tokenizer_2b = json.load(f)

vocab_2b = tokenizer_2b["model"]["vocab"]

with open("/gpfs-data/mikelee/alpamayo1_5_2b_init/tokenizer_config.json", "r") as f:
    config_2b = json.load(f)

added_tokens_2b = config_2b.get("added_tokens_decoder", {})

# 合并2B的完整vocab
full_vocab_2b = {}
for token, token_id in vocab_2b.items():
    full_vocab_2b[token] = token_id
for token_id_str, token_info in added_tokens_2b.items():
    token = token_info["content"]
    token_id = int(token_id_str)
    full_vocab_2b[token] = token_id

print("=" * 70)
print("2B Token分析")
print("=" * 70)
print(f"2B总token数: {len(full_vocab_2b)}")

# 按ID范围分类
print("\n=== 按ID范围分类 ===")

# 1. 原始Qwen3词汇 (0-151642)
qwen3_tokens = {k: v for k, v in full_vocab_2b.items() if v < 151643}
print(f"1. 原始Qwen3词汇: {len(qwen3_tokens)} 个 (ID: 0-151642)")

# 2. 基础特殊token (151643-151668)
base_special = {k: v for k, v in full_vocab_2b.items() if 151643 <= v <= 151668}
print(f"2. 基础特殊token: {len(base_special)} 个 (ID: 151643-151668)")
for token, tid in sorted(base_special.items(), key=lambda x: x[1]):
    print(f"   {token}: {tid}")

# 3. 额外特殊token (151669-151740) - 这部分可能是2B特有的
extra_special = {k: v for k, v in full_vocab_2b.items() if 151669 <= v < 151669 + 4000 and "traj_bin_" not in k}
print(f"\n3. 额外特殊token (在轨迹token范围内): {len(extra_special)} 个")
for token, tid in sorted(extra_special.items(), key=lambda x: x[1]):
    print(f"   {token}: {tid}")

# 4. 轨迹token (151669-155668)
traj_tokens = {k: v for k, v in full_vocab_2b.items() if "traj_bin_" in k}
print(f"\n4. 轨迹token: {len(traj_tokens)} 个 (ID: 151669-155668)")

# 5. 10B定义的traj_token_ids (155669-155696)
# 从10B config读取
with open("/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B/config.json", "r") as f:
    config_10b = json.load(f)

traj_token_ids_10b = config_10b.get("traj_token_ids", {})
print(f"\n5. 10B定义的traj_token_ids:")
for key, value in traj_token_ids_10b.items():
    print(f"   {key}: {value}")

# 6. Alpamayo特定token (155697-155713)
alpamayo_tokens = {k: v for k, v in full_vocab_2b.items() if v >= 155697}
print(f"\n6. Alpamayo特定token: {len(alpamayo_tokens)} 个 (ID: 155697+)")
for token, tid in sorted(alpamayo_tokens.items(), key=lambda x: x[1]):
    print(f"   {token}: {tid}")

# 分析差异
print("\n" + "=" * 70)
print("差异分析")
print("=" * 70)

print(f"\n10B config中的vocab_size: {config_10b.get('vocab_size')}")
print(f"10B config中的traj_token_start_idx: {config_10b.get('traj_token_start_idx')}")
print(f"10B config中的traj_vocab_size: {config_10b.get('traj_vocab_size')}")

# 计算10B的实际token分布
print(f"\n=== 推断10B的token分布 ===")
print(f"1. 原始Qwen3词汇: ~151643 个")
print(f"2. 基础特殊token: ~25 个 (151643-151668)")
print(f"3. 轨迹token: 4000 个 (151669-155668)")
print(f"4. 10B定义的traj_token_ids: {len(traj_token_ids_10b)} 个")
print(f"   - future: 155685")
print(f"   - future_end: 155683")
print(f"   - future_start: 155681")
print(f"   - history: 155684")
print(f"   - history_end: 155676")
print(f"   - history_start: 155674")

# 这些ID在2B中是什么
traj_id_range = set(range(155674, 155686))
print(f"\n=== 检查155674-155685范围在2B中是什么 ===")
for token, tid in full_vocab_2b.items():
    if tid in traj_id_range:
        print(f"   ID {tid}: {token}")

# 找出2B多出的token
print("\n" + "=" * 70)
print("找出2B多出的43个token")
print("=" * 70)

# 10B的vocab_size是155697
# 这意味着10B有token ID从0到155696
# 2B有token ID从0到155739

# 多出的token应该是ID > 155696的
extra_in_2b = {k: v for k, v in full_vocab_2b.items() if v > 155696}
print(f"\n2B中ID > 155696的token ({len(extra_in_2b)}个):")
for token, tid in sorted(extra_in_2b.items(), key=lambda x: x[1]):
    print(f"   ID={tid}: {token}")

# 检查是否有ID重复
print("\n=== 检查ID重复 ===")
from collections import Counter
id_counts = Counter(full_vocab_2b.values())
duplicates = {k: v for k, v in id_counts.items() if v > 1}
if duplicates:
    print(f"❌ 发现 {len(duplicates)} 个重复ID!")
    for tid, count in list(duplicates.items())[:10]:
        tokens = [k for k, v in full_vocab_2b.items() if v == tid]
        print(f"   ID {tid}: {tokens} (出现{count}次)")
else:
    print("✅ 没有重复ID")

# 最终总结
print("\n" + "=" * 70)
print("总结")
print("=" * 70)
print(f"2B总token数: {len(full_vocab_2b)}")
print(f"10B vocab_size: {config_10b.get('vocab_size')}")
print(f"差异: {len(full_vocab_2b) - config_10b.get('vocab_size')} 个token")
print(f"\n多出的token主要是:")
print(f"1. Alpamayo特定token (cot_start, traj_history等): 17个")
print(f"2. 其他额外token: {len(extra_in_2b) - 17}个")
