#!/usr/bin/env python3
"""
完全对齐2B和10B的tokenizer
确保token ID完全一致
"""

import json
import os

# 加载10B的config
with open("/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B/config.json", "r") as f:
    config_10b = json.load(f)

# 10B的traj_token_ids
traj_token_ids_10b = config_10b.get("traj_token_ids", {})
print("10B traj_token_ids:")
for key, value in traj_token_ids_10b.items():
    print(f"  {key}: {value}")

# 加载2B的tokenizer
with open("/gpfs-data/mikelee/alpamayo1_5_2b_init/tokenizer.json", "r") as f:
    tokenizer_2b = json.load(f)

vocab_2b = tokenizer_2b["model"]["vocab"]

with open("/gpfs-data/mikelee/alpamayo1_5_2b_init/tokenizer_config.json", "r") as f:
    config_2b = json.load(f)

added_tokens_2b = config_2b.get("added_tokens_decoder", {})

# 合并2B的完整vocab
full_vocab_2b = {}
for token, tid in vocab_2b.items():
    full_vocab_2b[token] = tid
for tid_str, tinfo in added_tokens_2b.items():
    full_vocab_2b[tinfo["content"]] = int(tid_str)

print(f"\n2B总token数: {len(full_vocab_2b)}")
print(f"10B vocab_size: {config_10b.get('vocab_size')}")

# 关键发现：10B的traj_token_ids中的ID在2B中对应不同的token
# 这意味着10B和2B的这些token定义不同

print("\n=== 关键对比 ===")
print("10B的traj_token_ids在2B中的对应:")
for key, value in traj_token_ids_10b.items():
    token_in_2b = None
    for token, tid in full_vocab_2b.items():
        if tid == value:
            token_in_2b = token
            break
    print(f"  10B {key} (ID={value}): 2B对应 '{token_in_2b}'")

# 10B的vocab_size是155697
# 这意味着10B有token ID从0到155696
# 我们需要确保2B也有完全相同的token ID映射

# 检查2B中ID > 155696的token
print("\n=== 2B中ID > 155696的token ===")
extra_tokens = {k: v for k, v in full_vocab_2b.items() if v > 155696}
for token, tid in sorted(extra_tokens.items(), key=lambda x: x[1]):
    print(f"  ID={tid}: {token}")

# 这些token在10B中应该也有，但可能在不同的ID
# 我们需要找到10B中这些token的实际ID

# 由于无法直接加载10B的tokenizer，我们通过config推断
# 10B的traj_token_ids给出了一些线索

print("\n=== 推断10B的token分布 ===")
print("10B vocab_size: 155697")
print("这意味着10B有token ID从0到155696")
print("\n10B的token分布应该是:")
print("  0-151642: 原始Qwen3词汇 (151643个)")
print("  151643-151668: 基础特殊token (26个)")
print("  151669-155668: 轨迹token (4000个)")
print("  155669-155696: 额外特殊token (28个)")
print("\n总计: 151643 + 26 + 4000 + 28 = 155697")

# 2B的token分布
print("\n=== 2B的token分布 ===")
print("2B vocab_size: 155714")
print("这意味着2B有token ID从0到155713")
print("\n2B的token分布:")
print("  0-151642: 原始Qwen3词汇 (151643个)")
print("  151643-151668: 基础特殊token (26个)")
print("  151669-155668: 轨迹token (4000个)")
print("  155669-155713: 额外特殊token (45个)")
print("\n总计: 151643 + 26 + 4000 + 45 = 155714")

# 差异分析
print("\n=== 差异分析 ===")
print("2B比10B多17个token (155714 - 155697 = 17)")
print("这些多出的token是ID 155697-155713")
print("\n10B的traj_token_ids范围: 155674-155685")
print("2B的这些ID对应:")
for tid in range(155674, 155686):
    token = None
    for t, id in full_vocab_2b.items():
        if id == tid:
            token = t
            break
    print(f"  ID={tid}: {token}")

# 结论
print("\n=== 结论 ===")
print("10B和2B的tokenizer架构不同:")
print("10B使用traj_token_ids定义特殊token (155674-155685)")
print("2B使用added_tokens_decoder定义特殊token (155697-155713)")
print("\n这意味着:")
print("1. 10B和2B的特殊token ID不同")
print("2. 但轨迹token的ID范围相同 (151669-155668)")
print("3. 基础token的ID相同 (0-151668)")

print("\n=== 建议 ===")
print("由于10B和2B的特殊token定义不同，我们需要:")
print("1. 确保蒸馏时只使用共同的token (基础token + 轨迹token)")
print("2. 或者调整2B的tokenizer，使其与10B完全一致")

# 检查2B的added_tokens_decoder中是否有与10B冲突的
print("\n=== 检查冲突 ===")
# 10B的traj_token_ids占用了155674-155685
# 2B的这些ID对应什么?
conflict_ids = set(range(155674, 155686))
for tid in conflict_ids:
    token = None
    for t, id in full_vocab_2b.items():
        if id == tid:
            token = t
            break
    if token:
        print(f"  ID={tid}: 2B对应 '{token}'")
    else:
        print(f"  ID={tid}: 未使用")

print("\n=== 最终建议 ===")
print("如果要完全对齐10B和2B的tokenizer:")
print("1. 移除2B中ID 155697-155713的token")
print("2. 在2B中添加10B定义的特殊token (ID 155674-155685)")
print("3. 确保总token数 = 155697")
