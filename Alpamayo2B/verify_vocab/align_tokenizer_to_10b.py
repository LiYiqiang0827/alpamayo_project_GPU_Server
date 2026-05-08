#!/usr/bin/env python3
"""
完全对齐2B的tokenizer到10B的标准
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

print(f"\n修复前:")
print(f"2B vocab size: {len(vocab_2b)}")
print(f"2B added_tokens: {len(added_tokens_2b)}")

# 策略：
# 1. 保留vocab_2b中的所有token（包括轨迹token）
# 2. 修改added_tokens_2b，使其与10B的traj_token_ids一致
# 3. 移除2B中ID > 155696的token
# 4. 添加10B定义的特殊token到正确的ID

# 步骤1: 创建新的added_tokens_decoder
new_added_tokens = {}

# 保留基础特殊token (ID 151643-151668)
for tid_str, tinfo in added_tokens_2b.items():
    tid = int(tid_str)
    if 151643 <= tid <= 151668:
        new_added_tokens[tid_str] = tinfo

# 添加10B定义的traj_token_ids
# 10B的traj_token_ids:
#   future: 155685
#   future_end: 155683
#   future_start: 155681
#   history: 155684
#   history_end: 155676
#   history_start: 155674

# 我们需要在2B中创建对应的token
# 但2B的这些ID已经被其他token占用了

# 检查2B中这些ID当前对应什么
target_ids = {
    "future": 155685,
    "future_end": 155683,
    "future_start": 155681,
    "history": 155684,
    "history_end": 155676,
    "history_start": 155674,
}

print("\n=== 检查2B中目标ID的当前占用 ===")
for key, target_id in target_ids.items():
    current_token = None
    for token, tid in vocab_2b.items():
        if tid == target_id:
            current_token = token
            break
    if not current_token:
        for tid_str, tinfo in added_tokens_2b.items():
            if int(tid_str) == target_id:
                current_token = tinfo["content"]
                break
    print(f"  ID {target_id} ({key}): 当前对应 '{current_token}'")

# 由于2B的这些ID已经被占用，我们需要:
# 方案A: 移除2B中这些ID的token，然后添加10B定义的token
# 方案B: 保持现状，接受差异

# 这里我们选择方案A: 完全对齐
print("\n=== 执行完全对齐 ===")

# 1. 从vocab_2b中移除ID 155674-155685的token
# 2. 从added_tokens_2b中移除ID 155674-155685的token
# 3. 添加10B定义的token到这些ID

# 首先，找出需要移除的token
tokens_to_remove = []
for token, tid in list(vocab_2b.items()):
    if 155674 <= tid <= 155685:
        tokens_to_remove.append(token)

print(f"需要从vocab中移除的token: {len(tokens_to_remove)}个")
for token in tokens_to_remove:
    print(f"  移除: {token} (ID={vocab_2b[token]})")
    del vocab_2b[token]

# 从added_tokens中移除ID 155674-155685的token
added_to_remove = []
for tid_str, tinfo in list(added_tokens_2b.items()):
    if 155674 <= int(tid_str) <= 155685:
        added_to_remove.append(tid_str)

print(f"\n需要从added_tokens中移除的token: {len(added_to_remove)}个")
for tid_str in added_to_remove:
    print(f"  移除: ID={tid_str} ({added_tokens_2b[tid_str]['content']})")
    del added_tokens_2b[tid_str]

# 添加10B定义的token
print(f"\n添加10B定义的token:")
for key, tid in target_ids.items():
    token_name = f"<|{key}|>"
    added_tokens_2b[str(tid)] = {
        "content": token_name,
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True
    }
    print(f"  添加: {token_name} (ID={tid})")

# 移除ID > 155696的token（2B特有的）
print(f"\n移除2B特有的token (ID > 155696):")
tokens_to_remove_high = []
for token, tid in list(vocab_2b.items()):
    if tid > 155696:
        tokens_to_remove_high.append(token)

for token in tokens_to_remove_high:
    print(f"  移除: {token} (ID={vocab_2b[token]})")
    del vocab_2b[token]

added_to_remove_high = []
for tid_str, tinfo in list(added_tokens_2b.items()):
    if int(tid_str) > 155696:
        added_to_remove_high.append(tid_str)

for tid_str in added_to_remove_high:
    print(f"  移除: ID={tid_str} ({added_tokens_2b[tid_str]['content']})")
    del added_tokens_2b[tid_str]

# 保存修复后的tokenizer
print(f"\n保存修复后的tokenizer...")

# 更新tokenizer.json
tokenizer_2b["model"]["vocab"] = vocab_2b
with open("/gpfs-data/mikelee/alpamayo1_5_2b_init/tokenizer.json", "w") as f:
    json.dump(tokenizer_2b, f, indent=2)

# 更新tokenizer_config.json
config_2b["added_tokens_decoder"] = added_tokens_2b
with open("/gpfs-data/mikelee/alpamayo1_5_2b_init/tokenizer_config.json", "w") as f:
    json.dump(config_2b, f, indent=2)

print(f"✅ tokenizer已修复并保存")

# 验证
print(f"\n=== 验证修复结果 ===")

# 重新加载验证
with open("/gpfs-data/mikelee/alpamayo1_5_2b_init/tokenizer.json", "r") as f:
    tokenizer_check = json.load(f)

vocab_check = tokenizer_check["model"]["vocab"]

with open("/gpfs-data/mikelee/alpamayo1_5_2b_init/tokenizer_config.json", "r") as f:
    config_check = json.load(f)

added_check = config_check.get("added_tokens_decoder", {})

# 合并
full_vocab_check = {}
for token, tid in vocab_check.items():
    full_vocab_check[token] = tid
for tid_str, tinfo in added_check.items():
    full_vocab_check[tinfo["content"]] = int(tid_str)

print(f"修复后总token数: {len(full_vocab_check)}")
print(f"10B vocab_size: {config_10b.get('vocab_size')}")

if len(full_vocab_check) == config_10b.get('vocab_size'):
    print(f"✅ 2B和10B的vocab_size完全一致!")
else:
    print(f"⚠️ 仍有差异: {len(full_vocab_check) - config_10b.get('vocab_size')} 个token")

# 检查关键token
print(f"\n关键token验证:")
for key, tid in target_ids.items():
    token_name = f"<|{key}|>"
    if token_name in full_vocab_check:
        actual_id = full_vocab_check[token_name]
        match = "✅" if actual_id == tid else "❌"
        print(f"{match} {token_name}: ID={actual_id} (期望 {tid})")
    else:
        print(f"❌ {token_name}: 未找到")

# 检查是否有重复ID
from collections import Counter
id_counts = Counter(full_vocab_check.values())
duplicates = {k: v for k, v in id_counts.items() if v > 1}

if duplicates:
    print(f"\n❌ 发现 {len(duplicates)} 个重复ID!")
else:
    print(f"\n✅ 没有重复ID，所有ID唯一!")

print("\n" + "=" * 70)
print("修复完成!")
print("=" * 70)
