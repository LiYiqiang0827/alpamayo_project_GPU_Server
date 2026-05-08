#!/usr/bin/env python3
"""
验证added_tokens_decoder中的token ID
"""

import json

# 加载tokenizer_config.json
with open("/gpfs-data/mikelee/alpamayo1_5_2b_init/tokenizer_config.json", "r") as f:
    config = json.load(f)

added_tokens = config.get("added_tokens_decoder", {})

print("=== Added Tokens ===")
for token_id, token_info in sorted(added_tokens.items(), key=lambda x: int(x[0])):
    print(f"  ID={token_id:6s}: {token_info['content']}")

# 检查关键token
key_tokens = {
    "151644": "<|im_start|>",
    "151645": "<|im_end|>",
    "151646": "<|object_ref_start|>",
    "151647": "<|object_ref_end|>",
    "151648": "<|box_start|>",
    "151649": "<|box_end|>",
    "151650": "<|quad_start|>",
    "151651": "<|quad_end|>",
    "151652": "<|vision_start|>",
    "151653": "<|vision_end|>",
    "151654": "<|vision_pad|>",
    "151655": "<|image_pad|>",
    "151656": "<|video_pad|>",
}

print("\n=== 关键Token验证 ===")
for expected_id, expected_token in key_tokens.items():
    if expected_id in added_tokens:
        actual_token = added_tokens[expected_id]["content"]
        match = "✅" if actual_token == expected_token else "❌"
        print(f"{match} ID={expected_id}: {actual_token}")
    else:
        print(f"❌ ID={expected_id}: 未找到")

# 检查我们的特殊token
print("\n=== 我们的特殊Token ===")
our_special_tokens = [
    "<|cot_start|>", "<|cot_end|>",
    "<|traj_history_start|>", "<|traj_history|>", "<|traj_history_end|>",
    "<|traj_future_start|>", "<|traj_future|>", "<|traj_future_end|>",
    "<|route_start|>", "<|route_end|>",
]

for token in our_special_tokens:
    # 在added_tokens中查找
    found = False
    for tid, tinfo in added_tokens.items():
        if tinfo["content"] == token:
            print(f"  ✅ {token}: ID={tid}")
            found = True
            break
    if not found:
        print(f"  ❌ {token}: 未找到")
