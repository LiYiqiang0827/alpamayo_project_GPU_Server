#!/usr/bin/env python3
"""
验证Alpamayo2B和Alpamayo1.5-10B的token ID是否完全一致
这是最关键的一点！
"""

import sys
sys.path.insert(0, "/home/user/mikelee/alpamayo_project/alpamayo_1_5/alpamayo1.5-main/src")

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
import torch

# 加载2B的tokenizer和processor
print("=" * 70)
print("加载Alpamayo2B tokenizer...")
print("=" * 70)
tokenizer_2b = AutoTokenizer.from_pretrained(
    "/gpfs-data/mikelee/alpamayo1_5_2b_init",
    trust_remote_code=True,
    local_files_only=True,
)
processor_2b = AutoProcessor.from_pretrained(
    "/gpfs-data/mikelee/alpamayo1_5_2b_init",
    trust_remote_code=True,
    local_files_only=True,
)

print(f"2B Tokenizer vocab size: {len(tokenizer_2b)}")

# 加载10B的tokenizer（使用Alpamayo1.5代码）
print("\n" + "=" * 70)
print("加载Alpamayo1.5-10B tokenizer...")
print("=" * 70)

from alpamayo1_5.config import Alpamayo1_5Config
from transformers import AutoConfig
AutoConfig.register("alpamayo1_5", Alpamayo1_5Config)

tokenizer_10b = AutoTokenizer.from_pretrained(
    "/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B",
    trust_remote_code=True,
    local_files_only=True,
)
processor_10b = AutoProcessor.from_pretrained(
    "/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B",
    trust_remote_code=True,
    local_files_only=True,
)

print(f"10B Tokenizer vocab size: {len(tokenizer_10b)}")

# 构建相同的输入
print("\n" + "=" * 70)
print("构建相同的测试输入")
print("=" * 70)

# 创建测试图片
img = Image.new("RGB", (576, 320), color="red")
images = [img] * 16

# 构建chat template（与Alpamayo1.5一致）
messages = [
    {"role": "system", "content": "You are a driving assistant that generates safe and accurate actions."},
    {"role": "user", "content": [
        {"type": "text", "text": "<|vision_start|><|image_pad|><|vision_end|>" * 16 + 
         "<|traj_history_start|><|traj_history|>" * 48 + "<|traj_history_end|>" +
         "output the chain-of-thought reasoning of the driving process, then output the future trajectory."}
    ]},
]

# 2B tokenization
print("\n--- 2B Tokenization ---")
text_2b = processor_2b.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
inputs_2b = processor_2b(text=[text_2b], images=images, return_tensors="pt")
input_ids_2b = inputs_2b['input_ids'][0].tolist()

print(f"Text length: {len(text_2b)} chars")
print(f"Input IDs shape: {inputs_2b['input_ids'].shape}")
print(f"Total tokens: {len(input_ids_2b)}")

# 10B tokenization
print("\n--- 10B Tokenization ---")
text_10b = processor_10b.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
inputs_10b = processor_10b(text=[text_10b], images=images, return_tensors="pt")
input_ids_10b = inputs_10b['input_ids'][0].tolist()

print(f"Text length: {len(text_10b)} chars")
print(f"Input IDs shape: {inputs_10b['input_ids'].shape}")
print(f"Total tokens: {len(input_ids_10b)}")

# 对比token IDs
print("\n" + "=" * 70)
print("对比Token IDs")
print("=" * 70)

if len(input_ids_2b) != len(input_ids_10b):
    print(f"❌ Token数量不一致!")
    print(f"2B: {len(input_ids_2b)} tokens")
    print(f"10B: {len(input_ids_10b)} tokens")
else:
    print(f"✅ Token数量一致: {len(input_ids_2b)} tokens")
    
    # 逐个对比
    mismatches = []
    for i, (id_2b, id_10b) in enumerate(zip(input_ids_2b, input_ids_10b)):
        if id_2b != id_10b:
            mismatches.append({
                'position': i,
                'id_2b': id_2b,
                'id_10b': id_10b,
                'token_2b': tokenizer_2b.convert_ids_to_tokens([id_2b])[0] if id_2b < len(tokenizer_2b) else "UNKNOWN",
                'token_10b': tokenizer_10b.convert_ids_to_tokens([id_10b])[0] if id_10b < len(tokenizer_10b) else "UNKNOWN",
            })
    
    if mismatches:
        print(f"\n❌ 发现 {len(mismatches)} 个不一致的token!")
        print(f"\n前10个不一致:")
        for m in mismatches[:10]:
            print(f"  Position {m['position']}: 2B={m['id_2b']}({m['token_2b']}) vs 10B={m['id_10b']}({m['token_10b']})")
    else:
        print(f"\n✅ 所有 {len(input_ids_2b)} 个token ID完全一致!")

# 统计特殊token
print("\n" + "=" * 70)
print("特殊Token统计")
print("=" * 70)

special_tokens = [
    '<|im_start|>', '<|im_end|>', '<|vision_start|>', '<|vision_end|>', '<|image_pad|>',
    '<|traj_history_start|>', '<|traj_history|>', '<|traj_history_end|>',
    '<|traj_future_start|>', '<|traj_future|>', '<|traj_future_end|>',
    '<|route_start|>', '<|route_end|>', '<|cot_start|>', '<|cot_end|>',
]

print("\n2B特殊Token ID:")
for token in special_tokens:
    id_2b = tokenizer_2b.convert_tokens_to_ids(token)
    print(f"  {token:30s} ID={id_2b:6d}")

print("\n10B特殊Token ID:")
for token in special_tokens:
    id_10b = tokenizer_10b.convert_tokens_to_ids(token)
    print(f"  {token:30s} ID={id_10b:6d}")

# 对比特殊token
print("\n" + "=" * 70)
print("特殊Token对比")
print("=" * 70)

all_match = True
for token in special_tokens:
    id_2b = tokenizer_2b.convert_tokens_to_ids(token)
    id_10b = tokenizer_10b.convert_tokens_to_ids(token)
    match = "✅" if id_2b == id_10b else "❌"
    if id_2b != id_10b:
        all_match = False
    print(f"{match} {token:30s} 2B={id_2b:6d} 10B={id_10b:6d}")

if all_match:
    print(f"\n✅ 所有特殊Token ID完全一致!")
else:
    print(f"\n❌ 存在特殊Token ID不一致!")

print("\n" + "=" * 70)
print("最终结论")
print("=" * 70)
if len(input_ids_2b) == len(input_ids_10b) and len(mismatches) == 0 and all_match:
    print("✅✅✅ 2B和10B的Token ID完全一致! 可以开始训练! ✅✅✅")
else:
    print("❌❌❌ 存在差异，需要修复! ❌❌❌")
