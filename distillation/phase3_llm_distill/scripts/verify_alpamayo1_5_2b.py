#!/usr/bin/env python3
"""
验证Alpamayo1.5-2B模型
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor

MODEL_PATH = "/gpfs-data/mikelee/alpamayo1_5_2b_init"

def verify_model():
    print("=" * 70)
    print("Verifying Alpamayo1.5-2B Model")
    print("=" * 70)
    
    # 1. 加载模型
    print("\n1. Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"   Model loaded successfully!")
    
    # 2. 检查参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params / 1e6:.0f}M")
    
    # 3. 加载tokenizer
    print("\n2. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"   Vocab size: {len(tokenizer)}")
    
    # 4. 检查特殊token
    print("\n3. Checking special tokens...")
    special_tokens = [
        "<|cot_start|>",
        "<|cot_end|>",
        "<|traj_history_start|>",
        "<|traj_history|>",
        "<|traj_history_end|>",
        "<|traj_future_start|>",
        "<|traj_future|>",
        "<|traj_future_end|>",
        "<|route_start|>",
        "<|route_end|>",
    ]
    
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"   {token}: {token_id}")
    
    # 5. 检查轨迹token
    print("\n4. Checking trajectory tokens...")
    traj_start = tokenizer.convert_tokens_to_ids("<i0>")
    traj_end = tokenizer.convert_tokens_to_ids("<i3999>")
    print(f"   Trajectory token range: {traj_start} - {traj_end}")
    
    # 6. 测试简单推理
    print("\n5. Testing simple inference...")
    messages = [
        {"role": "system", "content": "You are a driving assistant."},
        {"role": "user", "content": "What do you see?"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Generated: {response[:100]}...")
    
    print("\n" + "=" * 70)
    print("Verification complete!")
    print("=" * 70)

if __name__ == "__main__":
    verify_model()
