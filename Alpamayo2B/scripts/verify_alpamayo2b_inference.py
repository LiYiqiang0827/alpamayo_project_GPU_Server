#!/usr/bin/env python3
"""
Alpamayo2B 推理验证脚本
验证输入处理流程：图片加载、resize、chat template、token embedding等
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from einops import rearrange

# 设置离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

# 路径配置
MODEL_PATH = "/gpfs-data/mikelee/alpamayo1_5_2b_init"
DATA_BASE = "/data01/mikelee/data"

# 相机顺序（与Alpamayo1.5一致）
CAMERA_ORDER = [
    'camera_cross_left_120fov',
    'camera_front_wide_120fov', 
    'camera_cross_right_120fov',
    'camera_front_tele_30fov'
]


def load_images_for_frame(row, data_dir):
    """
    加载图片并验证
    
    输入: inference_index_strict.csv的一行
    输出: 16张图片的tensor (4相机 × 4时间帧)
    """
    print("\n=== 图片加载 ===")
    images = []
    
    for cam in CAMERA_ORDER:
        for t in range(4):
            idx_col = f'{cam}_f{t}_idx'
            frame_idx = int(row[idx_col])
            
            # 尝试加载_small.jpg（预缩放的）
            img_path_small = f'{data_dir}/camera_images/{cam}/{frame_idx:06d}_small.jpg'
            img_path_full = f'{data_dir}/camera_images/{cam}/{frame_idx:06d}.jpg'
            
            if os.path.exists(img_path_small):
                img_path = img_path_small
            elif os.path.exists(img_path_full):
                img_path = img_path_full
            else:
                print(f"  ⚠️  图片不存在: {img_path_small}")
                continue
            
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            images.append(img_array)
            
            if len(images) == 1:
                print(f"  第一张图片尺寸: {img_array.shape} (H×W×C)")
    
    images = np.stack(images, axis=0)  # (16, H, W, 3)
    print(f"  堆叠后形状: {images.shape}")
    
    # rearrange: (16, H, W, 3) -> (4相机, 4时间帧, 3通道, H, W)
    images = rearrange(images, '(c t) h w ch -> c t ch h w', c=4, t=4)
    print(f"  rearrange后: {images.shape} (相机×时间×通道×H×W)")
    
    return torch.from_numpy(images).float()


def build_chat_template(images, history_traj=None, nav_text=None):
    """
    构建Alpamayo1.5格式的chat template
    
    验证点:
    1. 图片占位符数量 (16个)
    2. 轨迹占位符 (48个)
    3. 特殊token是否正确
    """
    print("\n=== Chat Template构建 ===")
    
    # 图片占位符: 每张图对应 <|vision_start|><|image_pad|><|vision_end|>
    image_placeholders = ""
    for i in range(16):
        image_placeholders += "<|vision_start|><|image_pad|><|vision_end|>"
    
    print(f"  图片占位符数量: {image_placeholders.count('<|vision_start|>')} 个")
    print(f"  图片占位符长度: {len(image_placeholders)} 字符")
    
    # 轨迹占位符
    traj_text = "<|traj_history_start|>" + "<|traj_history|>" * 48 + "<|traj_history_end|>"
    print(f"  轨迹占位符: 48个 <|traj_history|>")
    
    # 导航指令（可选）
    route_text = ""
    if nav_text:
        route_text = f"<|route_start|>{nav_text}<|route_end|>"
        print(f"  导航指令: {nav_text}")
    
    # 构建messages
    messages = [
        {
            "role": "system",
            "content": "You are a driving assistant that generates safe and accurate actions."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{image_placeholders}{traj_text}{route_text}output the chain-of-thought reasoning of the driving process, then output the future trajectory."}
            ]
        },
    ]
    
    return messages


def verify_tokenization(processor, messages, images):
    """
    验证tokenization过程
    
    检查点:
    1. 特殊token是否被正确识别
    2. image_pad token的数量
    3. 总token数
    """
    print("\n=== Tokenization验证 ===")
    
    # 应用chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    print(f"  Chat template后的文本长度: {len(text)} 字符")
    print(f"  包含 <|vision_start|>: {text.count('<|vision_start|>')} 次")
    print(f"  包含 <|image_pad|>: {text.count('<|image_pad|>')} 次")
    print(f"  包含 <|traj_history|>: {text.count('<|traj_history|>')} 次")
    
    # Tokenization
    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt",
        padding=True,
    )
    
    input_ids = inputs['input_ids']
    print(f"\n  Tokenization结果:")
    print(f"    input_ids shape: {input_ids.shape}")
    print(f"    总token数: {input_ids.shape[1]}")
    
    # 统计特殊token
    tokenizer = processor.tokenizer
    image_pad_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    traj_history_id = tokenizer.convert_tokens_to_ids('<|traj_history|>')
    
    image_pad_count = (input_ids == image_pad_id).sum().item()
    vision_start_count = (input_ids == vision_start_id).sum().item()
    traj_history_count = (input_ids == traj_history_id).sum().item()
    
    print(f"    image_pad token (ID={image_pad_id}): {image_pad_count} 个")
    print(f"    vision_start token (ID={vision_start_id}): {vision_start_count} 个")
    print(f"    traj_history token (ID={traj_history_id}): {traj_history_count} 个")
    
    # 检查pixel_values
    if 'pixel_values' in inputs:
        pixel_values = inputs['pixel_values']
        print(f"\n    pixel_values shape: {pixel_values.shape}")
        if len(pixel_values.shape) == 3:
            print(f"    这表示: {pixel_values.shape[0]} patches, {pixel_values.shape[1]} features")
        elif len(pixel_values.shape) == 2:
            print(f"    这表示: {pixel_values.shape[0]} patches, {pixel_values.shape[1]} features")
        else:
            print(f"    形状: {pixel_values.shape}")
    
    # 检查image_grid_thw
    if 'image_grid_thw' in inputs:
        image_grid_thw = inputs['image_grid_thw']
        print(f"    image_grid_thw shape: {image_grid_thw.shape}")
        print(f"    内容: {image_grid_thw}")
    
    return inputs


def verify_model_input(model, inputs):
    """
    验证模型输入
    
    检查点:
    1. 模型是否能正确接收输入
    2. embedding查找是否正确
    3. 图片token是否被正确替换
    """
    print("\n=== 模型输入验证 ===")
    
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    pixel_values = inputs.get('pixel_values', None)
    if pixel_values is not None:
        pixel_values = pixel_values.to(model.device)
    image_grid_thw = inputs.get('image_grid_thw', None)
    if image_grid_thw is not None:
        if isinstance(image_grid_thw, torch.Tensor):
            if image_grid_thw.dim() == 3 and image_grid_thw.shape[0] == 1:
                image_grid_thw = image_grid_thw.squeeze(0)
            image_grid_thw = image_grid_thw.to(model.device).long()
    
    print(f"  input_ids device: {input_ids.device}")
    print(f"  input_ids shape: {input_ids.shape}")
    
    # 检查embedding层
    embed_layer = model.language_model.embed_tokens
    vocab_size, hidden_size = embed_layer.weight.shape
    print(f"\n  Embedding层:")
    print(f"    vocab_size: {vocab_size}")
    print(f"    hidden_size: {hidden_size}")
    print(f"    权重范围: [{embed_layer.weight.min():.4f}, {embed_layer.weight.max():.4f}]")
    print(f"    权重std: {embed_layer.weight.std():.4f}")
    
    # 尝试前向传播（不计算梯度）
    print(f"\n  尝试前向传播...")
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        
        logits = outputs.logits
        print(f"    ✅ 前向传播成功!")
        print(f"    logits shape: {logits.shape}")
        print(f"    logits范围: [{logits.min():.2f}, {logits.max():.2f}]")
        
    except Exception as e:
        print(f"    ❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Alpamayo2B 推理验证')
    parser.add_argument('--clip', type=str, required=True, help='Clip ID')
    parser.add_argument('--chunk', type=int, default=0, help='Chunk ID')
    parser.add_argument('--frame', type=int, default=0, help='Frame ID')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    args = parser.parse_args()
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    print("=" * 70)
    print("Alpamayo2B 推理验证")
    print("=" * 70)
    print(f"\n模型路径: {MODEL_PATH}")
    print(f"数据路径: {DATA_BASE}/data_sample_chunk{args.chunk}")
    print(f"Clip ID: {args.clip}")
    print(f"Frame ID: {args.frame}")
    
    # 1. 加载模型
    print("\n" + "=" * 70)
    print("1. 加载模型")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"模型加载完成: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M 参数")
    
    # 2. 加载数据
    print("\n" + "=" * 70)
    print("2. 加载数据")
    print("=" * 70)
    
    data_dir = f"{DATA_BASE}/data_sample_chunk{args.chunk}/infer/{args.clip}/data"
    index_csv = f"{data_dir}/inference_index_strict.csv"
    
    if not os.path.exists(index_csv):
        print(f"❌ 索引文件不存在: {index_csv}")
        return
    
    df = pd.read_csv(index_csv)
    row = df[df['frame_id'] == args.frame].iloc[0]
    
    print(f"找到 frame {args.frame} 的数据")
    
    # 3. 加载图片
    images = load_images_for_frame(row, data_dir)
    
    # 4. 构建chat template
    messages = build_chat_template(images)
    
    # 5. 验证tokenization
    inputs = verify_tokenization(processor, messages, images.flatten(0, 1))
    
    # 6. 验证模型输入
    verify_model_input(model, inputs)
    
    # 7. 尝试生成
    print("\n" + "=" * 70)
    print("7. 尝试生成")
    print("=" * 70)
    
    try:
        with torch.no_grad():
            # 确保所有输入都在同一设备
            device = model.device
            generate_inputs = {
                'input_ids': inputs['input_ids'].to(device),
                'attention_mask': inputs['attention_mask'].to(device),
            }
            
            if 'pixel_values' in inputs:
                generate_inputs['pixel_values'] = inputs['pixel_values'].to(device)
            
            if 'image_grid_thw' in inputs:
                image_grid_thw = inputs['image_grid_thw']
                if isinstance(image_grid_thw, torch.Tensor):
                    if image_grid_thw.dim() == 3 and image_grid_thw.shape[0] == 1:
                        image_grid_thw = image_grid_thw.squeeze(0)
                    generate_inputs['image_grid_thw'] = image_grid_thw.to(device).long()
            
            generated_ids = model.generate(
                **generate_inputs,
                max_new_tokens=50,
                do_sample=False,
            )
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        print(f"\n生成结果 (前200字符):")
        print(f"{generated_text[:200]}...")
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("验证完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
