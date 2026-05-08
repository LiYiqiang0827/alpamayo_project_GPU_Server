#!/usr/bin/env python3
"""
LLM蒸馏训练 - 小规模验证脚本
使用少量数据快速验证训练流程
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    AutoTokenizer, 
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Qwen3VLConfig,
)

# 路径配置
TOKENIZER_PATH = "/data01/mikelee/weight/alpamayo2B/tokenizer_final"
MODEL_PATH = "/data01/mikelee/weight/alpamayo2B"
DATASET_PATH = "/gpfs-data/mikelee/llm_distillation_data/distillation_dataset.jsonl"
OUTPUT_DIR = "/gpfs-data/mikelee/llm_distillation_output_test"


class QuickTestDataset(Dataset):
    """快速测试数据集 - 只加载少量样本"""
    
    def __init__(self, dataset_path, tokenizer, processor, max_samples=100):
        self.tokenizer = tokenizer
        self.processor = processor
        self.samples = []
        
        # 只加载前max_samples个样本
        with open(dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                self.samples.append(json.loads(line))
        
        print(f"Loaded {len(self.samples)} samples for quick test")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图片（只加载前4张以节省内存）
        from PIL import Image
        images = []
        for img_path in sample['image_paths'][:4]:  # 只加载4张
            img = Image.open(img_path).convert('RGB')
            images.append(img)
        
        # 构建prompt - 需要包含图片占位符
        cot_text = sample['cot_text']
        
        # 构建包含图片占位符的prompt
        # 图片占位符: <|vision_start|><|image_pad|><|vision_end|>
        image_placeholders = ""
        for _ in images:
            image_placeholders += "<|vision_start|><|image_pad|><|vision_end|>"
        
        messages = [
            {
                "role": "system",
                "content": "You are a driving assistant that generates safe and accurate actions."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{image_placeholders}<|traj_history_start|><|traj_history|>*48<|traj_history_end|>output the chain-of-thought reasoning of the driving process, then output the future trajectory."}
                ]
            },
            {
                "role": "assistant",
                "content": f"<|cot_start|>{cot_text}<|cot_end|>"
            }
        ]
        
        # 应用chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # 处理图片和文本
        try:
            inputs = self.processor(
                text=[text],
                images=images,
                return_tensors="pt",
                padding=True,
            )
            
            # 移除batch维度
            inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # 确保image_grid_thw是tensor
            if 'image_grid_thw' in inputs and not isinstance(inputs['image_grid_thw'], torch.Tensor):
                inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw'])
            
            inputs['labels'] = inputs['input_ids'].clone()
            
            return inputs
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # 返回空样本
            return {
                'input_ids': torch.tensor([0]),
                'attention_mask': torch.tensor([1]),
                'labels': torch.tensor([0]),
            }


def quick_test():
    """快速测试训练流程"""
    print("=" * 70)
    print("LLM Distillation Quick Test")
    print("=" * 70)
    
    # 加载tokenizer和processor
    print("\n1. Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    # processor需要从模型路径加载，因为tokenizer路径缺少config
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"   Tokenizer vocab size: {len(tokenizer)}")
    
    # 加载模型
    print("\n2. Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 加载配置
    config = Qwen3VLConfig.from_pretrained(MODEL_PATH)
    print(f"   Model config: vocab_size={config.text_config.vocab_size}, hidden_size={config.text_config.hidden_size}")
    
    # 加载模型（使用fp16节省显存）
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model loaded: {total_params / 1e6:.0f}M parameters")
    
    # 冻结ViT
    print("\n3. Freezing Vision Encoder...")
    for param in model.visual.parameters():
        param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {trainable / 1e6:.0f}M / {total_params / 1e6:.0f}M ({trainable/total_params*100:.1f}%)")
    
    # 创建数据集
    print("\n4. Creating dataset...")
    dataset = QuickTestDataset(DATASET_PATH, tokenizer, processor, max_samples=10)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 测试前向传播
    print("\n5. Testing forward pass...")
    model.train()
    
    for i, batch in enumerate(dataloader):
        print(f"\n   Batch {i+1}/10:")
        
        # 将数据移到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch.get('pixel_values', None)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        image_grid_thw = batch.get('image_grid_thw', None)
        
        if image_grid_thw is not None:
            if isinstance(image_grid_thw, torch.Tensor):
                # 如果形状是 [1, N, 3]，需要reshape为 [N, 3]
                if image_grid_thw.dim() == 3 and image_grid_thw.shape[0] == 1:
                    image_grid_thw = image_grid_thw.squeeze(0)
                image_grid_thw = image_grid_thw.to(device).long()
            else:
                image_grid_thw = torch.tensor(image_grid_thw, device=device, dtype=torch.long)
        
        labels = batch['labels'].to(device)
        
        print(f"     input_ids shape: {input_ids.shape}")
        print(f"     pixel_values shape: {pixel_values.shape if pixel_values is not None else None}")
        
        # Forward
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
            )
            
            loss = outputs.loss
            print(f"     Loss: {loss.item():.4f}")
            
            # 只测试第一个batch
            if i == 0:
                print("\n   ✓ Forward pass successful!")
                break
                
        except Exception as e:
            print(f"     Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 70)
    print("Quick test completed successfully!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
