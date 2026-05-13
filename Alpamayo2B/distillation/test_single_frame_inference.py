#!/usr/bin/env python3
"""
单帧推理测试脚本
================
从数据集中选一帧，验证 Alpamayo2B 模型能否正常推理生成 CoT

不关注 CoT 内容是否正确，只验证 pipeline 能跑通
"""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor

# 添加工作目录到路径
sys.path.insert(0, '/home/user/mikelee/alpamayo_project/Alpamayo2B/distillation')

from dataloader_Alpamayo2B import AlpamayoDistillationDataset
from model_setup_Alpamayo2B import setup_model_for_distillation

# ==================== 配置 ====================
CONFIG = {
    "model_path": "/data01/mikelee/weight/alpamayo2B",
    "tokenizer_path": "/data01/mikelee/weight/alpamayo2B/tokenizer_final",
    "infer_result_csv": "/gpfs-data/mikelee/infer_result/infer_result_20260507_195923/infer_results_all.csv",
    "teacher_logits_dir": "/gpfs-data/mikelee/infer_result/infer_result_20260507_195923/logits",
    "device": "cuda:1",
    "dtype": torch.bfloat16,
    "sample_idx": 0,  # 选第0个样本
    "max_new_tokens": 256,  # 生成最多256个token
}


def test_single_frame_inference():
    """测试单帧推理"""
    print("=" * 70)
    print("单帧推理测试")
    print("=" * 70)
    
    # 1. 加载 tokenizer 和 processor
    print("\n[1/5] 加载 Tokenizer 和 Processor...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["tokenizer_path"],
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        CONFIG["model_path"],
        trust_remote_code=True,
    )
    print(f"  ✓ Tokenizer vocab size: {len(tokenizer)}")
    
    # 2. 加载数据集（取一帧）
    print("\n[2/5] 加载数据集样本...")
    dataset = AlpamayoDistillationDataset(
        infer_result_csv=CONFIG["infer_result_csv"],
        teacher_logits_dir=CONFIG["teacher_logits_dir"],
        tokenizer=tokenizer,
        processor=processor,
        temperature=2.0,
    )
    
    sample = dataset[CONFIG["sample_idx"]]
    print(f"  ✓ 样本索引: {CONFIG['sample_idx']}")
    print(f"  ✓ input_ids shape: {sample['input_ids'].shape}")
    print(f"  ✓ pixel_values shape: {sample['pixel_values'].shape}")
    print(f"  ✓ image_grid_thw shape: {sample['image_grid_thw'].shape}")
    
    # 3. 加载模型
    print("\n[3/5] 加载模型...")
    model, _, _ = setup_model_for_distillation(
        model_path=CONFIG["model_path"],
        tokenizer_path=CONFIG["tokenizer_path"],
        device=CONFIG["device"],
        dtype=CONFIG["dtype"],
    )
    model.eval()
    print(f"  ✓ 模型加载完成，设备: {CONFIG['device']}")
    
    # 4. 准备输入（确保数据类型正确）
    print("\n[4/5] 准备输入...")
    
    # 关键：image_grid_thw 必须是 (batch_size*num_images, 3) 的 2D tensor
    # 而不是 (batch_size, num_images, 3) 的 3D tensor
    input_ids = sample["input_ids"].unsqueeze(0).to(CONFIG["device"])
    attention_mask = sample["attention_mask"].unsqueeze(0).to(CONFIG["device"])
    
    # pixel_values: (batch_size, num_patches, patch_dim) -> 已经是 2D
    pixel_values = sample["pixel_values"].unsqueeze(0).to(CONFIG["device"])
    
    # image_grid_thw: 必须是 (batch_size*num_images, 3) 的 2D tensor
    image_grid_thw = sample["image_grid_thw"].to(CONFIG["device"])
    # 确保是 2D: (16, 3)
    if image_grid_thw.dim() == 3:
        image_grid_thw = image_grid_thw.squeeze(0)  # 去掉 batch 维度
    if image_grid_thw.dtype != torch.long:
        image_grid_thw = image_grid_thw.long()
    
    print(f"  ✓ input_ids: {input_ids.shape}")
    print(f"  ✓ attention_mask: {attention_mask.shape}")
    print(f"  ✓ pixel_values: {pixel_values.shape}")
    print(f"  ✓ image_grid_thw: {image_grid_thw.shape}, dtype: {image_grid_thw.dtype}")
    
    # 5. 推理生成
    print("\n[5/5] 生成 CoT...")
    print("  生成中...")
    
    with torch.no_grad():
        # 使用模型的 generate 方法
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=CONFIG["max_new_tokens"],
            do_sample=False,  # 贪心解码，确定性输出
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码输出
    generated_ids = outputs[0]
    input_length = input_ids.shape[1]
    generated_tokens = generated_ids[input_length:]  # 去掉输入部分
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    
    print(f"\n  ✓ 生成完成!")
    print(f"  ✓ 输入长度: {input_length} tokens")
    print(f"  ✓ 生成长度: {len(generated_tokens)} tokens")
    
    # 打印结果
    print("\n" + "=" * 70)
    print("生成的 CoT (原始):")
    print("=" * 70)
    print(generated_text)
    
    print("\n" + "=" * 70)
    print("生成的 CoT (去掉特殊token):")
    print("=" * 70)
    generated_text_clean = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(generated_text_clean)
    
    # 检查是否包含轨迹token
    traj_tokens = [t for t in generated_tokens.tolist() if 151669 <= t <= 155668]
    print(f"\n  轨迹token数量: {len(traj_tokens)}")
    if traj_tokens:
        print(f"  轨迹token范围: {min(traj_tokens)} - {max(traj_tokens)}")
    
    print("\n" + "=" * 70)
    print("✅ 单帧推理测试完成!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    try:
        test_single_frame_inference()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
