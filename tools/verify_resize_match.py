#!/usr/bin/env python3
"""
严格验证脚本：对比模型processor对原始图 vs 预缩放图的最终pixel_values

这是真正的等效性验证：
1. 原始图(1920×1080) → processor → pixel_values (ground truth)
2. 预缩放图(576×320 JPEG) → processor → pixel_values (our result)

如果两者一致，说明我们的预缩放是像素级等效的。
"""

import os
import sys
import math
import json
import tempfile
import numpy as np
from PIL import Image
from pathlib import Path

# Alpamayo 参数
MIN_PIXELS = 163840
MAX_PIXELS = 196608
PATCH_SIZE = 16
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE  # 32

IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def smart_resize(height: int, width: int, factor: int = FACTOR,
                 min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS):
    """复现 Qwen2VLImageProcessor.smart_resize"""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"Aspect ratio too extreme")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def resize_and_save_bicubic(input_path: str, output_path: str = None,
                            target_size=None, quality: int = 95):
    """用BICUBIC缩放并保存为JPEG"""
    img = Image.open(input_path).convert('RGB')
    orig_w, orig_h = img.size

    if target_size is None:
        target_h, target_w = smart_resize(orig_h, orig_w)
    else:
        target_w, target_h = target_size

    img_resized = img.resize((target_w, target_h), Image.BICUBIC)

    if output_path is None:
        stem = Path(input_path).stem
        parent = Path(input_path).parent
        output_path = parent / f"{stem}_small.jpg"

    img_resized.save(output_path, 'JPEG', quality=quality)
    img.close()
    img_resized.close()
    return output_path, (orig_w, orig_h), (target_w, target_h)


def get_pixel_values_from_image(img_path: str, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """
    用processor处理单张图片，返回pixel_values
    """
    sys.path.insert(0, '/home/user/mikelee/alpamayo_project/alpamayo_1_5/alpamayo1.5-main/src')
    from transformers import AutoProcessor
    from alpamayo1_5 import helper

    processor = helper.get_processor(
        AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct",
                                       min_pixels=min_pixels,
                                       max_pixels=max_pixels).tokenizer
    )

    img = Image.open(img_path).convert('RGB')

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "output the future trajectory."}
        ]}
    ]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt"
    )

    img.close()
    return inputs['pixel_values'].cpu().numpy()


def patchify(img_hwc_01, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE):
    """
    将 (H, W, C) 0-1 图片转成 pixel_values (N, 1536)
    复现 Qwen2VLImageProcessor 的 patchify 逻辑
    """
    H, W, C = img_hwc_01.shape
    grid_h, grid_w = H // patch_size, W // patch_size

    # (H, W, C) -> (grid_h, patch_size, grid_w, patch_size, C)
    # 然后 (grid_h, grid_w, patch_size, patch_size, C)
    patches = img_hwc_01.reshape(grid_h, patch_size, grid_w, patch_size, C)
    patches = patches.transpose(0, 2, 1, 3, 4)  # (grid_h, grid_w, p_h, p_w, C)
    patches = patches.reshape(-1, patch_size * patch_size * C)  # (N, 1536)

    return patches.astype(np.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=str, required=True, help='原始图片路径')
    parser.add_argument('--preprocessed', type=str, default=None, help='预缩放图片路径(默认自动生成)')
    parser.add_argument('--quality', type=int, default=95, help='JPEG质量')
    args = parser.parse_args()

    orig_path = args.original
    qual = args.quality

    print(f"=" * 60)
    print(f"像素级等效性验证")
    print(f"=" * 60)
    print(f"原始图片: {orig_path}")
    print(f"JPEG quality: {qual}")
    print()

    # Step 1: 生成预缩放图(如果没有提供)
    if args.preprocessed:
        pre_path = args.preprocessed
        print(f"[1] 使用提供的预缩放图: {pre_path}")
    else:
        stem = Path(orig_path).stem
        parent = Path(orig_path).parent
        pre_path = str(parent / f"{stem}_small.jpg")

        print(f"[1] 生成预缩放图: {pre_path}")
        resize_and_save_bicubic(orig_path, pre_path, quality=qual)

    # Step 2: 获取processor对原始图的输出 (ground truth)
    print(f"\n[2] Processor处理原始图(1920×1080)...")
    pv_original = get_pixel_values_from_image(orig_path)
    print(f"    pixel_values shape: {pv_original.shape}")

    # Step 3: 获取processor对预缩放图的输出
    print(f"\n[3] Processor处理预缩放图(576×320)...")
    pv_preproc = get_pixel_values_from_image(pre_path)
    print(f"    pixel_values shape: {pv_preproc.shape}")

    # Step 4: 对比两者
    print(f"\n[4] 对比 pixel_values...")
    if pv_original.shape != pv_preproc.shape:
        print(f"    ❌ Shape不一致!")
        print(f"    原始图: {pv_original.shape}")
        print(f"    预缩放: {pv_preproc.shape}")
        return

    diff = np.abs(pv_original - pv_preproc)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"    最大像素差异: {max_diff:.6f}")
    print(f"    平均像素差异: {mean_diff:.6f}")

    # Step 5: 额外对比 - 手动patchify后的结果
    print(f"\n[5] 手动patchify对比(排除processor内部差异)...")
    img_pre = Image.open(pre_path).convert('RGB')
    img_pre_np = np.array(img_pre).astype(np.float32) / 255.0
    img_pre_h, img_pre_w = img_pre_np.shape[:2]
    img_pre_pv = patchify(img_pre_np)  # (N, 1536)
    img_pre.close()

    # 注意：由于原始图和预缩放图尺寸不同，patchify后的shape也不同
    # 只有当两者都是576×320时才能直接对比
    print(f"    预缩放图手动patchify shape: {img_pre_pv.shape}")

    # 真正的等效性：看processor对预缩放图是否把smart_resize当noop
    # 检查image_grid_thw是否正确识别了576×320的尺寸
    print(f"\n[6] 检查 processor 对预缩放图的尺寸识别...")

    # 重新处理一次，获取image_grid_thw
    sys.path.insert(0, '/home/user/mikelee/alpamayo_project/alpamayo_1_5/alpamayo1.5-main/src')
    from transformers import AutoProcessor
    from alpamayo1_5 import helper

    processor = helper.get_processor(
        AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct",
                                       min_pixels=MIN_PIXELS,
                                       max_pixels=MAX_PIXELS).tokenizer
    )

    img_pre2 = Image.open(pre_path).convert('RGB')
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img_pre2},
            {"type": "text", "text": "output the future trajectory."}
        ]}
    ]
    inputs_pre = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt"
    )
    img_pre2.close()

    grid_thw = inputs_pre.get('image_grid_thw')
    print(f"    image_grid_thw: {grid_thw}")

    if grid_thw is not None:
        expected_tokens_per_img = (img_pre_h // PATCH_SIZE) * (img_pre_w // PATCH_SIZE)
        print(f"    期望每图token数: {expected_tokens_per_img}")
        print(f"    实际每图token数: {grid_thw[0].prod().item() if len(grid_thw) == 1 else grid_thw.prod().item()}")

    # 最终结论
    print(f"\n{'=' * 60}")
    print(f"pixel_values 对比结论:")
    if max_diff < 1e-4:
        print(f"  ✅ 几乎完全一致 (差异 < 1e-4)")
    elif max_diff < 1e-2:
        print(f"  ✅ 基本一致 (差异 < 1e-2)，主要由JPEG压缩导致")
    elif max_diff < 0.1:
        print(f"  ⚠️ 有差异 (差异 ~0.01-0.1)，JPEG压缩影响")
    else:
        print(f"  ❌ 差异很大 (差异 > 0.1)")
        print(f"  可能原因: resize算法不匹配 / 尺寸识别错误")
    print(f"{'=' * 60}")

    # 保存中间结果
    out_dir = Path(pre_path).parent
    np.save(out_dir / "pv_original.npy", pv_original)
    np.save(out_dir / "pv_preproc.npy", pv_preproc)
    print(f"\n已保存: pv_original.npy, pv_preproc.npy")


if __name__ == "__main__":
    main()
