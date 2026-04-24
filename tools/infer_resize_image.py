#!/usr/bin/env python3
"""
图片预缩放脚本 - Alpamayo 1.5

功能: 将chunk目录下的camera图片从1920×1080缩放到576×320
用于批量推理时跳过模型内部的resize步骤，加快推理速度

使用方法:
    # 验证模式 - 处理单张图片
    python3 infer_resize_image.py --verify --input /path/to/000034.jpg

    # 批量处理 - 处理整个chunk
    python3 infer_resize_image.py --chunk 0 --workers 16

    # 批量处理 - 同时保存numpy原始数据（排除JPEG压缩影响）
    python3 infer_resize_image.py --chunk 0 --workers 16 --save_numpy

    # 批量处理 - 处理多个chunk
    python3 infer_resize_image.py --chunk 0,1,2 --workers 16

    # 清理_small文件和_small.npy文件
    python3 infer_resize_image.py --cleanup --chunk 0
"""

import os
import sys
import argparse
import math
import json
import time
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# Alpamayo 图像参数（与 helper.py 保持一致）
MIN_PIXELS = 163840
MAX_PIXELS = 196608
PATCH_SIZE = 16
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE  # 32

# 4个相机
CAMERA_ORDER = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]

# JPEG quality
JPEG_QUALITY = 95


def smart_resize(height: int, width: int, factor: int = FACTOR,
                 min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS):
    """
    复现 Qwen2VLImageProcessor 的 smart_resize 算法

    Args:
        height: 原始高度
        width: 原始宽度
        factor: patch_size * merge_size = 32
        min_pixels: 最小像素数
        max_pixels: 最大像素数

    Returns:
        (resize_height, resize_width)
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"Aspect ratio too extreme: {max(height, width) / min(height, width)}")

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


def resize_image(input_path: str, output_path: str = None, quality: int = JPEG_QUALITY,
                  save_numpy: bool = False) -> dict:
    """
    缩放单张图片

    Args:
        input_path: 输入图片路径
        output_path: 输出路径，默认在原目录加_small后缀
        quality: JPEG质量
        save_numpy: 是否保存resize后的原始数据为.npy文件（不含JPEG压缩）

    Returns:
        dict: 包含原始尺寸、目标尺寸、文件大小等信息
    """
    img = Image.open(input_path).convert('RGB')
    orig_width, orig_height = img.size

    # 计算目标尺寸
    target_height, target_width = smart_resize(orig_height, orig_width)
    target_size = (target_width, target_height)

    # 如果输出路径未指定，生成_small路径
    if output_path is None:
        stem = Path(input_path).stem
        parent = Path(input_path).parent
        output_path = parent / f"{stem}_small.jpg"

    # 缩放 (使用BICUBIC匹配Qwen2VLImageProcessor)
    img_resized = img.resize(target_size, Image.BICUBIC)

    # 保存 JPEG
    img_resized.save(output_path, 'JPEG', quality=quality)

    # 可选：保存 numpy 原始数据（resize后、JPEG压缩前）
    npy_path = None
    if save_numpy:
        npy_path = output_path.with_suffix('.npy')
        # 转为 (H, W, C) uint8 并保存
        npy_data = np.array(img_resized, dtype=np.uint8)
        np.save(npy_path, npy_data)

    # 清理
    img.close()
    img_resized.close()

    # 获取文件大小
    orig_size = os.path.getsize(input_path)
    new_size = os.path.getsize(output_path)

    return {
        'input_path': input_path,
        'output_path': output_path,
        'npy_path': npy_path,
        'orig_size': (orig_width, orig_height),
        'target_size': target_size,
        'orig_bytes': orig_size,
        'new_bytes': new_size,
        'compression_ratio': new_size / orig_size if orig_size > 0 else 0,
    }


def verify(input_path: str, quality: int = JPEG_QUALITY):
    """
    验证模式: 处理单张图片并显示详细信息
    """
    print(f"=" * 60)
    print(f"图片预缩放验证")
    print(f"=" * 60)
    print(f"输入文件: {input_path}")
    print(f"JPEG quality: {quality}")
    print()

    # 检查文件
    if not os.path.exists(input_path):
        print(f"错误: 文件不存在: {input_path}")
        return

    img = Image.open(input_path)
    orig_width, orig_height = img.size
    orig_size_bytes = os.path.getsize(input_path)
    print(f"原始尺寸: {orig_width} x {orig_height}")
    print(f"原始文件大小: {orig_size_bytes / 1024:.1f} KB")
    print()

    # 计算目标尺寸
    target_height, target_width = smart_resize(orig_height, orig_width)
    print(f"目标尺寸: {target_width} x {target_height} (W x H)")
    print(f"目标像素数: {target_width * target_height}")
    print()

    # 执行缩放
    stem = Path(input_path).stem
    parent = Path(input_path).parent
    output_path = parent / f"{stem}_small.jpg"

    start_time = time.time()
    # BICUBIC 匹配 Qwen2VLImageProcessor
    img_resized = img.resize((target_width, target_height), Image.BICUBIC)
    img_resized.save(output_path, 'JPEG', quality=quality)
    elapsed = time.time() - start_time

    new_size_bytes = os.path.getsize(output_path)

    print(f"输出文件: {output_path}")
    print(f"新文件大小: {new_size_bytes / 1024:.1f} KB")
    print(f"压缩比: {orig_size_bytes / new_size_bytes:.1f}x")
    print(f"处理耗时: {elapsed * 1000:.1f} ms")
    print()

    # 验证尺寸
    verify_img = Image.open(output_path)
    verify_width, verify_height = verify_img.size
    print(f"验证 - 输出图片尺寸: {verify_width} x {verify_height}")
    if verify_width == target_width and verify_height == target_height:
        print("✅ 尺寸验证通过")
    else:
        print(f"❌ 尺寸不匹配! 期望 {target_width}x{target_height}, 得到 {verify_width}x{verify_height}")

    img.close()
    img_resized.close()
    verify_img.close()


def process_single_image(args):
    """处理单张图片（用于并行）"""
    img_path, save_numpy, quality = args
    try:
        stem = Path(img_path).stem
        parent = Path(img_path).parent
        output_path = parent / f"{stem}_small.jpg"
        npy_path = parent / f"{stem}_small.npy"

        # 如果已存在则跳过
        if os.path.exists(output_path) and (not save_numpy or os.path.exists(npy_path)):
            return None

        img = Image.open(img_path).convert('RGB')
        orig_width, orig_height = img.size
        target_height, target_width = smart_resize(orig_height, orig_width)
        img_resized = img.resize((target_width, target_height), Image.BICUBIC)

        # 保存 JPEG
        img_resized.save(output_path, 'JPEG', quality=quality)

        # 可选：保存 numpy
        if save_numpy:
            npy_data = np.array(img_resized, dtype=np.uint8)
            np.save(npy_path, npy_data)

        img.close()
        img_resized.close()

        orig_size = os.path.getsize(img_path)
        new_size = os.path.getsize(output_path)

        return {
            'orig_size': orig_size,
            'new_size': new_size,
            'success': True
        }
    except Exception as e:
        return {
            'path': img_path,
            'error': str(e),
            'success': False
        }


def process_chunk(chunk_id: int, base_dir: str = "/data01/mikelee/data",
                 workers: int = 16, verbose: bool = True,
                 save_numpy: bool = False, quality: int = JPEG_QUALITY) -> dict:
    """
    处理整个chunk的所有图片

    Args:
        chunk_id: chunk编号
        base_dir: 数据根目录
        workers: 并行worker数
        verbose: 是否显示进度条
        save_numpy: 是否保存resize后原始数据为.npy
        quality: JPEG质量

    Returns:
        dict: 处理统计
    """
    chunk_dir = f"{base_dir}/data_sample_chunk{chunk_id}/infer"
    if not os.path.exists(chunk_dir):
        raise FileNotFoundError(f"Chunk目录不存在: {chunk_dir}")

    # 收集所有需要处理的图片
    image_files = []
    for root, dirs, files in os.walk(chunk_dir):
        for f in files:
            if f.endswith('.jpg') and '_small' not in f:
                image_files.append(os.path.join(root, f))

    if verbose:
        print(f"chunk{chunk_id}: 找到 {len(image_files)} 张图片")

    if not image_files:
        return {'chunk_id': chunk_id, 'total': 0, 'skipped': 0, 'processed': 0}

    # 并行处理
    total_orig = 0
    total_new = 0
    processed = 0
    skipped = 0

    with mp.Pool(workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, [(f, save_numpy, quality) for f in image_files]),
            total=len(image_files),
            desc=f"chunk{chunk_id}",
            disable=not verbose
        ))

    for r in results:
        if r is None:
            skipped += 1
        elif r['success']:
            total_orig += r['orig_size']
            total_new += r['new_size']
            processed += 1
        else:
            if verbose:
                print(f"错误: {r.get('path', 'unknown')}: {r.get('error', 'unknown')}")

    return {
        'chunk_id': chunk_id,
        'total': len(image_files),
        'processed': processed,
        'skipped': skipped,
        'total_orig_bytes': total_orig,
        'total_new_bytes': total_new,
        'space_saved_bytes': total_orig - total_new,
    }


def cleanup_chunk(chunk_id: int, base_dir: str = "/data01/mikelee/data", verbose: bool = True):
    """
    清理chunk的_small文件

    Args:
        chunk_id: chunk编号
        base_dir: 数据根目录
        verbose: 是否显示进度条
    """
    chunk_dir = f"{base_dir}/data_sample_chunk{chunk_id}/infer"

    small_files = []
    for root, dirs, files in os.walk(chunk_dir):
        for f in files:
            if f.endswith('_small.jpg') or f.endswith('_small.npy'):
                small_files.append(os.path.join(root, f))

    if verbose:
        print(f"chunk{chunk_id}: 找到 {len(small_files)} 个_small文件待删除")

    total_size = 0
    for f in tqdm(small_files, desc=f"chunk{chunk_id} cleanup", disable=not verbose):
        total_size += os.path.getsize(f)
        os.remove(f)

    if verbose:
        print(f"已删除 {len(small_files)} 个文件，释放 {total_size / 1024**3:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="图片预缩放脚本 - Alpamayo 1.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 验证模式
    python3 infer_resize_image.py --verify --input /path/to/000034.jpg

    # 批量处理
    python3 infer_resize_image.py --chunk 0 --workers 16

    # 批量处理 + 保存numpy（用于验证resize精度）
    python3 infer_resize_image.py --chunk 0 --workers 16 --save_numpy

    # 清理（同时删除_small.jpg和_small.npy）
    python3 infer_resize_image.py --cleanup --chunk 0
        """
    )
    parser.add_argument('--verify', action='store_true',
                        help='验证模式，处理单张图片')
    parser.add_argument('--input', type=str,
                        help='验证模式的输入图片路径')
    parser.add_argument('--chunk', type=str, default='',
                        help='chunk编号，支持逗号分隔如 "0,1,2" 或范围如 "0-5"')
    parser.add_argument('--base_dir', type=str, default='/data01/mikelee/data',
                        help='数据根目录')
    parser.add_argument('--workers', type=int, default=16,
                        help='并行worker数 (默认: 16)')
    parser.add_argument('--quality', type=int, default=JPEG_QUALITY,
                        help=f'JPEG质量 (默认: {JPEG_QUALITY})')
    parser.add_argument('--save_numpy', action='store_true',
                        help='同时保存resize后、JPEG压缩前的原始数据为.npy文件')
    parser.add_argument('--cleanup', action='store_true',
                        help='清理模式，删除_small文件')
    parser.add_argument('--dry_run', action='store_true',
                        help='干跑模式，不实际处理')

    args = parser.parse_args()

    # 验证模式
    if args.verify:
        if not args.input:
            print("错误: --verify 模式需要 --input 参数")
            sys.exit(1)
        verify(args.input, args.quality)
        return

    # 清理模式
    if args.cleanup:
        if not args.chunk:
            print("错误: --cleanup 模式需要 --chunk 参数")
            sys.exit(1)
        chunk_ids = []
        for part in args.chunk.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-')
                chunk_ids.extend(range(int(start), int(end) + 1))
            else:
                chunk_ids.append(int(part))
        for cid in chunk_ids:
            cleanup_chunk(cid, args.base_dir)
        return

    # 批量处理模式
    if not args.chunk:
        print("错误: 需要指定 --chunk 参数")
        parser.print_help()
        sys.exit(1)

    # 解析chunk列表
    chunk_ids = []
    for part in args.chunk.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            chunk_ids.extend(range(int(start), int(end) + 1))
        else:
            chunk_ids.append(int(part))

    print(f"=" * 60)
    print(f"图片预缩放 - Alpamayo 1.5")
    print(f"=" * 60)
    print(f"Chunk列表: {chunk_ids}")
    print(f"Workers: {args.workers}")
    print(f"JPEG quality: {args.quality}")
    print(f"保存numpy: {args.save_numpy}")
    print(f"数据目录: {args.base_dir}")
    print()

    if args.dry_run:
        for cid in chunk_ids:
            chunk_dir = f"{args.base_dir}/data_sample_chunk{cid}/infer"
            if os.path.exists(chunk_dir):
                count = sum(1 for _ in Path(chunk_dir).rglob('*.jpg') if '_small' not in _.name)
                print(f"chunk{cid}: {count} 张图片 (干跑)")
        return

    total_orig = 0
    total_new = 0
    total_processed = 0
    total_skipped = 0

    for cid in chunk_ids:
        print(f"\n处理 chunk{cid}...")
        result = process_chunk(cid, args.base_dir, args.workers, save_numpy=args.save_numpy, quality=args.quality)
        total_orig += result.get('total_orig_bytes', 0)
        total_new += result.get('total_new_bytes', 0)
        total_processed += result.get('processed', 0)
        total_skipped += result.get('skipped', 0)

        print(f"  已处理: {result['processed']} 张")
        print(f"  跳过: {result['skipped']} 张")
        if result['processed'] > 0:
            saved = result['total_orig_bytes'] - result['total_new_bytes']
            print(f"  节省空间: {saved / 1024**2:.1f} MB")

    print()
    print(f"=" * 60)
    print(f"全部完成!")
    print(f"总处理: {total_processed} 张")
    print(f"总跳过: {total_skipped} 张")
    if total_processed > 0:
        print(f"原始大小: {total_orig / 1024**3:.2f} GB")
        print(f"缩放后大小: {total_new / 1024**3:.2f} GB")
        print(f"总共节省: {(total_orig - total_new) / 1024**3:.2f} GB ({(1 - total_new/total_orig)*100:.1f}%%)")


if __name__ == "__main__":
    main()
