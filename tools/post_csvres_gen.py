#!/usr/bin/env python3
"""
post_csvres_gen.py - 推理结果 CSV 汇总生成工具

从推理结果目录的 cot/ 子目录中读取 CoT 文本文件，生成 infer_results_all.csv。
适用于推理过程被提前终止、未生成汇总 CSV 的场景。

用法:
    python3 post_csvres_gen.py --infer_result infer_result_20260507_195923
    python3 post_csvres_gen.py --infer_result /gpfs-data/mikelee/infer_result/infer_result_20260507_195923
    python3 post_csvres_gen.py --infer_result infer_result_20260507_195923 --parallel 20

输入:
    推理结果目录 (包含 cot/ 子目录，内有 chunk{chunk_id:04d}_{clip_id}_{frame_id:06d}_cot.txt 文件)

输出:
    {infer_result_dir}/infer_results_all.csv
    列: chunk_id, worker_id, clip_name, frame_number, cot_result
"""

import os
import sys
import re
import argparse
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm


# ===================== 配置 =====================
INFER_RESULT_BASE = "/gpfs-data/mikelee/infer_result"


# ===================== 路径解析 =====================

def resolve_infer_result_path(infer_result_arg: str) -> str:
    """将推理结果参数扩展为完整路径"""
    if infer_result_arg.startswith("/"):
        return infer_result_arg
    if infer_result_arg.startswith("infer_result_"):
        return os.path.join(INFER_RESULT_BASE, infer_result_arg)
    # 可能是裸名称
    return os.path.join(INFER_RESULT_BASE, f"infer_result_{infer_result_arg}")


# ===================== 文件名解析 =====================

def parse_cot_filename(filename: str) -> Optional[Tuple[int, str, int]]:
    """
    从 CoT 文件名解析 chunk_id, clip_name, frame_number
    
    文件名格式: chunk{chunk_id:04d}_{clip_id}_{frame_id:06d}_cot.txt
    示例: chunk0000_01d3588e-bca7-4a18-8e74-c6cfe9e996db_000000_cot.txt
    
    Returns:
        (chunk_id, clip_name, frame_number) 或 None（解析失败）
    """
    # 去掉目录路径
    basename = os.path.basename(filename)
    
    # 正则匹配: chunk(\d{4})_(.+?)_(\d{6})_cot\.txt$
    pattern = r'^chunk(\d{4})_(.+?)_(\d{6})_cot\.txt$'
    match = re.match(pattern, basename)
    
    if match:
        chunk_id = int(match.group(1))
        clip_name = match.group(2)
        frame_number = int(match.group(3))
        return (chunk_id, clip_name, frame_number)
    
    return None


# ===================== 单文件处理 =====================

def process_single_cot(cot_file_path: str) -> Optional[Dict]:
    """
    处理单个 CoT 文件，返回 CSV 行数据
    
    Args:
        cot_file_path: CoT txt 文件完整路径
        
    Returns:
        dict 包含: chunk_id, worker_id, clip_name, frame_number, cot_result
        或 None（解析失败或读取失败）
    """
    parsed = parse_cot_filename(cot_file_path)
    if parsed is None:
        return None
    
    chunk_id, clip_name, frame_number = parsed
    
    try:
        # 读取 CoT 文本内容
        with open(cot_file_path, 'r', encoding='utf-8') as f:
            cot_result = f.read().strip()
    except Exception as e:
        print(f"  警告: 读取文件失败 {cot_file_path}: {e}")
        return None
    
    return {
        'chunk_id': chunk_id,
        'worker_id': -1,  # 后处理生成，标记为 -1
        'clip_name': clip_name,
        'frame_number': frame_number,
        'cot_result': cot_result,
    }


# ===================== 批量处理 =====================

def process_cot_batch(cot_files: List[str]) -> List[Dict]:
    """
    批量处理 CoT 文件（用于多进程）
    
    Args:
        cot_files: CoT 文件路径列表
        
    Returns:
        解析成功的结果列表
    """
    results = []
    for cot_file in cot_files:
        result = process_single_cot(cot_file)
        if result is not None:
            results.append(result)
    return results


# ===================== 主流程 =====================

def generate_csv_from_cot(infer_result_dir: str, parallel: int = 1):
    """
    从 CoT 目录生成汇总 CSV
    
    Args:
        infer_result_dir: 推理结果目录路径
        parallel: 并行进程数（默认1，单进程）
    """
    print("=" * 70)
    print("post_csvres_gen.py - 推理结果 CSV 汇总生成")
    print("=" * 70)
    
    # 检查目录
    if not os.path.exists(infer_result_dir):
        print(f"错误: 目录不存在: {infer_result_dir}")
        sys.exit(1)
    
    cot_dir = os.path.join(infer_result_dir, "cot")
    if not os.path.exists(cot_dir):
        print(f"错误: CoT 目录不存在: {cot_dir}")
        sys.exit(1)
    
    print(f"\n输入目录: {infer_result_dir}")
    print(f"CoT 目录: {cot_dir}")
    
    # 收集所有 CoT 文件
    print("\n[1/4] 扫描 CoT 文件...")
    cot_files = sorted(glob.glob(os.path.join(cot_dir, "*.txt")))
    print(f"  找到 {len(cot_files)} 个 CoT 文件")
    
    if len(cot_files) == 0:
        print("错误: 未找到任何 CoT 文件")
        sys.exit(1)
    
    # 处理文件
    print(f"\n[2/4] 解析 CoT 文件 (并行数: {parallel})...")
    all_results = []
    
    if parallel > 1:
        # 多进程处理
        batch_size = max(1, len(cot_files) // parallel)
        batches = [
            cot_files[i:i + batch_size] 
            for i in range(0, len(cot_files), batch_size)
        ]
        
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(process_cot_batch, batch): idx 
                for idx, batch in enumerate(batches)
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理批次"):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    print(f"  批次处理失败: {e}")
    else:
        # 单进程处理（带进度条）
        for cot_file in tqdm(cot_files, desc="解析 CoT 文件"):
            result = process_single_cot(cot_file)
            if result is not None:
                all_results.append(result)
    
    print(f"  成功解析 {len(all_results)} 条记录")
    
    # 构建 DataFrame
    print("\n[3/4] 构建 DataFrame...")
    df = pd.DataFrame(all_results)
    
    # 按 chunk_id, clip_name, frame_number 排序
    df = df.sort_values(['chunk_id', 'clip_name', 'frame_number']).reset_index(drop=True)
    
    print(f"  DataFrame 形状: {df.shape}")
    print(f"  列: {list(df.columns)}")
    print(f"  chunk 数: {df['chunk_id'].nunique()}")
    print(f"  clip 数: {df['clip_name'].nunique()}")
    print(f"  总帧数: {len(df)}")
    
    # 保存 CSV
    print("\n[4/4] 保存 CSV...")
    output_csv = os.path.join(infer_result_dir, "infer_results_all.csv")
    df.to_csv(output_csv, index=False)
    
    # 获取文件大小
    csv_size = os.path.getsize(output_csv)
    print(f"  CSV 已保存: {output_csv}")
    print(f"  文件大小: {csv_size / 1024 / 1024:.2f} MB")
    
    # 显示样本
    print("\n" + "=" * 70)
    print("样本数据（前3行）:")
    print("=" * 70)
    print(df.head(3).to_string())
    
    print("\n" + "=" * 70)
    print("✅ 完成!")
    print("=" * 70)
    
    return output_csv


# ===================== 命令行入口 =====================

def main():
    parser = argparse.ArgumentParser(
        description="从推理结果目录生成 infer_results_all.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用目录名（自动补全路径）
    python3 post_csvres_gen.py --infer_result infer_result_20260507_195923
    
    # 使用完整路径
    python3 post_csvres_gen.py --infer_result /gpfs-data/mikelee/infer_result/infer_result_20260507_195923
    
    # 并行处理（推荐用于大量文件）
    python3 post_csvres_gen.py --infer_result infer_result_20260507_195923 --parallel 20
        """
    )
    
    parser.add_argument(
        "--infer_result", "-i",
        required=True,
        help="推理结果目录（可接受: 完整路径、infer_result_前缀、或裸名称）"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        help="并行进程数（默认: 1，单进程）"
    )
    
    args = parser.parse_args()
    
    # 解析路径
    infer_result_dir = resolve_infer_result_path(args.infer_result)
    
    # 执行生成
    generate_csv_from_cot(infer_result_dir, parallel=args.parallel)


if __name__ == "__main__":
    main()
