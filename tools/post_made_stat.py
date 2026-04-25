#!/usr/bin/env python3
"""
post_made_stat.py - mADE 统计脚本

对 post_made_cal.py 的输出进行分组统计，按 (chunk_id, clip_name) 分组。

用法:
    python3 post_made_stat.py --infer_result infer_result_20260424_010530
    python3 post_made_stat.py --infer_result /data01/mikelee/infer_result/infer_result_20260424_010530

输入:
    {infer_result_dir}/post_made_all.csv

输出:
    {infer_result_dir}/post_made_stat.csv
    包含每个 clip 的 mADE 统计和去重后的 CoT
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Set, Dict
from collections import Counter

import numpy as np
import pandas as pd


# ===================== 配置 =====================
INFER_RESULT_BASE = "/data01/mikelee/infer_result"


# ===================== 路径构建 =====================

def resolve_infer_result_path(infer_result_arg: str) -> str:
    """将推理结果参数扩展为完整路径"""
    if infer_result_arg.startswith("/"):
        return infer_result_arg
    if infer_result_arg.startswith("infer_result_"):
        return os.path.join(INFER_RESULT_BASE, infer_result_arg)
    # 可能是裸名称
    return os.path.join(INFER_RESULT_BASE, f"infer_result_{infer_result_arg}")


# ===================== 统计计算 =====================

def format_cot_list(cot_series: pd.Series, frame_count: int) -> str:
    """
    将 CoT 系列统计频次并格式化为每行一个 CoT 的格式
    
    Args:
        cot_series: 包含 CoT 文本的 pandas Series
        frame_count: 该 clip 的总帧数，用于在末尾添加帧数统计
        
    Returns:
        格式化后的字符串，每行一个 CoT 及其出现次数，如:
        "{Keep lane} x 5,\n{Slow down} x 3,\n(共126帧)"
    """
    # 统计每个 CoT 的出现次数
    cot_counter: Dict[str, int] = Counter()
    for cot in cot_series.dropna():
        cot_str = str(cot).strip()
        if cot_str and cot_str.lower() != 'nan':
            cot_counter[cot_str] += 1
    
    # 按出现次数降序排序，次数相同则按文本排序
    sorted_cots = sorted(cot_counter.items(), key=lambda x: (-x[1], x[0]))
    
    # 格式化为每行一个 CoT，末尾加帧数统计
    if not sorted_cots:
        return f"(共{frame_count}帧)"
    
    formatted = ",\n".join(f"{{{cot}}} x {count}" for cot, count in sorted_cots)
    formatted += f",\n(共{frame_count}帧)"
    return formatted


def compute_clip_stats(group_df: pd.DataFrame) -> dict:
    """
    计算单个 clip 的统计信息
    
    Args:
        group_df: 包含单个 clip 所有帧的 DataFrame
        
    Returns:
        包含统计字段的字典
    """
    # 过滤有效的 mADE 值
    valid_made = group_df['mADE'].dropna()
    
    if len(valid_made) == 0:
        return {
            'mADE_min': np.nan,
            'mADE_max': np.nan,
            'mADE_mean': np.nan,
            'mADE_median': np.nan,
            'mADE_std': np.nan,
            'frame_count': len(group_df),
            'valid_frame_count': 0,
            'unique_cot_count': 0,
            'cot_list': ""
        }
    
    # CoT 统计（传入 frame_count 用于在末尾添加帧数统计）
    cot_list = format_cot_list(group_df['cot_result'], len(group_df))
    unique_cot_count = len([c for c in cot_list.split(",\n") if c and not c.startswith("(共")]) if cot_list else 0
    
    return {
        'mADE_min': float(valid_made.min()),
        'mADE_max': float(valid_made.max()),
        'mADE_mean': float(valid_made.mean()),
        'mADE_median': float(valid_made.median()),
        'mADE_std': float(valid_made.std()),
        'frame_count': len(group_df),
        'valid_frame_count': len(valid_made),
        'unique_cot_count': unique_cot_count,
        'cot_list': cot_list
    }


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(
        description='mADE 统计脚本 - 按 clip 分组统计'
    )
    parser.add_argument(
        '--infer_result', 
        type=str, 
        required=True,
        help='推理结果名称或路径, 如 infer_result_20260424_010530'
    )
    args = parser.parse_args()

    # 解析路径
    infer_result_dir = resolve_infer_result_path(args.infer_result)
    if not os.path.exists(infer_result_dir):
        print(f"错误: 推理结果目录不存在: {infer_result_dir}")
        sys.exit(1)

    input_csv = os.path.join(infer_result_dir, "post_made_all.csv")
    if not os.path.exists(input_csv):
        print(f"错误: post_made_all.csv 不存在: {input_csv}")
        print("请先运行 post_made_cal.py 计算 mADE")
        sys.exit(1)

    output_csv = os.path.join(infer_result_dir, "post_made_stat.csv")

    print(f"{'=' * 60}")
    print(f"mADE 统计 - 按 Clip 分组")
    print(f"{'=' * 60}")
    print(f"输入文件: {input_csv}")
    print(f"输出文件: {output_csv}")
    print()

    # Step 1: 读取 post_made_all.csv
    print(f"[1/3] 读取 post_made_all.csv ...")
    df = pd.read_csv(input_csv)
    total_rows = len(df)
    print(f"  总帧数: {total_rows}")

    # 检查必需列
    required_cols = ['chunk_id', 'clip_name', 'mADE', 'cot_result']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"错误: 缺少必需列: {missing_cols}")
        print(f"可用列: {list(df.columns)}")
        sys.exit(1)

    # Step 2: 按 (chunk_id, clip_name) 分组统计
    print(f"[2/3] 按 (chunk_id, clip_name) 分组统计 ...")
    
    # 确保 chunk_id 和 clip_name 存在
    group_cols = ['chunk_id', 'clip_name']
    
    # 分组并计算统计
    stats_list = []
    grouped = df.groupby(group_cols, sort=False)
    
    print(f"  Clip 总数: {len(grouped)}")
    
    for (chunk_id, clip_name), group_df in grouped:
        stats = compute_clip_stats(group_df)
        stats['chunk_id'] = chunk_id
        stats['clip_name'] = clip_name
        stats_list.append(stats)

    # 构建结果 DataFrame
    stats_df = pd.DataFrame(stats_list)
    
    # 调整列顺序
    column_order = [
        'chunk_id', 'clip_name',
        'mADE_min', 'mADE_max', 'mADE_mean', 'mADE_median', 'mADE_std',
        'frame_count', 'valid_frame_count', 'unique_cot_count',
        'cot_list'
    ]
    stats_df = stats_df[column_order]

    # Step 3: 保存结果
    print(f"[3/3] 保存统计结果 ...")
    stats_df.to_csv(output_csv, index=False)

    # 统计输出
    valid_clips = stats_df[stats_df['valid_frame_count'] > 0]
    
    print(f"\n{'=' * 60}")
    print(f"统计完成!")
    print(f"  Clip 总数: {len(stats_df)}")
    print(f"  有效 Clip 数: {len(valid_clips)}")
    print(f"  空 Clip 数: {len(stats_df) - len(valid_clips)}")
    
    if len(valid_clips) > 0:
        print(f"\n  全局 mADE 统计 (米):")
        print(f"    Mean of Means:  {valid_clips['mADE_mean'].mean():.4f}")
        print(f"    Mean of Mins:   {valid_clips['mADE_min'].mean():.4f}")
        print(f"    Mean of Maxs:   {valid_clips['mADE_max'].mean():.4f}")
        print(f"    Best Clip Min:  {valid_clips['mADE_min'].min():.4f}")
        print(f"    Worst Clip Max: {valid_clips['mADE_max'].max():.4f}")
        
        # CoT 统计
        total_unique_cots = stats_df['unique_cot_count'].sum()
        print(f"\n  CoT 统计:")
        print(f"    总唯一 CoT 数: {total_unique_cots}")
        
    print(f"\n  输出文件: {output_csv}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
