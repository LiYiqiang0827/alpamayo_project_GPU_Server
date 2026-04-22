#!/usr/bin/env python3
"""
分析 chunk0 所有 clips，找出转弯场景
- 通过 egomotion 的曲率 (curvature) 来识别转弯
- 曲率绝对值越大，转弯越急
"""
import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = '/data01/vla/data/data_sample_chunk0'

def analyze_clip_turns(clip_id):
    """分析单个 clip 的转弯情况"""
    ego_file = f'{DATA_DIR}/labels/egomotion/{clip_id}.egomotion.parquet'
    if not os.path.exists(ego_file):
        return None
    
    df = pd.read_parquet(ego_file)
    
    # 曲率统计
    curvature = df['curvature'].abs()
    max_curvature = curvature.max()
    mean_curvature = curvature.mean()
    
    # 转弯帧数 (曲率 > 0.01 约等于半径100m)
    turn_frames = (curvature > 0.01).sum()
    sharp_turn_frames = (curvature > 0.05).sum()  # 急转弯
    
    return {
        'clip_id': clip_id,
        'total_frames': len(df),
        'max_curvature': max_curvature,
        'mean_curvature': mean_curvature,
        'turn_frames': turn_frames,
        'sharp_turn_frames': sharp_turn_frames,
        'turn_ratio': turn_frames / len(df) * 100
    }

def main():
    print('=== 分析 chunk0 转弯场景 ===\n')
    
    # 获取所有 clip IDs
    ego_files = glob.glob(f'{DATA_DIR}/labels/egomotion/*.egomotion.parquet')
    clip_ids = [os.path.basename(f).replace('.egomotion.parquet', '') for f in ego_files]
    
    print(f'找到 {len(clip_ids)} 个 clips')
    
    # 分析每个 clip
    results = []
    for clip_id in sorted(clip_ids)[:50]:  # 先分析前50个
        result = analyze_clip_turns(clip_id)
        if result:
            results.append(result)
    
    # 排序：急转弯帧数最多的排前面
    df = pd.DataFrame(results)
    df_sorted = df.sort_values('sharp_turn_frames', ascending=False)
    
    print('\n=== 转弯场景排名 (按急转弯帧数) ===')
    print(f"{'Clip ID':<40} {'急转弯帧':<10} {'转弯帧':<10} {'最大曲率':<12} {'转弯占比':<10}")
    print('-' * 90)
    for _, row in df_sorted.head(10).iterrows():
        print(f"{row['clip_id']:<40} {row['sharp_turn_frames']:<10} {row['turn_frames']:<10} {row['max_curvature']:<12.4f} {row['turn_ratio']:<10.1f}%")
    
    # 推荐 clip
    best_turn_clip = df_sorted.iloc[0]['clip_id']
    print(f'\n=== 推荐转弯场景 ===')
    print(f'Clip ID: {best_turn_clip}')
    print(f'急转弯帧数: {df_sorted.iloc[0]["sharp_turn_frames"]} / {df_sorted.iloc[0]["total_frames"]}')
    print(f'最大曲率: {df_sorted.iloc[0]["max_curvature"]:.4f}')

if __name__ == '__main__':
    main()
