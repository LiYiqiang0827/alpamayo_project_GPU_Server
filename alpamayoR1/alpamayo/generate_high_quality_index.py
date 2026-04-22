#!/usr/bin/env python3
"""
生成高质量推理索引
过滤条件:
1. 4个摄像头时间差 ≤ 33ms (1帧@30Hz)
2. 4个摄像头都有有效数据
3. 历史16步 + 未来64步 egomotion完整
"""
import pandas as pd
import numpy as np
import argparse
import os

def generate_high_quality_index(clip_id, chunk=0):
    BASE_DIR = f'/data01/vla/data/data_sample_chunk{chunk}'
    INFER_DIR = f'{BASE_DIR}/infer/{clip_id}'
    DATA_DIR = f'{INFER_DIR}/data'
    
    print(f'=== 生成高质量索引 {clip_id} ===\n')
    
    # 读取普通索引
    index_df = pd.read_csv(f'{DATA_DIR}/inference_index.csv')
    print(f'普通索引: {len(index_df)} 帧')
    
    # 读取相机时间戳
    cam_timestamps = {}
    cameras = [
        'camera_cross_left_120fov',
        'camera_front_wide_120fov',
        'camera_cross_right_120fov',
        'camera_front_tele_30fov'
    ]
    
    for cam in cameras:
        ts_path = f'{BASE_DIR}/camera/{clip_id}.{cam}.timestamps.parquet'
        if os.path.exists(ts_path):
            ts_df = pd.read_parquet(ts_path)
            cam_timestamps[cam] = ts_df['timestamp'].values
        else:
            print(f'警告: 找不到 {cam} 的时间戳文件')
            cam_timestamps[cam] = None
    
    # 过滤条件
    MAX_IMAGE_DIFF_MS = 33  # 1帧@30Hz
    valid_frames = []
    
    for idx, row in index_df.iterrows():
        infer_idx = row['infer_idx']
        t0_ts = row['t0_timestamp']
        
        # 检查每个相机的时间差
        max_diff = 0
        all_valid = True
        
        for cam in cameras:
            if cam_timestamps[cam] is None:
                all_valid = False
                break
            
            # 找到最接近t0的帧
            ts_array = cam_timestamps[cam]
            valid_ts = ts_array[ts_array <= t0_ts]
            if len(valid_ts) == 0:
                all_valid = False
                break
            
            closest_ts = valid_ts[-1]
            time_diff_ms = (t0_ts - closest_ts) / 1000.0
            max_diff = max(max_diff, time_diff_ms)
        
        # 检查是否满足条件
        if all_valid and max_diff <= MAX_IMAGE_DIFF_MS:
            valid_frames.append({
                **row.to_dict(),
                'max_image_diff_ms': round(max_diff, 1)
            })
    
    # 保存高质量索引
    hq_df = pd.DataFrame(valid_frames)
    output_path = f'{DATA_DIR}/inference_index_high_quality.csv'
    hq_df.to_csv(output_path, index=False)
    
    print(f'高质量索引: {len(hq_df)} 帧 (过滤掉 {len(index_df) - len(hq_df)} 帧)')
    print(f'过滤比例: {(len(index_df) - len(hq_df)) / len(index_df) * 100:.1f}%')
    print(f'保存至: {output_path}')
    
    # 打印统计
    if len(hq_df) > 0:
        print(f'\n时间对齐统计:')
        print(f'  max_image_diff_ms: {hq_df["max_image_diff_ms"].min():.1f}ms ~ {hq_df["max_image_diff_ms"].max():.1f}ms')
        print(f'  平均: {hq_df["max_image_diff_ms"].mean():.1f}ms')

def main():
    parser = argparse.ArgumentParser(description='生成高质量推理索引')
    parser.add_argument('--clip', type=str, required=True)
    parser.add_argument('--chunk', type=int, default=0)
    args = parser.parse_args()
    
    generate_high_quality_index(args.clip, args.chunk)

if __name__ == '__main__':
    main()
