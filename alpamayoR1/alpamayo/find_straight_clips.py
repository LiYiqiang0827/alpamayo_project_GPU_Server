#!/usr/bin/env python3
"""查找直道clip：检查egomotion的转弯角度变化"""
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

BASE_DIR = "/data01/vla/data/data_sample_chunk0"

def calculate_curvature(clip_id):
    """计算clip的曲率变化，返回偏航角变化范围"""
    try:
        ego_path = f"{BASE_DIR}/labels/egomotion/{clip_id}.egomotion.parquet"
        if not os.path.exists(ego_path):
            return None
        
        ego_df = pd.read_parquet(ego_path)
        
        # 计算每帧的偏航角
        yaws = []
        for i in range(len(ego_df)):
            quat = ego_df.iloc[i][['qx', 'qy', 'qz', 'qw']].values
            rot = R.from_quat(quat).as_matrix()
            yaw = np.arctan2(rot[1,0], rot[0,0]) * 180 / np.pi
            yaws.append(yaw)
        
        yaws = np.array(yaws)
        
        # 计算偏航角变化范围
        yaw_range = np.max(yaws) - np.min(yaws)
        
        # 计算位置变化范围（判断是否直行）
        x_range = ego_df['x'].max() - ego_df['x'].min()
        y_range = ego_df['y'].max() - ego_df['y'].min()
        
        return {
            'clip_id': clip_id,
            'yaw_range': yaw_range,
            'x_range': x_range,
            'y_range': y_range,
            'total_frames': len(ego_df)
        }
    except Exception as e:
        return None

# 获取所有clip
clips = [f.replace('.egomotion.parquet', '') for f in os.listdir(f"{BASE_DIR}/labels/egomotion/") if f.endswith('.parquet')]

print("=== 分析所有clips的转弯程度 ===\n")

results = []
for clip in clips[:20]:  # 先分析前20个
    result = calculate_curvature(clip)
    if result:
        results.append(result)

# 按偏航角变化排序（小的排前面，即直道）
results.sort(key=lambda x: x['yaw_range'])

print("偏航角变化最小的clips（可能是直道）：")
print(f"{'Clip ID':<40} {'偏航角范围':<12} {'X范围':<10} {'Y范围':<10}")
print("-" * 80)
for r in results[:10]:
    print(f"{r['clip_id']:<40} {r['yaw_range']:>8.1f}°   {r['x_range']:>8.1f}m  {r['y_range']:>8.1f}m")

print("\n偏航角变化最大的clips（转弯多）：")
for r in results[-5:]:
    print(f"{r['clip_id']:<40} {r['yaw_range']:>8.1f}°   {r['x_range']:>8.1f}m  {r['y_range']:>8.1f}m")
