#!/usr/bin/env python3
"""对比两个clip的egomotion数据"""
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

BASE_DIR = "/data01/vla/data/data_sample_chunk0"

clips = [
    ("01d3588e-bca7-4a18-8e74-c6cfe9e996db", "好的clip (ADE=0.8m)"),
    ("e9162b7f-00f1-4563-890e-15c9fc89ad7a", "直道clip (ADE=22m)")
]

print("=== 对比两个clip的egomotion数据 ===\n")

for clip_id, desc in clips:
    ego_path = f"{BASE_DIR}/labels/egomotion/{clip_id}.egomotion.parquet"
    ego_df = pd.read_parquet(ego_path)
    
    print(f"{desc}")
    print(f"  总帧数: {len(ego_df)}")
    
    # 检查前100帧的数据
    sample = ego_df.iloc[:100]
    
    # 位置变化
    x_range = sample['x'].max() - sample['x'].min()
    y_range = sample['y'].max() - sample['y'].min()
    print(f"  前100帧X范围: {x_range:.1f}m")
    print(f"  前100帧Y范围: {y_range:.1f}m")
    
    # 速度（通过位置差分估算）
    dx = sample['x'].diff().dropna()
    dy = sample['y'].diff().dropna()
    speed = np.sqrt(dx**2 + dy**2) / (sample['timestamp'].diff().dropna() / 1e6)
    print(f"  前100帧平均速度: {speed.mean():.2f}m/s")
    
    # 偏航角
    yaws = []
    for i in range(min(100, len(ego_df))):
        quat = ego_df.iloc[i][['qx', 'qy', 'qz', 'qw']].values
        rot = R.from_quat(quat).as_matrix()
        yaw = np.arctan2(rot[1,0], rot[0,0]) * 180 / np.pi
        yaws.append(yaw)
    yaws = np.array(yaws)
    print(f"  前100帧偏航角范围: {yaws.max() - yaws.min():.1f}°")
    print()
