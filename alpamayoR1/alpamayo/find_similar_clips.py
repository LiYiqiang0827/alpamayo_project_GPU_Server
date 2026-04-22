#!/usr/bin/env python3
"""分析与第一个clip相似的clips"""
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

BASE_DIR = "/data01/vla/data/data_sample_chunk0"

def analyze_clip_features(clip_id):
    """分析clip的特征"""
    try:
        ego_path = f"{BASE_DIR}/labels/egomotion/{clip_id}.egomotion.parquet"
        if not os.path.exists(ego_path):
            return None
        
        ego_df = pd.read_parquet(ego_path)
        
        # 检查前100帧的数据
        sample = ego_df.iloc[:100]
        
        # 位置变化
        x_range = sample['x'].max() - sample['x'].min()
        y_range = sample['y'].max() - sample['y'].min()
        
        # 速度（通过位置差分估算）
        dx = sample['x'].diff().dropna()
        dy = sample['y'].diff().dropna()
        dt = sample['timestamp'].diff().dropna() / 1e6
        speed = np.sqrt(dx**2 + dy**2) / dt
        avg_speed = speed.mean()
        
        # 偏航角变化
        yaws = []
        for i in range(min(100, len(ego_df))):
            quat = ego_df.iloc[i][['qx', 'qy', 'qz', 'qw']].values
            rot = R.from_quat(quat).as_matrix()
            yaw = np.arctan2(rot[1,0], rot[0,0]) * 180 / np.pi
            yaws.append(yaw)
        yaws = np.array(yaws)
        yaw_range = yaws.max() - yaws.min()
        
        return {
            'clip_id': clip_id,
            'yaw_range': yaw_range,
            'x_range': x_range,
            'y_range': y_range,
            'avg_speed': avg_speed,
            'total_frames': len(ego_df)
        }
    except Exception as e:
        return None

# 第一个clip的特征（参考）
reference = {
    'clip_id': '01d3588e-bca7-4a18-8e74-c6cfe9e996db',
    'yaw_range': 0.1,  # 偏航角变化
    'x_range': 6.9,    # X范围
    'y_range': 0.0,    # Y范围
    'avg_speed': 7.03  # 平均速度
}

print("=== 参考clip特征（ADE=0.8m）===")
print(f"偏航角变化: {reference['yaw_range']:.1f}°")
print(f"前100帧X范围: {reference['x_range']:.1f}m")
print(f"前100帧Y范围: {reference['y_range']:.1f}m")
print(f"平均速度: {reference['avg_speed']:.2f}m/s")
print()

# 获取所有clip
clips = [f.replace('.egomotion.parquet', '') for f in os.listdir(f"{BASE_DIR}/labels/egomotion/") if f.endswith('.parquet')]

print("=== 分析与参考clip相似的clips ===\n")

results = []
for clip in clips:
    result = analyze_clip_features(clip)
    if result:
        # 计算与参考的相似度（欧氏距离）
        dist = np.sqrt(
            ((result['yaw_range'] - reference['yaw_range']) / 10) ** 2 +
            ((result['x_range'] - reference['x_range']) / 10) ** 2 +
            ((result['y_range'] - reference['y_range']) / 5) ** 2 +
            ((result['avg_speed'] - reference['avg_speed']) / 5) ** 2
        )
        result['similarity'] = dist
        results.append(result)

# 按相似度排序
results.sort(key=lambda x: x['similarity'])

print("最相似的clips（特征接近参考clip）：")
print(f"{'Clip ID':<40} {'偏航角':<8} {'X范围':<8} {'Y范围':<8} {'速度':<8} {'相似度':<8}")
print("-" * 90)
for r in results[:10]:
    print(f"{r['clip_id']:<40} {r['yaw_range']:>6.1f}° {r['x_range']:>7.1f}m {r['y_range']:>7.1f}m {r['avg_speed']:>7.2f}m/s {r['similarity']:>7.2f}")
