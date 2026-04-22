#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

clip = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"
BASE_DIR = "/data01/vla/data/data_sample_chunk0"

ego_path = f"{BASE_DIR}/labels/egomotion/{clip}.egomotion.parquet"
ego_df = pd.read_parquet(ego_path)

# Frame 1903
t0_idx = 1950
t0_row = ego_df.iloc[t0_idx]
t0_xyz = t0_row[['x', 'y', 'z']].values
t0_quat = t0_row[['qx', 'qy', 'qz', 'qw']].values
t0_rot = R.from_quat(t0_quat).as_matrix()
t0_rot_inv = t0_rot.T

print("=== 检查坐标系转换 ===\n")
print(f"t0世界坐标: [{t0_xyz[0]:.2f}, {t0_xyz[1]:.2f}]")

# 检查t0时刻的朝向
t0_yaw = np.arctan2(t0_rot[1,0], t0_rot[0,0]) * 180 / np.pi
print(f"t0偏航角: {t0_yaw:.1f}°")

# 原始未来数据（世界坐标）
future_world = ego_df.iloc[t0_idx:t0_idx+64][['x', 'y', 'z']].values

# 转换到局部坐标系
future_local = (future_world - t0_xyz) @ t0_rot_inv

print(f"\n原始未来数据（世界坐标）:")
print(f"  第一帧: [{future_world[0][0]:.2f}, {future_world[0][1]:.2f}]")
print(f"  最后一帧: [{future_world[-1][0]:.2f}, {future_world[-1][1]:.2f}]")
print(f"  总位移: {np.linalg.norm(future_world[-1][:2] - future_world[0][:2]):.2f}m")

print(f"\n转换后（局部坐标系）:")
print(f"  第一帧: [{future_local[0][0]:.2f}, {future_local[0][1]:.2f}]")
print(f"  最后一帧: [{future_local[-1][0]:.2f}, {future_local[-1][1]:.2f}]")
print(f"  总位移: {np.linalg.norm(future_local[-1][:2] - future_local[0][:2]):.2f}m")

# 对比预处理后数据
data_dir = f"{BASE_DIR}/infer/{clip}/data"
future_processed = np.load(f"{data_dir}/egomotion/ego_001903_future_gt.npy", allow_pickle=True).item()['xyz']

print(f"\n预处理后数据:")
print(f"  第一帧: [{future_processed[0][0]:.2f}, {future_processed[0][1]:.2f}]")
print(f"  最后一帧: [{future_processed[-1][0]:.2f}, {future_processed[-1][1]:.2f}]")
print(f"  总位移: {np.linalg.norm(future_processed[-1][:2] - future_processed[0][:2]):.2f}m")

print(f"\n差异分析:")
print(f"  手动转换 - 预处理: [{future_local[-1][0]-future_processed[-1][0]:.2f}, {future_local[-1][1]-future_processed[-1][1]:.2f}]")
