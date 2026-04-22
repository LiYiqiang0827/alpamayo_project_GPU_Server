#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

# 检查转弯clip的原始egomotion数据
clip = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"
BASE_DIR = "/data01/vla/data/data_sample_chunk0"

# 读取原始egomotion
ego_path = f"{BASE_DIR}/labels/egomotion/{clip}.egomotion.parquet"
ego_df = pd.read_parquet(ego_path)

# 读取索引找到t0
data_dir = f"{BASE_DIR}/infer/{clip}/data"
index_df = pd.read_csv(f"{data_dir}/inference_index.csv")
t0_ts = index_df[index_df['infer_idx'] == 1903]['t0_timestamp'].values[0]

# 找到t0对应的ego帧
t0_idx = (ego_df['timestamp'] - t0_ts).abs().idxmin()

print("=== 转弯场景原始数据检查 (Frame 1903) ===\n")
print(f"t0_timestamp: {t0_ts}")
print(f"t0_idx in ego_df: {t0_idx}")

# 获取t0姿态
t0_row = ego_df.iloc[t0_idx]
t0_xyz = t0_row[['x', 'y', 'z']].values
t0_quat = t0_row[['qx', 'qy', 'qz', 'qw']].values
t0_rot = R.from_quat(t0_quat).as_matrix()

print(f"\nt0世界坐标: [{t0_xyz[0]:.2f}, {t0_xyz[1]:.2f}, {t0_xyz[2]:.2f}]")

# 历史16帧（原始数据）
hist_start_idx = t0_idx - 16
hist_world = ego_df.iloc[hist_start_idx:t0_idx][['x', 'y', 'z']].values

print(f"\n原始历史数据 (世界坐标):")
print(f"  第一帧: [{hist_world[0][0]:.2f}, {hist_world[0][1]:.2f}]")
print(f"  最后一帧(t0): [{hist_world[-1][0]:.2f}, {hist_world[-1][1]:.2f}]")

# 未来64帧（原始数据）
future_end_idx = t0_idx + 64
future_world = ego_df.iloc[t0_idx:future_end_idx][['x', 'y', 'z']].values

print(f"\n原始未来数据 (世界坐标):")
print(f"  第一帧(t0): [{future_world[0][0]:.2f}, {future_world[0][1]:.2f}]")
print(f"  最后一帧: [{future_world[-1][0]:.2f}, {future_world[-1][1]:.2f}]")

# 手动转换到局部坐标系
t0_rot_inv = t0_rot.T
hist_local_manual = (hist_world - t0_xyz) @ t0_rot_inv
future_local_manual = (future_world - t0_xyz) @ t0_rot_inv

print(f"\n转换后局部坐标:")
print(f"  历史第一帧: [{hist_local_manual[0][0]:.2f}, {hist_local_manual[0][1]:.2f}]")
print(f"  历史最后一帧: [{hist_local_manual[-1][0]:.2f}, {hist_local_manual[-1][1]:.2f}]")
print(f"  未来第一帧: [{future_local_manual[0][0]:.2f}, {future_local_manual[0][1]:.2f}]")
print(f"  未来最后一帧: [{future_local_manual[-1][0]:.2f}, {future_local_manual[-1][1]:.2f}]")

# 对比预处理后数据
prefix = "ego_001903"
hist_processed = np.load(f"{data_dir}/egomotion/{prefix}_history_local.npy", allow_pickle=True).item()['xyz']
future_processed = np.load(f"{data_dir}/egomotion/{prefix}_future_gt.npy", allow_pickle=True).item()['xyz']

print(f"\n预处理后数据对比:")
print(f"  历史最后一帧差异: {np.linalg.norm(hist_processed[-1] - hist_local_manual[-1]):.4f}m")
print(f"  未来最后一帧差异: {np.linalg.norm(future_processed[-1] - future_local_manual[-1]):.4f}m")
