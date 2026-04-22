#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

clip = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"
BASE_DIR = "/data01/vla/data/data_sample_chunk0"

ego_path = f"{BASE_DIR}/labels/egomotion/{clip}.egomotion.parquet"
ego_df = pd.read_parquet(ego_path)

# 从索引文件
data_dir = f"{BASE_DIR}/infer/{clip}/data"
index_df = pd.read_csv(f"{data_dir}/inference_index.csv")
row = index_df[index_df['infer_idx'] == 1903].iloc[0]

t0_ts = row['t0_timestamp']
ego_idx = row['ego_idx']

print("=== 检查预处理后t0位置 ===\n")
print(f"t0_timestamp: {t0_ts}")
print(f"ego_idx: {ego_idx}")

# 读取预处理后数据
prefix = "ego_001903"
hist_local = np.load(f"{data_dir}/egomotion/{prefix}_history_local.npy", allow_pickle=True).item()
future_gt = np.load(f"{data_dir}/egomotion/{prefix}_future_gt.npy", allow_pickle=True).item()

hist_xyz = hist_local['xyz']
future_xyz = future_gt['xyz']

print(f"\n预处理后历史数据:")
print(f"  第一帧: [{hist_xyz[0][0]:.2f}, {hist_xyz[0][1]:.2f}, {hist_xyz[0][2]:.2f}]")
print(f"  最后一帧(t0): [{hist_xyz[-1][0]:.2f}, {hist_xyz[-1][1]:.2f}, {hist_xyz[-1][2]:.2f}]")

print(f"\n预处理后未来数据:")
print(f"  第一帧: [{future_xyz[0][0]:.2f}, {future_xyz[0][1]:.2f}, {future_xyz[0][2]:.2f}]")
print(f"  最后一帧: [{future_xyz[-1][0]:.2f}, {future_xyz[-1][1]:.2f}, {future_xyz[-1][2]:.2f}]")

print(f"\n问题：t0应该在局部坐标系原点 [0,0,0]，但实际是 [{hist_xyz[-1][0]:.2f}, {hist_xyz[-1][1]:.2f}, {hist_xyz[-1][2]:.2f}]")

# 检查原始egomotion在ego_idx处的数据
t0_row = ego_df.iloc[ego_idx]
t0_xyz = t0_row[['x', 'y', 'z']].values
t0_quat = t0_row[['qx', 'qy', 'qz', 'qw']].values

print(f"\n原始egomotion在ego_idx={ego_idx}:")
print(f"  xyz: [{t0_xyz[0]:.2f}, {t0_xyz[1]:.2f}, {t0_xyz[2]:.2f}]")
print(f"  timestamp: {t0_row['timestamp']}")

# 手动计算历史第一帧
hist_start_idx = ego_idx - 16
hist_world_0 = ego_df.iloc[hist_start_idx][['x', 'y', 'z']].values
t0_rot = R.from_quat(t0_quat).as_matrix()
t0_rot_inv = t0_rot.T
hist_local_0 = (hist_world_0 - t0_xyz) @ t0_rot_inv

print(f"\n手动计算历史第一帧 (索引{hist_start_idx}):")
print(f"  世界坐标: [{hist_world_0[0]:.2f}, {hist_world_0[1]:.2f}, {hist_world_0[2]:.2f}]")
print(f"  局部坐标: [{hist_local_0[0]:.2f}, {hist_local_0[1]:.2f}, {hist_local_0[2]:.2f}]")
print(f"  预处理后: [{hist_xyz[0][0]:.2f}, {hist_xyz[0][1]:.2f}, {hist_xyz[0][2]:.2f}]")
