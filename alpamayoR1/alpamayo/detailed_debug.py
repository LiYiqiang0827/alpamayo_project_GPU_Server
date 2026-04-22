#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

clip = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"
BASE_DIR = "/data01/vla/data/data_sample_chunk0"

ego_path = f"{BASE_DIR}/labels/egomotion/{clip}.egomotion.parquet"
ego_df = pd.read_parquet(ego_path)

# Frame 1903参数
t0_ts = 19349601.0
ego_idx = 1950
HISTORY_STEPS = 16
TIME_STEP = 0.1

print("=== 详细检查预处理过程 ===\n")

# 计算历史时间戳
hist_offsets = np.arange(-(HISTORY_STEPS-1) * int(TIME_STEP * 1e6), 
                          int(TIME_STEP * 1e6 / 2), 
                          int(TIME_STEP * 1e6)).astype(np.int64)
hist_timestamps = t0_ts + hist_offsets

print("历史时间戳 (前5个):")
for i, ts in enumerate(hist_timestamps[:5]):
    idx = (ego_df['timestamp'] - ts).abs().idxmin()
    actual_ts = ego_df.loc[idx, 'timestamp']
    xyz = ego_df.loc[idx, ['x', 'y', 'z']].values
    print(f"  目标{i}: {ts}, 找到索引{idx}, 实际ts={actual_ts}, 坐标=[{xyz[0]:.2f}, {xyz[1]:.2f}]")

print(f"\n手动计算 (直接取ego_idx-16到ego_idx):")
for i in range(5):
    idx = ego_idx - 16 + i
    xyz = ego_df.iloc[idx][['x', 'y', 'z']].values
    ts = ego_df.iloc[idx]['timestamp']
    print(f"  索引{idx}: ts={ts}, 坐标=[{xyz[0]:.2f}, {xyz[1]:.2f}]")

# 获取t0姿态
t0_row = ego_df.iloc[ego_idx]
t0_xyz = t0_row[['x', 'y', 'z']].values
t0_quat = t0_row[['qx', 'qy', 'qz', 'qw']].values
t0_rot = R.from_quat(t0_quat).as_matrix()
t0_rot_inv = t0_rot.T

print(f"\nt0坐标: [{t0_xyz[0]:.2f}, {t0_xyz[1]:.2f}]")

# 检查历史第一帧的转换
print(f"\n历史第一帧转换对比:")

# 方法1: 插值找到的第一帧
ts0 = hist_timestamps[0]
idx_interp = (ego_df['timestamp'] - ts0).abs().idxmin()
xyz_interp = ego_df.loc[idx_interp, ['x', 'y', 'z']].values
local_interp = (xyz_interp - t0_xyz) @ t0_rot_inv
print(f"  插值法 - 索引{idx_interp}: 世界=[{xyz_interp[0]:.2f}, {xyz_interp[1]:.2f}], 局部=[{local_interp[0]:.2f}, {local_interp[1]:.2f}]")

# 方法2: 直接取ego_idx-16
idx_direct = ego_idx - 16
xyz_direct = ego_df.iloc[idx_direct][['x', 'y', 'z']].values
local_direct = (xyz_direct - t0_xyz) @ t0_rot_inv
print(f"  直接取 - 索引{idx_direct}: 世界=[{xyz_direct[0]:.2f}, {xyz_direct[1]:.2f}], 局部=[{local_direct[0]:.2f}, {local_direct[1]:.2f}]")

# 方法3: 预处理后数据
data_dir = f"{BASE_DIR}/infer/{clip}/data"
hist_processed = np.load(f"{data_dir}/egomotion/ego_001903_history_local.npy", allow_pickle=True).item()['xyz']
print(f"  预处理: 局部=[{hist_processed[0][0]:.2f}, {hist_processed[0][1]:.2f}]")
