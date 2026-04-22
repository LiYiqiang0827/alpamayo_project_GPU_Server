import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
ego_df = pd.read_parquet(f"/data01/vla/data/data_sample_chunk0/labels/egomotion/{clip}.egomotion.parquet")

# 当前v3代码: 历史是 ego_idx-16 : ego_idx (不包含ego_idx)
# 试试包含ego_idx: ego_idx-15 : ego_idx+1

ego_idx = 25  # infer_idx=0对应的ego_idx
t0_row = ego_df.iloc[ego_idx]
t0_xyz = t0_row[['x', 'y', 'z']].values
t0_quat = t0_row[['qx', 'qy', 'qz', 'qw']].values
t0_rot = R.from_quat(t0_quat).as_matrix()
t0_rot_inv = t0_rot.T

print("=== 测试不同历史轨迹定义 ===")

# 方法1: 当前v3 (t0-1.6s 到 t0-0.1s, 不包含t0)
hist1 = ego_df.iloc[ego_idx-16:ego_idx]
hist1_local = (hist1[['x', 'y', 'z']].values - t0_xyz) @ t0_rot_inv
print(f"\n方法1 (v3当前): t0-1.6s 到 t0-0.1s")
print(f"  最后点: {hist1_local[-1]}")
print(f"  步数: {len(hist1_local)}")

# 方法2: 包含t0 (t0-1.5s 到 t0)
hist2 = ego_df.iloc[ego_idx-15:ego_idx+1]
hist2_local = (hist2[['x', 'y', 'z']].values - t0_xyz) @ t0_rot_inv
print(f"\n方法2: t0-1.5s 到 t0 (包含t0)")
print(f"  最后点: {hist2_local[-1]}")
print(f"  步数: {len(hist2_local)}")

# 方法3: 严格10Hz插值 (t0-1.6, t0-1.5, ..., t0-0.1)
t0_ts = ego_df.iloc[ego_idx]['timestamp']
hist_timestamps = [t0_ts - (16 - i) * 100000 for i in range(16)]  # 100ms = 100000us
hist3_local = []
for ts in hist_timestamps:
    idx = (ego_df['timestamp'] - ts).abs().idxmin()
    row = ego_df.loc[idx]
    xyz = row[['x', 'y', 'z']].values
    hist3_local.append((xyz - t0_xyz) @ t0_rot_inv)
hist3_local = np.array(hist3_local)
print(f"\n方法3: 严格10Hz (t0-1.6s 到 t0-0.1s)")
print(f"  最后点: {hist3_local[-1]}")
print(f"  步数: {len(hist3_local)}")

# 方法4: 官方代码方式 (t0-1.5s 到 t0)
# history_offsets_us = np.arange(-(16-1)*0.1*1e6, 0.1*1e6/2, 0.1*1e6)
# = [-1500000, -1400000, ..., -100000, 0]
hist_offsets = np.arange(-15 * 100000, 50000, 100000).astype(np.int64)
hist_ts_official = t0_ts + hist_offsets
hist4_local = []
for ts in hist_ts_official:
    idx = (ego_df['timestamp'] - ts).abs().idxmin()
    row = ego_df.loc[idx]
    xyz = row[['x', 'y', 'z']].values
    hist4_local.append((xyz - t0_xyz) @ t0_rot_inv)
hist4_local = np.array(hist4_local)
print(f"\n方法4 (官方): t0-1.5s 到 t0")
print(f"  最后点: {hist4_local[-1]}")
print(f"  步数: {len(hist4_local)}")
