import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

# 读取转弯场景的原始egomotion
ego_df = pd.read_parquet("/data01/vla/data/data_sample_chunk0/labels/egomotion/46003675-4b4e-4c0f-ae54-3f7622bddf6a.egomotion.parquet")

print("=== 原始 Egomotion 数据 ===")
print(f"总帧数: {len(ego_df)}")
print(f"\n前5帧 (世界坐标):")
print(ego_df[["timestamp", "x", "y", "z"]].head())

print(f"\n第16帧 (t0, 作为例子):")
t0_idx = 16
t0_row = ego_df.iloc[t0_idx]
print(f"t0位置: x={t0_row['x']:.3f}, y={t0_row['y']:.3f}, z={t0_row['z']:.3f}")

# 未来64帧世界坐标
future_world = ego_df.iloc[t0_idx:t0_idx+64][["x", "y", "z"]].values
print(f"\n未来64帧世界坐标:")
print(f"  起点: {future_world[0]}")
print(f"  终点: {future_world[-1]}")
print(f"  X位移: {future_world[-1, 0] - future_world[0, 0]:.3f}m")
print(f"  Y位移: {future_world[-1, 1] - future_world[0, 1]:.3f}m")

# 转换到局部坐标系
t0_xyz = t0_row[["x", "y", "z"]].values
t0_quat = t0_row[["qx", "qy", "qz", "qw"]].values
t0_rot = R.from_quat(t0_quat).as_matrix()
t0_rot_inv = t0_rot.T

future_local = (future_world - t0_xyz) @ t0_rot_inv
print(f"\n转换到局部坐标系后:")
print(f"  起点: {future_local[0]}")
print(f"  终点: {future_local[-1]}")
print(f"  X位移: {future_local[-1, 0] - future_local[0, 0]:.3f}m")
print(f"  Y位移: {future_local[-1, 1] - future_local[0, 1]:.3f}m")

# 检查我们保存的npy文件
print(f"\n=== 检查保存的npy文件 ===")
saved = np.load(f"/data01/vla/data/data_sample_chunk0/infer/46003675-4b4e-4c0f-ae54-3f7622bddf6a/data/egomotion/ego_000000_future_gt.npy", allow_pickle=True).item()
print(f"保存的真值起点: {saved['xyz'][0]}")
print(f"保存的真值终点: {saved['xyz'][-1]}")
