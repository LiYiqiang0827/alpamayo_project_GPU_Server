import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

# 读取原始egomotion
ego_df = pd.read_parquet("/data01/vla/data/data_sample_chunk0/labels/egomotion/46003675-4b4e-4c0f-ae54-3f7622bddf6a.egomotion.parquet")

# infer_idx=0 对应 ego_idx=27
ego_idx = 27

print(f"=== ego_idx={ego_idx} (infer_idx=0) ===")

# t0时刻
t0_row = ego_df.iloc[ego_idx]
t0_xyz = t0_row[["x", "y", "z"]].values
t0_quat = t0_row[["qx", "qy", "qz", "qw"]].values

print(f"t0世界坐标: {t0_xyz}")
print(f"t0四元数: {t0_quat}")

# 计算旋转矩阵
t0_rot = R.from_quat(t0_quat).as_matrix()
print(f"t0旋转矩阵:\n{t0_rot}")

# 未来64帧世界坐标
future_world = ego_df.iloc[ego_idx:ego_idx+64][["x", "y", "z"]].values
print(f"\n未来64帧世界坐标:")
print(f"  起点: {future_world[0]}")
print(f"  终点: {future_world[-1]}")

# 转换到局部坐标系
t0_rot_inv = t0_rot.T
future_local = (future_world - t0_xyz) @ t0_rot_inv
print(f"\n转换到局部坐标系:")
print(f"  起点: {future_local[0]}")
print(f"  终点: {future_local[-1]}")

# 检查保存的npy文件
saved = np.load("/data01/vla/data/data_sample_chunk0/infer/46003675-4b4e-4c0f-ae54-3f7622bddf6a/data/egomotion/ego_000000_future_gt.npy", allow_pickle=True).item()
print(f"\n保存的npy文件 (ego_000000):")
print(f"  起点: {saved['xyz'][0]}")
print(f"  终点: {saved['xyz'][-1]}")

print(f"\n应该一致吗？局部坐标起点应该是[0,0,0]吗？")
print(f"future_local[0] = {future_local[0]}")
print(f"saved['xyz'][0] = {saved['xyz'][0]}")
