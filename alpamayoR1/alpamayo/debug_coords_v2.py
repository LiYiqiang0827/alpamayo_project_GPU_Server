import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# 读取原始egomotion数据
ego_df = pd.read_parquet("/data01/vla/data/data_sample_chunk0/labels/egomotion/46003675-4b4e-4c0f-ae54-3f7622bddf6a.egomotion.parquet")

# infer_idx=0 对应 ego_idx=27
ego_idx = 27
print(f"=== ego_idx={ego_idx} ===")

# t0时刻
t0_row = ego_df.iloc[ego_idx]
t0_xyz = t0_row[['x', 'y', 'z']].values
t0_quat = t0_row[['qx', 'qy', 'qz', 'qw']].values

print(f"t0世界坐标: {t0_xyz}")
print(f"t0四元数: {t0_quat}")

# 计算旋转矩阵
t0_rot = R.from_quat(t0_quat).as_matrix()
print(f"\nt0旋转矩阵 (世界->局部):")
print(t0_rot)

# 历史16帧世界坐标 (t0之前的16帧)
hist_world = ego_df.iloc[ego_idx-16:ego_idx][['x', 'y', 'z']].values
print(f"\n历史轨迹 (世界坐标):")
print(f"  第1点 (最早): {hist_world[0]}")
print(f"  最后点 (t0-1): {hist_world[-1]}")

# 转换到局部坐标系
t0_rot_inv = t0_rot.T
hist_local_calc = (hist_world - t0_xyz) @ t0_rot_inv
print(f"\n历史轨迹 (计算出的局部坐标):")
print(f"  第1点: {hist_local_calc[0]}")
print(f"  最后点: {hist_local_calc[-1]}")

# 对比保存的数据
saved_hist = np.load(f"/data01/vla/data/data_sample_chunk0/infer/46003675-4b4e-4c0f-ae54-3f7622bddf6a/data/egomotion/ego_000000_history_local.npy", allow_pickle=True).item()
print(f"\n保存的历史轨迹 (local):")
print(f"  第1点: {saved_hist['xyz'][0]}")
print(f"  最后点: {saved_hist['xyz'][-1]}")

# 检查是否一致
print(f"\n=== 对比 ===")
print(f"计算的X[0]: {hist_local_calc[0,0]:.6f}, 保存的X[0]: {saved_hist['xyz'][0,0]:.6f}")
print(f"计算的Y[0]: {hist_local_calc[0,1]:.6f}, 保存的Y[0]: {saved_hist['xyz'][0,1]:.6f}")

# 关键检查：历史轨迹相对于t0的方向
print(f"\n=== 方向分析 ===")
print(f"在局部坐标系中:")
print(f"  历史第1点相对t0: X={hist_local_calc[0,0]:.3f}, Y={hist_local_calc[0,1]:.3f}")
print(f"  这意味着历史轨迹在车 {'前方' if hist_local_calc[0,1] > 0 else '后方'} {'右侧' if hist_local_calc[0,0] > 0 else '左侧'}")
