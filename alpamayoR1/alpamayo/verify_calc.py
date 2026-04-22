import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
ego_df = pd.read_parquet(f"/data01/vla/data/data_sample_chunk0/labels/egomotion/{clip}.egomotion.parquet")

# infer_idx=0 对应 ego_idx=?
df_idx = pd.read_csv(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data/inference_index.csv")
ego_idx = df_idx.iloc[0]['ego_idx']
print(f"infer_idx=0 -> ego_idx={ego_idx}")

# t0时刻
t0_row = ego_df.iloc[ego_idx]
t0_xyz = t0_row[['x', 'y', 'z']].values
t0_quat = t0_row[['qx', 'qy', 'qz', 'qw']].values

print(f"\nt0时刻 (ego_idx={ego_idx}):")
print(f"  世界坐标: {t0_xyz}")

# 历史轨迹世界坐标
hist_world = ego_df.iloc[ego_idx-16:ego_idx][['x', 'y', 'z']].values
print(f"\n历史轨迹世界坐标:")
print(f"  第1点 (最早): {hist_world[0]}")
print(f"  最后点 (t0-1): {hist_world[-1]}")

# 转换
t0_rot = R.from_quat(t0_quat).as_matrix()
t0_rot_inv = t0_rot.T
hist_local = (hist_world - t0_xyz) @ t0_rot_inv

print(f"\n历史轨迹局部坐标 (计算):")
print(f"  第1点: {hist_local[0]}")
print(f"  最后点: {hist_local[-1]}")

# 关键验证: hist_world[-1] - t0_xyz 应该是什么?
print(f"\n关键验证:")
print(f"  hist_world[-1] - t0_xyz = {hist_world[-1] - t0_xyz}")
print(f"  这在世界坐标系中表示 t0-1 到 t0 的向量")

# 这个向量转换到局部坐标系
vec_world = hist_world[-1] - t0_xyz
vec_local = vec_world @ t0_rot_inv
print(f"  该向量在局部坐标系中: {vec_local}")
print(f"  理论上应该是 [0, -0.1, 0] 左右 (如果车在向前开)")

# 检查车的朝向
print(f"\n车的朝向 (旋转矩阵):")
print(t0_rot)
print(f"第一列 (局部X在世界中的方向): {t0_rot[:, 0]}")
print(f"第二列 (局部Y在世界中的方向): {t0_rot[:, 1]}")
