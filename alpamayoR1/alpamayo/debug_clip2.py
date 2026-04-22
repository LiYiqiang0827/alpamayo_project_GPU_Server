import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

clip = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"

# 读取索引
df_idx = pd.read_csv(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data/inference_index.csv")
print("Clip2 索引表:")
print(df_idx[['infer_idx', 'ego_idx', 't0_timestamp']].head())

# infer_idx=0 对应 ego_idx=27
ego_idx = df_idx.iloc[0]['ego_idx']
print(f"\ninfer_idx=0 -> ego_idx={ego_idx}")

# 读取原始egomotion
ego_df = pd.read_parquet(f"/data01/vla/data/data_sample_chunk0/labels/egomotion/{clip}.egomotion.parquet")

# t0时刻
t0_row = ego_df.iloc[ego_idx]
t0_xyz = t0_row[['x', 'y', 'z']].values
t0_quat = t0_row[['qx', 'qy', 'qz', 'qw']].values

print(f"\nt0时刻 (ego_idx={ego_idx}):")
print(f"  世界坐标: {t0_xyz}")
print(f"  四元数: {t0_quat}")

# 历史轨迹 (世界坐标)
hist_world = ego_df.iloc[ego_idx-16:ego_idx][['x', 'y', 'z']].values
print(f"\n历史轨迹 (世界坐标):")
print(f"  第1点: {hist_world[0]}")
print(f"  最后点 (t0-1): {hist_world[-1]}")

# 转换到局部坐标系
t0_rot = R.from_quat(t0_quat).as_matrix()
t0_rot_inv = t0_rot.T
hist_local = (hist_world - t0_xyz) @ t0_rot_inv

print(f"\n历史轨迹 (局部坐标 - 计算):")
print(f"  第1点: {hist_local[0]}")
print(f"  最后点 (应该是[0,0,0]或接近): {hist_local[-1]}")

# 对比保存的数据
saved = np.load(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data/egomotion/ego_000000_history_local.npy", allow_pickle=True).item()
print(f"\n保存的 history_local:")
print(f"  第1点: {saved['xyz'][0]}")
print(f"  最后点: {saved['xyz'][-1]}")

# 检查是否一致
print(f"\n=== 对比 ===")
print(f"计算的最后点: {hist_local[-1]}")
print(f"保存的最后点: {saved['xyz'][-1]}")
print(f"差值: {np.linalg.norm(hist_local[-1] - saved['xyz'][-1]):.6f}")
