import numpy as np
import pandas as pd

# 读取egomotion数据
prefix = "ego_000000"
base = "/data01/vla/data/data_sample_chunk0/infer/46003675-4b4e-4c0f-ae54-3f7622bddf6a/data/egomotion"

hist_world = np.load(f"{base}/{prefix}_history_world.npy", allow_pickle=True).item()
hist_local = np.load(f"{base}/{prefix}_history_local.npy", allow_pickle=True).item()
future_gt = np.load(f"{base}/{prefix}_future_gt.npy", allow_pickle=True).item()

print("=== 历史轨迹（世界坐标）===")
print(f"形状: {hist_world['xyz'].shape}")
print(f"前3点: {hist_world['xyz'][:3]}")
print(f"最后3点: {hist_world['xyz'][-3:]}")

print("\n=== 历史轨迹（局部坐标）===")
print(f"形状: {hist_local['xyz'].shape}")
print(f"前3点: {hist_local['xyz'][:3]}")
print(f"最后3点: {hist_local['xyz'][-3:]}")

print("\n=== 未来真值（局部坐标 - 修复后）===")
print(f"形状: {future_gt['xyz'].shape}")
print(f"前5点: {future_gt['xyz'][:5]}")
print(f"最后5点: {future_gt['xyz'][-5:]}")
print(f"Y方向位移（向前）: {future_gt['xyz'][-1, 1] - future_gt['xyz'][0, 1]:.2f}m")

print("\n=== 检查索引表帧顺序 ===")
df = pd.read_csv("/data01/vla/data/data_sample_chunk0/infer/46003675-4b4e-4c0f-ae54-3f7622bddf6a/data/inference_index.csv")
print(f"总行数: {len(df)}")
print("\n第一行的帧路径（检查f0-f3顺序）:")
row = df.iloc[0]
print(f"cam_left_f0 (t-0.3): {row['cam_left_f0']}")
print(f"cam_left_f1 (t-0.2): {row['cam_left_f1']}")
print(f"cam_left_f2 (t-0.1): {row['cam_left_f2']}")
print(f"cam_left_f3 (t):     {row['cam_left_f3']}")
