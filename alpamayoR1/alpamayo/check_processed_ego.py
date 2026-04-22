#!/usr/bin/env python3
import pandas as pd
import numpy as np

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
DATA_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data"

# 读取第一个帧的egomotion
prefix = "ego_000138"
hist_local = np.load(f"{DATA_DIR}/egomotion/{prefix}_history_local.npy", allow_pickle=True).item()
future_gt = np.load(f"{DATA_DIR}/egomotion/{prefix}_future_gt.npy", allow_pickle=True).item()

# 读取索引
index_df = pd.read_csv(f"{DATA_DIR}/inference_index.csv")
row = index_df[index_df['infer_idx'] == 138].iloc[0]
t0_ts = row['t0_timestamp']

print("=== 预处理后数据检查 (Frame 138) ===")
print(f"t0_timestamp: {t0_ts}")
print()

# 检查历史数据
hist_xyz = hist_local['xyz']
print(f"历史数据:")
print(f"  shape: {hist_xyz.shape} (应为16帧 = 1.6秒)")
print(f"  第一帧: [{hist_xyz[0][0]:.2f}, {hist_xyz[0][1]:.2f}, {hist_xyz[0][2]:.2f}]")
print(f"  最后一帧(t0): [{hist_xyz[-1][0]:.2f}, {hist_xyz[-1][1]:.2f}, {hist_xyz[-1][2]:.2f}]")

# 检查相邻帧的距离（验证时间间隔）
print(f"\n历史数据相邻帧位移:")
for i in range(1, min(5, len(hist_xyz))):
    dist = np.linalg.norm(hist_xyz[i][:2] - hist_xyz[i-1][:2])
    print(f"  {i-1}->{i}: {dist:.3f}m")

# 检查未来数据
future_xyz = future_gt['xyz']
print(f"\n未来数据:")
print(f"  shape: {future_xyz.shape} (应为64帧 = 6.4秒)")
print(f"  第一帧(t0+0.1s): [{future_xyz[0][0]:.2f}, {future_xyz[0][1]:.2f}, {future_xyz[0][2]:.2f}]")
print(f"  最后一帧(t0+6.4s): [{future_xyz[-1][0]:.2f}, {future_xyz[-1][1]:.2f}, {future_xyz[-1][2]:.2f}]")

# 检查相邻帧的距离
print(f"\n未来数据相邻帧位移:")
for i in range(1, min(5, len(future_xyz))):
    dist = np.linalg.norm(future_xyz[i][:2] - future_xyz[i-1][:2])
    print(f"  {i-1}->{i}: {dist:.3f}m")

# 验证总位移
hist_total = np.linalg.norm(hist_xyz[-1][:2] - hist_xyz[0][:2])
future_total = np.linalg.norm(future_xyz[-1][:2] - future_xyz[0][:2])
print(f"\n总位移:")
print(f"  历史(1.6s): {hist_total:.2f}m")
print(f"  未来(6.4s): {future_total:.2f}m")
print(f"  平均每帧未来位移: {future_total/63:.2f}m")
