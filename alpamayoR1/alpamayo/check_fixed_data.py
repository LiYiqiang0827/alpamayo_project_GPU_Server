#!/usr/bin/env python3
import numpy as np

clip = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"
DATA_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data"

# 读取修复后的数据
prefix = "ego_001000"
hist_local = np.load(f"{DATA_DIR}/egomotion/{prefix}_history_local.npy", allow_pickle=True).item()
future_gt = np.load(f"{DATA_DIR}/egomotion/{prefix}_future_gt.npy", allow_pickle=True).item()

hist_xyz = hist_local['xyz']
future_xyz = future_gt['xyz']

print("=== 修复后数据检查 (Frame 1000) ===")
print(f"历史数据:")
print(f"  shape: {hist_xyz.shape}")
print(f"  第一帧: [{hist_xyz[0][0]:.2f}, {hist_xyz[0][1]:.2f}]")
print(f"  最后一帧(t0): [{hist_xyz[-1][0]:.2f}, {hist_xyz[-1][1]:.2f}]")
print(f"  位移: {np.linalg.norm(hist_xyz[-1][:2] - hist_xyz[0][:2]):.2f}m")

print(f"\n未来数据:")
print(f"  shape: {future_xyz.shape}")
print(f"  第一帧: [{future_xyz[0][0]:.2f}, {future_xyz[0][1]:.2f}]")
print(f"  最后一帧: [{future_xyz[-1][0]:.2f}, {future_xyz[-1][1]:.2f}]")
print(f"  位移: {np.linalg.norm(future_xyz[-1][:2] - future_xyz[0][:2]):.2f}m")
