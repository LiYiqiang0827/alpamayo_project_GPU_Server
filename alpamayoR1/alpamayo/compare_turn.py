#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

clip = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"
DATA_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data"

# 读取转弯帧的egomotion (infer_idx=1903, ADE=31m)
prefix = "ego_001903"
hist_local = np.load(f"{DATA_DIR}/egomotion/{prefix}_history_local.npy", allow_pickle=True).item()
future_gt = np.load(f"{DATA_DIR}/egomotion/{prefix}_future_gt.npy", allow_pickle=True).item()

hist_xyz = hist_local['xyz']
future_xyz = future_gt['xyz']

print("=== 转弯场景数据检查 (Frame 1903, ADE=31m) ===\n")

print("历史数据 (1.6s):")
print(f"  第一帧: [{hist_xyz[0][0]:.2f}, {hist_xyz[0][1]:.2f}]")
print(f"  最后一帧(t0): [{hist_xyz[-1][0]:.2f}, {hist_xyz[-1][1]:.2f}]")

print("\n未来数据 (6.4s):")
print(f"  第一帧: [{future_xyz[0][0]:.2f}, {future_xyz[0][1]:.2f}]")
print(f"  第10帧: [{future_xyz[9][0]:.2f}, {future_xyz[9][1]:.2f}]")
print(f"  第30帧: [{future_xyz[29][0]:.2f}, {future_xyz[29][1]:.2f}]")
print(f"  最后一帧: [{future_xyz[-1][0]:.2f}, {future_xyz[-1][1]:.2f}]")

# 计算转弯角度
hist_angle = np.arctan2(hist_xyz[-1][1] - hist_xyz[0][1], hist_xyz[-1][0] - hist_xyz[0][0]) * 180 / np.pi
future_angle = np.arctan2(future_xyz[-1][1] - future_xyz[0][1], future_xyz[-1][0] - future_xyz[0][0]) * 180 / np.pi
print(f"\n轨迹方向:")
print(f"  历史轨迹角度: {hist_angle:.1f}°")
print(f"  未来轨迹角度: {future_angle:.1f}°")
print(f"  转弯角度: {future_angle - hist_angle:.1f}°")

# 对比直行clip
print("\n=== 对比直行场景 (Frame 138, ADE=0.8m) ===")
clip2 = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
DATA_DIR2 = f"/data01/vla/data/data_sample_chunk0/infer/{clip2}/data"
prefix2 = "ego_000138"
hist2 = np.load(f"{DATA_DIR2}/egomotion/{prefix2}_history_local.npy", allow_pickle=True).item()
future2 = np.load(f"{DATA_DIR2}/egomotion/{prefix2}_future_gt.npy", allow_pickle=True).item()

hist_xyz2 = hist2['xyz']
future_xyz2 = future2['xyz']

hist_angle2 = np.arctan2(hist_xyz2[-1][1] - hist_xyz2[0][1], hist_xyz2[-1][0] - hist_xyz2[0][0]) * 180 / np.pi
future_angle2 = np.arctan2(future_xyz2[-1][1] - future_xyz2[0][1], future_xyz2[-1][0] - future_xyz2[0][0]) * 180 / np.pi

print(f"历史轨迹角度: {hist_angle2:.1f}°")
print(f"未来轨迹角度: {future_angle2:.1f}°")
print(f"转弯角度: {future_angle2 - hist_angle2:.1f}°")
