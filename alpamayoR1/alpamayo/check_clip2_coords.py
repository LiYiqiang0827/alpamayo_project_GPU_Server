#!/usr/bin/env python3
import numpy as np
import pandas as pd

clip = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"
DATA_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data"

print("=== 第二个clip坐标系检查 ===\n")

for infer_idx in [0, 100, 200]:
    prefix = f"ego_{infer_idx:06d}"
    
    hist_local = np.load(f"{DATA_DIR}/egomotion/{prefix}_history_local.npy", allow_pickle=True).item()
    hist_world = np.load(f"{DATA_DIR}/egomotion/{prefix}_history_world.npy", allow_pickle=True).item()
    future_gt = np.load(f"{DATA_DIR}/egomotion/{prefix}_future_gt.npy", allow_pickle=True).item()
    
    xyz_key = 'xyz'
    
    print(f"Frame {infer_idx}:")
    print(f"  历史局部坐标 (前3帧):")
    for i in range(3):
        print(f"    [{hist_local[xyz_key][i][0]:.2f}, {hist_local[xyz_key][i][1]:.2f}]")
    print(f"  历史局部坐标 (t0): [{hist_local[xyz_key][-1][0]:.2f}, {hist_local[xyz_key][-1][1]:.2f}]")
    
    print(f"  未来真值 (前3帧):")
    for i in range(3):
        print(f"    [{future_gt[xyz_key][i][0]:.2f}, {future_gt[xyz_key][i][1]:.2f}]")
    print(f"  未来真值 (最后): [{future_gt[xyz_key][-1][0]:.2f}, {future_gt[xyz_key][-1][1]:.2f}]")
    
    # 检查历史轨迹方向（Y轴应该是前进方向）
    hist_y_diff = hist_local[xyz_key][-1][1] - hist_local[xyz_key][0][1]
    print(f"  历史Y轴变化: {hist_y_diff:.2f}m (应该是正值，表示前进)")
    
    # 检查未来轨迹方向
    future_y_diff = future_gt[xyz_key][-1][1] - future_gt[xyz_key][0][1]
    print(f"  未来Y轴变化: {future_y_diff:.2f}m (应该是正值，表示前进)")
    print()
