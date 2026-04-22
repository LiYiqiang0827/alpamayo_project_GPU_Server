#!/usr/bin/env python3
import numpy as np
import pandas as pd

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
DATA_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data"

index_df = pd.read_csv(f"{DATA_DIR}/inference_index.csv")
print(f"总有效帧: {len(index_df)}")

# 检查几个帧的egomotion
for infer_idx in [0, 100, 500, 1000]:
    prefix = f"ego_{infer_idx:06d}"
    ego_idx = index_df.iloc[infer_idx]["ego_idx"]
    t0_ts = index_df.iloc[infer_idx]["t0_timestamp"]
    
    hist_local = np.load(f"{DATA_DIR}/egomotion/{prefix}_history_local.npy", allow_pickle=True).item()
    hist_world = np.load(f"{DATA_DIR}/egomotion/{prefix}_history_world.npy", allow_pickle=True).item()
    future_gt = np.load(f"{DATA_DIR}/egomotion/{prefix}_future_gt.npy", allow_pickle=True).item()
    
    print(f"\n=== Frame {infer_idx} (ego_idx={ego_idx}) ===")
    xyz_key = 'xyz'
    print(f"  历史世界坐标 shape: {hist_world[xyz_key].shape}")
    print(f"    第一帧: [{hist_world[xyz_key][0][0]:.2f}, {hist_world[xyz_key][0][1]:.2f}, {hist_world[xyz_key][0][2]:.2f}]")
    print(f"    最后一帧(t0): [{hist_world[xyz_key][-1][0]:.2f}, {hist_world[xyz_key][-1][1]:.2f}, {hist_world[xyz_key][-1][2]:.2f}]")
    print(f"  历史局部坐标 shape: {hist_local[xyz_key].shape}")
    print(f"    第一帧: [{hist_local[xyz_key][0][0]:.2f}, {hist_local[xyz_key][0][1]:.2f}, {hist_local[xyz_key][0][2]:.2f}]")
    print(f"    最后一帧(t0): [{hist_local[xyz_key][-1][0]:.2f}, {hist_local[xyz_key][-1][1]:.2f}, {hist_local[xyz_key][-1][2]:.2f}]")
    print(f"  未来真值 shape: {future_gt[xyz_key].shape}")
    print(f"    第一帧(t0+0.1s): [{future_gt[xyz_key][0][0]:.2f}, {future_gt[xyz_key][0][1]:.2f}, {future_gt[xyz_key][0][2]:.2f}]")
    print(f"    最后一帧(t0+6.4s): [{future_gt[xyz_key][-1][0]:.2f}, {future_gt[xyz_key][-1][1]:.2f}, {future_gt[xyz_key][-1][2]:.2f}]")

    # 检查局部坐标系原点附近
    t0_local = hist_local[xyz_key][-1]
    dist_from_origin = np.linalg.norm(t0_local[:2])
    print(f"  t0在局部坐标系中距离原点: {dist_from_origin:.4f}m")
