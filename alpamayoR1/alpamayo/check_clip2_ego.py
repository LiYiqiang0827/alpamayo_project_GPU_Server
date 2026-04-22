#!/usr/bin/env python3
import numpy as np
import pandas as pd

clip = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"
DATA_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data"

index_df = pd.read_csv(f"{DATA_DIR}/inference_index.csv")

print("=== 第二个clip历史数据检查 ===\n")

for infer_idx in [0, 100, 500]:
    prefix = f"ego_{infer_idx:06d}"
    
    hist_local = np.load(f"{DATA_DIR}/egomotion/{prefix}_history_local.npy", allow_pickle=True).item()
    hist_world = np.load(f"{DATA_DIR}/egomotion/{prefix}_history_world.npy", allow_pickle=True).item()
    future_gt = np.load(f"{DATA_DIR}/egomotion/{prefix}_future_gt.npy", allow_pickle=True).item()
    
    xyz_key = 'xyz'
    local_disp = np.linalg.norm(hist_local[xyz_key][-1][:2] - hist_local[xyz_key][0][:2])
    world_disp = np.linalg.norm(hist_world[xyz_key][-1][:2] - hist_world[xyz_key][0][:2])
    
    print(f"Frame {infer_idx}:")
    print(f"  历史局部坐标位移: {local_disp:.2f}m")
    print(f"  历史世界坐标位移: {world_disp:.2f}m")
    print(f"  未来真值第一帧: [{future_gt[xyz_key][0][0]:.2f}, {future_gt[xyz_key][0][1]:.2f}]")
    print(f"  未来真值最后一帧: [{future_gt[xyz_key][-1][0]:.2f}, {future_gt[xyz_key][-1][1]:.2f}]")
    
    # 检查t0是否在局部坐标系原点
    t0_local = hist_local[xyz_key][-1]
    dist_from_origin = np.linalg.norm(t0_local[:2])
    print(f"  t0距离局部原点: {dist_from_origin:.4f}m")
    print()
