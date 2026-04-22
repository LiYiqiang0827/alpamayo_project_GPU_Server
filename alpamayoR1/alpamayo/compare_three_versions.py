#!/usr/bin/env python3
import numpy as np
import pandas as pd

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"

# 三个版本的数据目录
V1_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data_backup_0318_1810"  # 最早正确
V2_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data_v2_backup"  # v2版本
V3_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data"  # 当前v3

print("=== 三个版本的历史数据对比 ===\n")

for infer_idx in [0, 100, 500]:
    prefix = f"ego_{infer_idx:06d}"
    print(f"\n--- Frame {infer_idx} ---")
    
    for ver, path in [("V1(正确)", V1_DIR), ("V2", V2_DIR), ("V3(当前)", V3_DIR)]:
        try:
            hist = np.load(f"{path}/egomotion/{prefix}_history_local.npy", allow_pickle=True).item()
            xyz = hist['xyz']
            
            # 计算历史跨度
            displacement = np.linalg.norm(xyz[-1][:2] - xyz[0][:2])
            print(f"  {ver}: shape={xyz.shape}, 位移={displacement:.2f}m, 起点={xyz[0][:2]}, 终点={xyz[-1][:2]}")
        except Exception as e:
            print(f"  {ver}: 错误 - {e}")
