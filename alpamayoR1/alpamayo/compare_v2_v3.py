#!/usr/bin/env python3
import numpy as np
import pandas as pd

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
V2_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data_v2_backup"
V3_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data"

print("=== v2 vs v3 历史数据对比 ===\n")

for infer_idx in [0, 100, 500]:
    prefix = f"ego_{infer_idx:06d}"
    
    try:
        v2_world = np.load(f"{V2_DIR}/egomotion/{prefix}_history_world.npy", allow_pickle=True).item()
        v3_world = np.load(f"{V3_DIR}/egomotion/{prefix}_history_world.npy", allow_pickle=True).item()
        
        v2_disp = np.linalg.norm(v2_world['xyz'][-1][:2] - v2_world['xyz'][0][:2])
        v3_disp = np.linalg.norm(v3_world['xyz'][-1][:2] - v3_world['xyz'][0][:2])
        
        print(f"Frame {infer_idx}:")
        print(f"  v2历史位移: {v2_disp:.2f}m (0.15秒)")
        print(f"  v3历史位移: {v3_disp:.2f}m (1.50秒)")
        print(f"  差距: {v3_disp/v2_disp:.1f}x")
        print()
    except Exception as e:
        print(f"Frame {infer_idx}: {e}")
