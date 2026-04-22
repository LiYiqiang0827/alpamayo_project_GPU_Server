#!/usr/bin/env python3
import numpy as np
import pandas as pd

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
DATA_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data"

# 读取高质量索引
hq_df = pd.read_csv(f"{DATA_DIR}/inference_index_high_quality.csv")

# 找到infer_idx=138对应的行
row_138 = hq_df[hq_df['infer_idx'] == 138]
if len(row_138) > 0:
    prefix = "ego_000138"
    
    hist_local = np.load(f"{DATA_DIR}/egomotion/{prefix}_history_local.npy", allow_pickle=True).item()
    future_gt = np.load(f"{DATA_DIR}/egomotion/{prefix}_future_gt.npy", allow_pickle=True).item()
    
    hist_xyz = hist_local['xyz']
    future_xyz = future_gt['xyz']
    
    print(f"好的clip (Frame 138, ADE=0.8m):")
    print(f"  历史位移: {np.linalg.norm(hist_xyz[-1][:2] - hist_xyz[0][:2]):.2f}m")
    print(f"  未来位移: {np.linalg.norm(future_xyz[-1][:2] - future_xyz[0][:2]):.2f}m")
    print(f"  未来最后坐标: [{future_xyz[-1][0]:.2f}, {future_xyz[-1][1]:.2f}]")
else:
    print("Frame 138不在高质量索引中")
    print(f"高质量索引中的infer_idx范围: {hq_df['infer_idx'].min()} - {hq_df['infer_idx'].max()}")
