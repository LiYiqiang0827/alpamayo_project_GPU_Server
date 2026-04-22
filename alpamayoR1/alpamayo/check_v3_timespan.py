#!/usr/bin/env python3
import numpy as np
import pandas as pd

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
DATA_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data"

index_df = pd.read_csv(f"{DATA_DIR}/inference_index.csv")

print("=== v3预处理 - 历史数据时间跨度检查 ===\n")

for infer_idx in [0, 100, 500]:
    prefix = f"ego_{infer_idx:06d}"
    ego_idx = index_df.iloc[infer_idx]["ego_idx"]
    
    hist_world = np.load(f"{DATA_DIR}/egomotion/{prefix}_history_world.npy", allow_pickle=True).item()
    
    # 检查实际覆盖的时间
    print(f"Frame {infer_idx} (ego_idx={ego_idx}):")
    print(f"  历史数据shape: {hist_world['xyz'].shape}")
    print(f"  第一帧: [{hist_world['xyz'][0][0]:.2f}, {hist_world['xyz'][0][1]:.2f}]")
    print(f"  最后一帧(t0): [{hist_world['xyz'][-1][0]:.2f}, {hist_world['xyz'][-1][1]:.2f}]")
    
    # 计算位移
    displacement = np.linalg.norm(hist_world['xyz'][-1][:2] - hist_world['xyz'][0][:2])
    print(f"  历史位移: {displacement:.2f}m (期望~1.6秒内车辆行驶的位移)")
    print()
