#!/usr/bin/env python3
import pandas as pd
import numpy as np

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
DATA_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data"

# 读取索引和原始ego
index_df = pd.read_csv(f"{DATA_DIR}/inference_index.csv")
ego_path = f"/data01/vla/data/data_sample_chunk0/labels/egomotion/{clip}.egomotion.parquet"
ego_df = pd.read_parquet(ego_path)

print("=== 检查v2预处理的历史数据时间跨度 ===\n")

for infer_idx in [0, 100, 500]:
    ego_idx = index_df.iloc[infer_idx]["ego_idx"]
    
    # v2方式：直接取16帧
    hist_v2 = ego_df.iloc[ego_idx - 16:ego_idx]
    time_span_v2 = (hist_v2['timestamp'].iloc[-1] - hist_v2['timestamp'].iloc[0]) / 1e6
    
    # 正确方式：按10Hz采样
    t0_ts = ego_df.iloc[ego_idx]['timestamp']
    hist_timestamps = [t0_ts - i * 100000 for i in range(16, 0, -1)]  # 10Hz: 100ms间隔
    time_span_correct = (hist_timestamps[-1] - hist_timestamps[0]) / 1e6
    
    print(f"Frame {infer_idx} (ego_idx={ego_idx}):")
    print(f"  v2方式 (16帧):  {time_span_v2:.3f}秒  ❌")
    print(f"  正确 (10Hz采样): {time_span_correct:.3f}秒  ✅")
    print()
