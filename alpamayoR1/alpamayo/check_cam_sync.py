#!/usr/bin/env python3
"""检查相机时间戳差异"""
import pandas as pd

BASE_DIR = "/data01/vla/data/data_sample_chunk0"

clips = [
    ("01d3588e-bca7-4a18-8e74-c6cfe9e996db", "参考clip"),
    ("b9b4ddc7-feb0-4749-9d14-931a70fe3e17", "相似clip")
]

print("=== 检查相机时间戳 ===\n")

for clip_id, desc in clips:
    DATA_DIR = f"{BASE_DIR}/infer/{clip_id}/data"
    
    # 读取索引
    index_df = pd.read_csv(f"{DATA_DIR}/inference_index.csv")
    row = index_df.iloc[100]  # Frame 100
    t0_ts = row['t0_timestamp']
    
    print(f"{desc} (Frame 100, t0={t0_ts}):")
    
    # 读取相机时间戳
    cam_name = 'camera_cross_left_120fov'
    ts_path = f"{BASE_DIR}/camera/{clip_id}.{cam_name}.timestamps.parquet"
    ts_df = pd.read_parquet(ts_path)
    
    # 检查f3帧的时间差
    frame_file = row['cam_left_f3']
    frame_num = int(frame_file.split('/')[-1].replace('.jpg', ''))
    frame_ts = ts_df.iloc[frame_num]['timestamp']
    time_diff = (t0_ts - frame_ts) / 1000.0
    
    print(f"  cam_left_f3 (帧{frame_num}): 与t0差={time_diff:.1f}ms")
    print()
