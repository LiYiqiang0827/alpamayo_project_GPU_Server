#!/usr/bin/env python3
import pandas as pd

clip = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"
BASE_DIR = "/data01/vla/data/data_sample_chunk0"
DATA_DIR = f"{BASE_DIR}/infer/{clip}/data"

# 读取索引
hq_df = pd.read_csv(f"{DATA_DIR}/inference_index_high_quality.csv")

# 读取相机时间戳
cameras = [
    'camera_cross_left_120fov',
    'camera_front_wide_120fov',
    'camera_cross_right_120fov',
    'camera_front_tele_30fov'
]

cam_timestamps = {}
for cam in cameras:
    ts_path = f"{BASE_DIR}/camera/{clip}.{cam}.timestamps.parquet"
    ts_df = pd.read_parquet(ts_path)
    cam_timestamps[cam] = ts_df['timestamp'].values

print("=== 第二个clip相机时间对齐检查 ===\n")

for pos in [0, 1, 2]:
    row = hq_df.iloc[pos]
    t0_ts = row['t0_timestamp']
    infer_idx = row['infer_idx']
    
    print(f"位置{pos} (infer_idx={infer_idx}):")
    print(f"  t0_timestamp: {t0_ts}")
    
    for cam in cameras:
        ts_array = cam_timestamps[cam]
        valid_ts = ts_array[ts_array <= t0_ts]
        if len(valid_ts) > 0:
            closest_ts = valid_ts[-1]
            time_diff_ms = (t0_ts - closest_ts) / 1000.0
            print(f"  {cam}: 帧号{len(valid_ts)-1}, 时间差={time_diff_ms:.1f}ms")
    
    print(f"  记录max_image_diff_ms: {row['max_image_diff_ms']:.1f}ms")
    print()
