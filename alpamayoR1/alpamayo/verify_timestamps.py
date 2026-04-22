#!/usr/bin/env python3
import pandas as pd

clip = "054da32b-9f3d-4074-93ab-044036b679f8"
BASE_DIR = "/data01/vla/data/data_sample_chunk0"
DATA_DIR = f"{BASE_DIR}/infer/{clip}/data"

# 读取索引
index_df = pd.read_csv(f"{DATA_DIR}/inference_index.csv")
row = index_df.iloc[0]
t0_ts = row['t0_timestamp']

print(f"=== Clip: {clip} ===")
print(f"t0_timestamp: {t0_ts}")
print()

# 读取各相机时间戳
cameras = [
    ('camera_cross_left_120fov', 'cam_left'),
    ('camera_front_wide_120fov', 'cam_front'),
    ('camera_cross_right_120fov', 'cam_right'),
    ('camera_front_tele_30fov', 'cam_tele')
]

print("相机帧时间对齐:")
for cam_name, cam_prefix in cameras:
    ts_path = f"{BASE_DIR}/camera/{clip}.{cam_name}.timestamps.parquet"
    ts_df = pd.read_parquet(ts_path)
    
    # 找到t0对应的帧
    ts_array = ts_df['timestamp'].values
    valid_ts = ts_array[ts_array <= t0_ts]
    if len(valid_ts) > 0:
        closest_idx = len(valid_ts) - 1
        closest_ts = valid_ts[-1]
        time_diff_ms = (t0_ts - closest_ts) / 1000.0
        
        print(f"  {cam_name}:")
        print(f"    帧号: {closest_idx}")
        print(f"    时间戳: {closest_ts}")
        print(f"    与t0差: {time_diff_ms:.1f}ms")

print()

# 读取原始egomotion
ego_path = f"{BASE_DIR}/labels/egomotion/{clip}.egomotion.parquet"
ego_df = pd.read_parquet(ego_path)

# 找到t0对应的ego帧
idx = (ego_df['timestamp'] - t0_ts).abs().idxmin()
print(f"Egomotion:")
print(f"  帧号: {idx}")
print(f"  时间戳: {ego_df.iloc[idx]['timestamp']}")
print(f"  与t0差: {abs(ego_df.iloc[idx]['timestamp'] - t0_ts)/1000:.2f}ms")
