#!/usr/bin/env python3
import pandas as pd

clip = "054da32b-9f3d-4074-93ab-044036b679f8"
BASE_DIR = "/data01/vla/data/data_sample_chunk0"
DATA_DIR = f"{BASE_DIR}/infer/{clip}/data"

# 读取索引
index_df = pd.read_csv(f"{DATA_DIR}/inference_index.csv")
row = index_df.iloc[0]
t0_ts = row['t0_timestamp']

print(f"=== Clip: {clip}, Frame 0 ===")
print(f"t0_timestamp: {t0_ts}")
print()

# 检查各相机f0-f3的时间戳
cameras = [
    ('camera_cross_left_120fov', 'cam_left'),
    ('camera_front_wide_120fov', 'cam_front'),
    ('camera_cross_right_120fov', 'cam_right'),
    ('camera_front_tele_30fov', 'cam_tele')
]

for cam_name, cam_prefix in cameras:
    ts_path = f"{BASE_DIR}/camera/{clip}.{cam_name}.timestamps.parquet"
    ts_df = pd.read_parquet(ts_path)
    
    print(f"{cam_name}:")
    for i, f in enumerate(['f0', 'f1', 'f2', 'f3']):
        frame_file = row[f'{cam_prefix}_{f}']
        frame_num = int(frame_file.split('/')[-1].replace('.jpg', ''))
        frame_ts = ts_df.iloc[frame_num]['timestamp']
        time_diff = (t0_ts - frame_ts) / 1000.0
        print(f"  {f} (帧{frame_num}): ts={frame_ts}, 与t0差={time_diff:.1f}ms")
    print()

# 检查时间跨度
print("时间跨度检查:")
print(f"  f0到f3应该是300ms (4帧@30Hz)")
