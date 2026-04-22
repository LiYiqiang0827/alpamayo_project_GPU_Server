#!/usr/bin/env python3
import pandas as pd

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
BASE_DIR = "/data01/vla/data/data_sample_chunk0"

# 读取v1和v4的索引
v1_df = pd.read_csv(f"{BASE_DIR}/infer/{clip}/data_v1_backup_0318_1918/inference_index.csv")
v4_df = pd.read_csv(f"{BASE_DIR}/infer/{clip}/data/inference_index.csv")

# 读取相机时间戳
cam_name = 'camera_cross_left_120fov'
ts_path = f"{BASE_DIR}/camera/{clip}.{cam_name}.timestamps.parquet"
ts_df = pd.read_parquet(ts_path)

print("=== 对比v1和v4的帧时间戳 ===\n")

# 找到共同的t0
ts = 249502.0
v1_row = v1_df[v1_df['t0_timestamp'] == ts].iloc[0]
v4_row = v4_df[v4_df['t0_timestamp'] == ts].iloc[0]

print(f"t0={ts}:\n")
print(f"v1 (infer_idx={v1_row['infer_idx']}):")
for f in ['f0', 'f1', 'f2', 'f3']:
    frame_file = v1_row[f'cam_left_{f}']
    frame_num = int(frame_file.split('/')[-1].replace('.jpg', ''))
    frame_ts = ts_df.iloc[frame_num]['timestamp']
    time_diff = (ts - frame_ts) / 1000.0
    print(f"  {f} (帧{frame_num}): 与t0差={time_diff:.1f}ms")

print(f"\nv4 (infer_idx={v4_row['infer_idx']}):")
for f in ['f0', 'f1', 'f2', 'f3']:
    frame_file = v4_row[f'cam_left_{f}']
    frame_num = int(frame_file.split('/')[-1].replace('.jpg', ''))
    frame_ts = ts_df.iloc[frame_num]['timestamp']
    time_diff = (ts - frame_ts) / 1000.0
    print(f"  {f} (帧{frame_num}): 与t0差={time_diff:.1f}ms")
