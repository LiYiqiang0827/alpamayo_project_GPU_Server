#!/usr/bin/env python3
import pandas as pd

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
BASE_DIR = "/data01/vla/data/data_sample_chunk0"

# 读取v1高质量索引
v1_hq = pd.read_csv(f"{BASE_DIR}/infer/{clip}/data_v1_backup_0318_1918/inference_index_high_quality.csv")
row = v1_hq.iloc[0]  # 第一个高质量帧
t0_ts = row["t0_timestamp"]

print(f"v1高质量索引第一帧: t0={t0_ts}")
print(f"infer_idx: {row['infer_idx']}")
print()

# 读取相机时间戳
cam_name = "camera_cross_left_120fov"
ts_path = f"{BASE_DIR}/camera/{clip}.{cam_name}.timestamps.parquet"
ts_df = pd.read_parquet(ts_path)

print(f"{cam_name} 帧时间戳:")
for f in ['f0', 'f1', 'f2', 'f3']:
    frame_file = row[f'cam_left_{f}']
    frame_num = int(frame_file.split("/")[-1].replace(".jpg", ""))
    frame_ts = ts_df.iloc[frame_num]["timestamp"]
    time_diff = (t0_ts - frame_ts) / 1000.0
    print(f"  {f} (帧{frame_num}): 与t0差={time_diff:.1f}ms")
