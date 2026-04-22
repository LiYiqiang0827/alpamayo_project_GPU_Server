#!/usr/bin/env python3
import pandas as pd

clip = "054da32b-9f3d-4074-93ab-044036b679f8"
DATA_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data"

# 读取高质量索引
hq_df = pd.read_csv(f"{DATA_DIR}/inference_index_high_quality.csv")
row = hq_df.iloc[0]

print("=== 验证推理时加载的图像路径 ===")
print(f"infer_idx: {row['infer_idx']}")
print(f"t0_timestamp: {row['t0_timestamp']}")
print()

# 相机顺序（与推理脚本一致）
CAMERA_ORDER = [
    'camera_cross_left_120fov',
    'camera_front_wide_120fov', 
    'camera_cross_right_120fov',
    'camera_front_tele_30fov'
]

cam_cols = {
    'camera_cross_left_120fov': ['cam_left_f0', 'cam_left_f1', 'cam_left_f2', 'cam_left_f3'],
    'camera_front_wide_120fov': ['cam_front_f0', 'cam_front_f1', 'cam_front_f2', 'cam_front_f3'],
    'camera_cross_right_120fov': ['cam_right_f0', 'cam_right_f1', 'cam_right_f2', 'cam_right_f3'],
    'camera_front_tele_30fov': ['cam_tele_f0', 'cam_tele_f1', 'cam_tele_f2', 'cam_tele_f3'],
}

print("推理时加载的图像顺序:")
for cam in CAMERA_ORDER:
    print(f"\n  {cam}:")
    for col in cam_cols[cam]:
        img_path = f"{DATA_DIR}/{row[col]}"
        print(f"    {col}: {img_path}")
