#!/usr/bin/env python3
import pandas as pd
import numpy as np

clip = "054da32b-9f3d-4074-93ab-044036b679f8"
BASE_DIR = "/data01/vla/data/data_sample_chunk0"
DATA_DIR = f"{BASE_DIR}/infer/{clip}/data"

# 读取索引
index_df = pd.read_csv(f"{DATA_DIR}/inference_index.csv")
row = index_df.iloc[0]

print("=== 验证 Clip:", clip, "===")
print(f"infer_idx: {row['infer_idx']}")
print(f"t0_timestamp: {row['t0_timestamp']}")
print()

# 检查相机图像路径
print("相机图像路径:")
for cam in ['cam_left', 'cam_front', 'cam_right', 'cam_tele']:
    key = f'{cam}_f3'
    f3 = row[key]
    print(f"  {key}: {f3}")
print()

# 检查egomotion文件
print("Egomotion文件:")
prefix = f"ego_{int(row['infer_idx']):06d}"
print(f"  前缀: {prefix}")
print(f"  文件: {DATA_DIR}/egomotion/{prefix}_history_local.npy")
print()

# 读取原始egomotion验证时间戳
ego_path = f"{BASE_DIR}/labels/egomotion/{clip}.egomotion.parquet"
ego_df = pd.read_parquet(ego_path)

# 找到t0对应的ego帧
t0_ts = row['t0_timestamp']
idx = (ego_df['timestamp'] - t0_ts).abs().idxmin()
print(f"Egomotion验证:")
print(f"  t0_timestamp: {t0_ts}")
print(f"  最接近的ego帧: idx={idx}, ts={ego_df.iloc[idx]['timestamp']}")
print(f"  时间差: {abs(ego_df.iloc[idx]['timestamp'] - t0_ts)/1000:.2f}ms")
print()

# 检查历史数据时间跨度
hist_local = np.load(f"{DATA_DIR}/egomotion/{prefix}_history_local.npy", allow_pickle=True).item()
print(f"历史数据:")
print(f"  shape: {hist_local['xyz'].shape}")
print(f"  第一帧: {hist_local['xyz'][0]}")
print(f"  最后一帧(t0): {hist_local['xyz'][-1]}")
