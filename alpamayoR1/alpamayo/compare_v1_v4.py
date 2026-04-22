#!/usr/bin/env python3
import pandas as pd

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
BASE_DIR = "/data01/vla/data/data_sample_chunk0"

# 读取v1和v4的索引
v1_df = pd.read_csv(f"{BASE_DIR}/infer/{clip}/data_v1_backup_0318_1918/inference_index.csv")
v4_df = pd.read_csv(f"{BASE_DIR}/infer/{clip}/data/inference_index.csv")

print("=== 对比v1和v4的索引 ===\n")
print(f"v1索引: {len(v1_df)} 帧")
print(f"v4索引: {len(v4_df)} 帧")
print()

# 找到共同的帧（通过t0_timestamp）
print("共同帧对比（前5个）:")
v1_ts_set = set(v1_df['t0_timestamp'])
v4_ts_set = set(v4_df['t0_timestamp'])
common_ts = sorted(list(v1_ts_set & v4_ts_set))[:5]

for ts in common_ts:
    v1_row = v1_df[v1_df['t0_timestamp'] == ts].iloc[0]
    v4_row = v4_df[v4_df['t0_timestamp'] == ts].iloc[0]
    
    print(f"t0={ts}:")
    print(f"  v1: infer_idx={v1_row['infer_idx']}, left_f3={v1_row['cam_left_f3']}")
    print(f"  v4: infer_idx={v4_row['infer_idx']}, left_f3={v4_row['cam_left_f3']}")
    print()

# 检查高质量索引
try:
    v1_hq = pd.read_csv(f"{BASE_DIR}/infer/{clip}/data_v1_backup_0318_1918/inference_index_high_quality.csv")
    v4_hq = pd.read_csv(f"{BASE_DIR}/infer/{clip}/data/inference_index_high_quality.csv")
    print(f"高质量索引:")
    print(f"  v1: {len(v1_hq)} 帧")
    print(f"  v4: {len(v4_hq)} 帧")
except:
    print("高质量索引对比失败")
