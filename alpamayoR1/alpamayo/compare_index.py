#!/usr/bin/env python3
import pandas as pd

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"

V1_IDX = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data_backup_0318_1810/inference_index.csv"
V3_IDX = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data/inference_index.csv"

v1_df = pd.read_csv(V1_IDX)
v3_df = pd.read_csv(V3_IDX)

print("=== 索引对比 ===")
print(f"V1 总行数: {len(v1_df)}")
print(f"V3 总行数: {len(v3_df)}")
print()

for i in [0, 100, 500]:
    print(f"Frame {i}:")
    key_ego = 'ego_idx'
    key_t0 = 't0_timestamp'
    key_f3 = 'cam_left_f3'
    print(f"  V1 ego_idx: {v1_df.iloc[i][key_ego]}, t0: {v1_df.iloc[i][key_t0]}")
    print(f"  V3 ego_idx: {v3_df.iloc[i][key_ego]}, t0: {v3_df.iloc[i][key_t0]}")
    print(f"  V1 cam_left_f3: {v1_df.iloc[i][key_f3]}")
    print(f"  V3 cam_left_f3: {v3_df.iloc[i][key_f3]}")
    print()
