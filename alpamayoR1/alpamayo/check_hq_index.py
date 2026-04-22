#!/usr/bin/env python3
import pandas as pd

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
DATA_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data"

normal_df = pd.read_csv(f"{DATA_DIR}/inference_index.csv")
hq_df = pd.read_csv(f"{DATA_DIR}/inference_index_high_quality.csv")

print(f"普通索引: {len(normal_df)} 帧")
print(f"高质量索引: {len(hq_df)} 帧")
print()

key = 'infer_idx'
print("Frame 138:")
normal_pos = normal_df[normal_df[key] == 138].index.tolist()
hq_pos = hq_df[hq_df[key] == 138].index.tolist()
print(f"  普通索引中的位置: {normal_pos}")
print(f"  高质量索引中的位置: {hq_pos}")
print()

# 如果138在高质量索引中，查看它的实际ego_idx
if hq_pos:
    ego_idx_hq = hq_df.iloc[hq_pos[0]]['ego_idx']
    ego_idx_normal = normal_df.iloc[normal_pos[0]]['ego_idx']
    print(f"Frame 138 - 普通索引 ego_idx: {ego_idx_normal}")
    print(f"Frame 138 - 高质量索引 ego_idx: {ego_idx_hq}")
