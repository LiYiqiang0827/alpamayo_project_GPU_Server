#!/usr/bin/env python3
import pandas as pd

clip = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"
DATA_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data"

index_df = pd.read_csv(f"{DATA_DIR}/inference_index.csv")

print("=== 检查图像帧索引顺序 ===\n")

for infer_idx in [0, 100]:
    row = index_df.iloc[infer_idx]
    print(f"Frame {infer_idx} (infer_idx={row['infer_idx']}, t0={row['t0_timestamp']}):")
    print(f"  cam_left_f0:  {row['cam_left_f0']}")
    print(f"  cam_left_f1:  {row['cam_left_f1']}")
    print(f"  cam_left_f2:  {row['cam_left_f2']}")
    print(f"  cam_left_f3:  {row['cam_left_f3']}")
    print()
