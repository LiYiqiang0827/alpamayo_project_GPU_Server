#!/usr/bin/env python3
import pandas as pd

clip1 = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
clip2 = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"

for clip in [clip1, clip2]:
    DATA_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data"
    index_df = pd.read_csv(f"{DATA_DIR}/inference_index.csv")
    
    row = index_df.iloc[100]
    print(f"=== {clip[:8]}... Frame 100 ===")
    print(f"  t0_timestamp: {row['t0_timestamp']}")
    print(f"  cam_left_f3:  {row['cam_left_f3']}")
    print(f"  cam_front_f3: {row['cam_front_f3']}")
    print()
