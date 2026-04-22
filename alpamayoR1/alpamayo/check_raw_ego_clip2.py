#!/usr/bin/env python3
import pandas as pd

clip = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"
ego_path = f"/data01/vla/data/data_sample_chunk0/labels/egomotion/{clip}.egomotion.parquet"

ego_df = pd.read_parquet(ego_path)

print("=== 第二个clip原始egomotion检查 ===\n")
print(f"总帧数: {len(ego_df)}")
print(f"\n前20帧的xyz:")
for i in range(20):
    x, y, z = ego_df.iloc[i]['x'], ego_df.iloc[i]['y'], ego_df.iloc[i]['z']
    ts = ego_df.iloc[i]['timestamp']
    print(f"  {i}: ts={ts}, xyz=[{x:.2f}, {y:.2f}, {z:.2f}]")

print(f"\n第100-110帧的xyz:")
for i in range(100, 110):
    x, y, z = ego_df.iloc[i]['x'], ego_df.iloc[i]['y'], ego_df.iloc[i]['z']
    ts = ego_df.iloc[i]['timestamp']
    print(f"  {i}: ts={ts}, xyz=[{x:.2f}, {y:.2f}, {z:.2f}]")
