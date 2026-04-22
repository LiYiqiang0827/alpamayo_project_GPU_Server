#!/usr/bin/env python3
import pandas as pd
import numpy as np

clip = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"
BASE_DIR = "/data01/vla/data/data_sample_chunk0"

ego_path = f"{BASE_DIR}/labels/egomotion/{clip}.egomotion.parquet"
ego_df = pd.read_parquet(ego_path)

print("=== 检查原始数据时间戳 ===\n")

# 检查索引1800到1950的时间戳
print("索引1800-1950的时间戳差异:")
for idx in [1800, 1850, 1900, 1934, 1950]:
    ts = ego_df.iloc[idx]['timestamp']
    print(f"  索引{idx}: {ts}")

# 计算时间差
ts_1800 = ego_df.iloc[1800]['timestamp']
ts_1950 = ego_df.iloc[1950]['timestamp']
print(f"\n索引1800到1950的时间差: {(ts_1950 - ts_1800)/1000:.1f}s")
print(f"帧数差: {1950-1800}帧")
print(f"平均间隔: {(ts_1950 - ts_1800)/(1950-1800)/1000:.1f}ms")

# 检查是否有异常间隔
print(f"\n索引1930-1950的时间戳:")
for i in range(1930, 1951):
    ts = ego_df.iloc[i]['timestamp']
    print(f"  {i}: {ts}")
