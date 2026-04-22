#!/usr/bin/env python3
import pandas as pd
import numpy as np

clip = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"
BASE_DIR = "/data01/vla/data/data_sample_chunk0"

ego_path = f"{BASE_DIR}/labels/egomotion/{clip}.egomotion.parquet"
ego_df = pd.read_parquet(ego_path)

# 从索引文件读取
data_dir = f"{BASE_DIR}/infer/{clip}/data"
index_df = pd.read_csv(f"{data_dir}/inference_index.csv")
row = index_df[index_df['infer_idx'] == 1903].iloc[0]

t0_ts_from_index = row['t0_timestamp']
ego_idx_from_index = row['ego_idx']

print("=== 索引匹配检查 ===\n")
print(f"从索引文件:")
print(f"  t0_timestamp: {t0_ts_from_index}")
print(f"  ego_idx: {ego_idx_from_index}")

# 手动查找
t0_ts_actual = 19349601.0
idx_actual = (ego_df['timestamp'] - t0_ts_actual).abs().idxmin()

print(f"\n手动查找 (t0_ts={t0_ts_actual}):")
print(f"  找到索引: {idx_actual}")
print(f"  时间戳: {ego_df.iloc[idx_actual]['timestamp']}")

# 检查差异
print(f"\n差异:")
print(f"  ego_idx 差异: {ego_idx_from_index - idx_actual}")
print(f"  t0_ts 差异: {t0_ts_from_index - t0_ts_actual}us = {(t0_ts_from_index - t0_ts_actual)/1000:.2f}ms")

# 检查该索引对应的实际坐标
print(f"\n索引{ego_idx_from_index}的坐标:")
xyz = ego_df.iloc[ego_idx_from_index][['x', 'y', 'z']].values
print(f"  [{xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}]")

print(f"\n索引{idx_actual}的坐标:")
xyz2 = ego_df.iloc[idx_actual][['x', 'y', 'z']].values
print(f"  [{xyz2[0]:.2f}, {xyz2[1]:.2f}, {xyz2[2]:.2f}]")
