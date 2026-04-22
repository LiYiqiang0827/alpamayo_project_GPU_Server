#!/usr/bin/env python3
import pandas as pd
import numpy as np

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
ego_path = f"/data01/vla/data/data_sample_chunk0/labels/egomotion/{clip}.egomotion.parquet"
ego_df = pd.read_parquet(ego_path)

print("=== 原始egomotion数据分析 ===")
print(f"总帧数: {len(ego_df)}")

# 检查时间戳间隔
ts_col = 'timestamp'
ts_diff = ego_df[ts_col].diff().dropna()
print(f"\n时间戳间隔统计:")
print(f"  均值: {ts_diff.mean():.0f}us = {ts_diff.mean()/1000:.2f}ms")
print(f"  中位数: {ts_diff.median():.0f}us = {ts_diff.median()/1000:.2f}ms")
print(f"  最小: {ts_diff.min():.0f}us")
print(f"  最大: {ts_diff.max():.0f}us")

# 计算频率
freq = 1e6 / ts_diff.median()
print(f"\n采样频率: {freq:.1f}Hz")

# 检查前20个间隔
print(f"\n前20个时间戳间隔:")
for i in range(1, 21):
    diff_us = ego_df[ts_col].iloc[i] - ego_df[ts_col].iloc[i-1]
    print(f"  {i-1}->{i}: {diff_us}us = {diff_us/1000:.2f}ms")
