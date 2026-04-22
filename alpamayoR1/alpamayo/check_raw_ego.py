#!/usr/bin/env python3
import pandas as pd

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
ego_path = f"/data01/vla/data/data_sample_chunk0/labels/egomotion/{clip}.egomotion.parquet"
ego_df = pd.read_parquet(ego_path)

print("=== 原始egomotion数据 ===")
print(f"总帧数: {len(ego_df)}")
print(f"\n前10行时间戳:")
for i in range(min(10, len(ego_df))):
    ts = ego_df['timestamp'].iloc[i]
    if i > 0:
        diff = (ts - ego_df['timestamp'].iloc[i-1]) / 1000  # ms
        print(f"  {i}: {ts} (diff: {diff:.2f}ms)")
    else:
        print(f"  {i}: {ts}")

# 计算频率
ts_diff = ego_df['timestamp'].diff().dropna()
print(f"\n时间差统计:")
print(f"  均值: {ts_diff.mean()/1000:.2f}ms = {1e6/ts_diff.mean():.1f}Hz")
print(f"  中位数: {ts_diff.median()/1000:.2f}ms")
print(f"  最小: {ts_diff.min()/1000:.2f}ms, 最大: {ts_diff.max()/1000:.2f}ms")

# 检查是否有间隔异常的地方
print(f"\n异常间隔 (>100ms) 数量: {(ts_diff > 100000).sum()}")
if (ts_diff > 100000).sum() > 0:
    large_gaps = ts_diff[ts_diff > 100000]
    print(f"  最大间隔: {large_gaps.max()/1000:.2f}ms = {large_gaps.max()/1e6:.2f}s")
