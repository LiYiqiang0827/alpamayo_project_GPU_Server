#!/usr/bin/env python3
import pandas as pd

clips = [
    "01d3588e-bca7-4a18-8e74-c6cfe9e996db",
    "46003675-4b4e-4c0f-ae54-3f7622bddf6a"
]

for clip in clips:
    ego_path = f"/data01/vla/data/data_sample_chunk0/labels/egomotion/{clip}.egomotion.parquet"
    ego_df = pd.read_parquet(ego_path)
    
    ts_diff = ego_df['timestamp'].diff().dropna()
    
    print(f"\n=== Clip: {clip[:8]}... ===")
    print(f"总帧数: {len(ego_df)}")
    print(f"时间差均值: {ts_diff.mean()/1000:.2f}ms")
    print(f"时间差中位数: {ts_diff.median()/1000:.2f}ms")
    print(f"频率(按中位数): {1e6/ts_diff.median():.0f}Hz")
    
    # 只看正常间隔（<100ms）
    normal_diff = ts_diff[ts_diff < 100000]
    print(f"正常间隔(<100ms)均值: {normal_diff.mean()/1000:.2f}ms")
    print(f"正常间隔数量: {len(normal_diff)}")
