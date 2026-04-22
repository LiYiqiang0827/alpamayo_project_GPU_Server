#!/usr/bin/env python3
import pandas as pd
import numpy as np

clip = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"
BASE_DIR = "/data01/vla/data/data_sample_chunk0"

ego_path = f"{BASE_DIR}/labels/egomotion/{clip}.egomotion.parquet"
ego_df = pd.read_parquet(ego_path)

# Frame 1903的t0
t0_ts = 19349601.0
TIME_STEP = 0.1

print("=== 验证未来数据插值 ===\n")
print(f"t0_timestamp: {t0_ts}")
print(f"时间步长: {TIME_STEP}s ({TIME_STEP*1000}ms)")
print()

# 计算前5个未来时间点
for i in range(1, 6):
    target_ts = t0_ts + i * int(TIME_STEP * 1e6)
    idx = (ego_df['timestamp'] - target_ts).abs().idxmin()
    actual_ts = ego_df.loc[idx, 'timestamp']
    xyz = ego_df.loc[idx, ['x', 'y', 'z']].values
    
    print(f"未来第{i}帧 (target={target_ts}):")
    print(f"  找到索引: {idx}")
    print(f"  实际时间戳: {actual_ts}")
    print(f"  时间差: {abs(actual_ts - target_ts)/1000:.2f}ms")
    print(f"  坐标: [{xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}]")
    print()

# 对比直接使用原始帧（假设10Hz就是每10帧取一个）
print("=== 对比：直接使用原始帧（每10帧）===")
t0_idx = 1950
for i in range(1, 6):
    raw_idx = t0_idx + i * 10  # 每10帧（100ms）
    if raw_idx < len(ego_df):
        raw_ts = ego_df.iloc[raw_idx]['timestamp']
        raw_xyz = ego_df.iloc[raw_idx][['x', 'y', 'z']].values
        print(f"未来第{i}帧 (索引{raw_idx}):")
        print(f"  时间戳: {raw_ts}")
        print(f"  与目标时间差: {abs(raw_ts - (t0_ts + i*100000))/1000:.2f}ms")
        print(f"  坐标: [{raw_xyz[0]:.2f}, {raw_xyz[1]:.2f}, {raw_xyz[2]:.2f}]")
        print()
