#!/usr/bin/env python3
import pandas as pd
import numpy as np

clip = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"
BASE_DIR = "/data01/vla/data/data_sample_chunk0"

ego_path = f"{BASE_DIR}/labels/egomotion/{clip}.egomotion.parquet"
ego_df = pd.read_parquet(ego_path)

print(f"总帧数: {len(ego_df)}")
print(f"Frame 1903对应的t0_idx: 1950")
print(f"需要未来64帧，到索引: 1950+64=2014")
print(f"实际最大索引: {len(ego_df)-1}")
print(f"是否足够: {2014 < len(ego_df)}")

# 检查实际的未来数据
print(f"\n实际原始未来数据 (索引1950-2014):")
future_world = ego_df.iloc[1950:2014][['x', 'y', 'z']].values
print(f"  第一帧: [{future_world[0][0]:.2f}, {future_world[0][1]:.2f}]")
print(f"  最后一帧: [{future_world[-1][0]:.2f}, {future_world[-1][1]:.2f}]")
displacement = ((future_world[-1][0]-future_world[0][0])**2 + (future_world[-1][1]-future_world[0][1])**2)**0.5
print(f"  总位移: {displacement:.2f}m")

# 检查预处理后的数据
data_dir = f"{BASE_DIR}/infer/{clip}/data"
future_processed = np.load(f"{data_dir}/egomotion/ego_001903_future_gt.npy", allow_pickle=True).item()['xyz']
print(f"\n预处理后未来数据:")
print(f"  第一帧: [{future_processed[0][0]:.2f}, {future_processed[0][1]:.2f}]")
print(f"  最后一帧: [{future_processed[-1][0]:.2f}, {future_processed[-1][1]:.2f}]")
disp2 = ((future_processed[-1][0]-future_processed[0][0])**2 + (future_processed[-1][1]-future_processed[0][1])**2)**0.5
print(f"  总位移: {disp2:.2f}m")

print(f"\n差异分析:")
print(f"  原始最后一帧 - 预处理最后一帧: [{future_world[-1][0]-future_processed[-1][0]:.2f}, {future_world[-1][1]-future_processed[-1][1]:.2f}]")
