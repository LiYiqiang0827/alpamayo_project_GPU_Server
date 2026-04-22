#!/usr/bin/env python3
"""详细对比两个相似clips的差异"""
import pandas as pd
import numpy as np

BASE_DIR = "/data01/vla/data/data_sample_chunk0"

clips = [
    ("01d3588e-bca7-4a18-8e74-c6cfe9e996db", "参考clip (ADE=0.8m)"),
    ("b9b4ddc7-feb0-4749-9d14-931a70fe3e17", "相似clip (ADE=4.6m)")
]

print("=== 详细对比两个clips ===\n")

for clip_id, desc in clips:
    DATA_DIR = f"{BASE_DIR}/infer/{clip_id}/data"
    
    # 读取高质量索引
    hq_df = pd.read_csv(f"{DATA_DIR}/inference_index_high_quality.csv")
    
    # 读取第一个帧的历史和未来数据
    hist_local = np.load(f"{DATA_DIR}/egomotion/ego_000100_history_local.npy", allow_pickle=True).item()
    future_gt = np.load(f"{DATA_DIR}/egomotion/ego_000100_future_gt.npy", allow_pickle=True).item()
    
    hist_xyz = hist_local['xyz']
    future_xyz = future_gt['xyz']
    
    print(f"{desc} (Frame 100):")
    print(f"  历史位移: {np.linalg.norm(hist_xyz[-1][:2] - hist_xyz[0][:2]):.2f}m")
    print(f"  未来位移: {np.linalg.norm(future_xyz[-1][:2] - future_xyz[0][:2]):.2f}m")
    print(f"  未来最后坐标: [{future_xyz[-1][0]:.2f}, {future_xyz[-1][1]:.2f}]")
    print(f"  高质量索引总数: {len(hq_df)}")
    
    # 检查时间对齐质量
    print(f"  时间对齐: max_diff={hq_df['max_image_diff_ms'].max():.1f}ms, 平均={hq_df['max_image_diff_ms'].mean():.1f}ms")
    print()
