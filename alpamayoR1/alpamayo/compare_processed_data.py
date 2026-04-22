#!/usr/bin/env python3
"""对比预处理后数据的差异"""
import numpy as np

clips = [
    ("01d3588e-bca7-4a18-8e74-c6cfe9e996db", "好的clip"),
    ("e9162b7f-00f1-4563-890e-15c9fc89ad7a", "直道clip")
]

print("=== 对比预处理后数据 ===\n")

for clip_id, desc in clips:
    DATA_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip_id}/data"
    
    # 读取第一个帧的历史和未来数据
    hist_local = np.load(f"{DATA_DIR}/egomotion/ego_000000_history_local.npy", allow_pickle=True).item()
    future_gt = np.load(f"{DATA_DIR}/egomotion/ego_000000_future_gt.npy", allow_pickle=True).item()
    
    hist_xyz = hist_local['xyz']
    future_xyz = future_gt['xyz']
    
    print(f"{desc} (Frame 0):")
    print(f"  历史位移: {np.linalg.norm(hist_xyz[-1][:2] - hist_xyz[0][:2]):.2f}m")
    print(f"  未来位移: {np.linalg.norm(future_xyz[-1][:2] - future_xyz[0][:2]):.2f}m")
    print(f"  未来最后坐标: [{future_xyz[-1][0]:.2f}, {future_xyz[-1][1]:.2f}]")
    print()
