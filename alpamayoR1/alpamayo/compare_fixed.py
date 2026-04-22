#!/usr/bin/env python3
import numpy as np

# 好的clip
clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
DATA_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data"
prefix = "ego_000138"
future_gt = np.load(f"{DATA_DIR}/egomotion/{prefix}_future_gt.npy", allow_pickle=True).item()
future_xyz = future_gt['xyz']

print("=== 好的clip (Frame 138, ADE=0.8m) ===")
print(f"未来位移: {np.linalg.norm(future_xyz[-1][:2] - future_xyz[0][:2]):.2f}m")
print(f"Y方向位移: {future_xyz[-1][1] - future_xyz[0][1]:.2f}m (直行)")

# 转弯clip (修复后)
clip2 = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"
DATA_DIR2 = f"/data01/vla/data/data_sample_chunk0/infer/{clip2}/data"
prefix2 = "ego_001000"
future_gt2 = np.load(f"{DATA_DIR2}/egomotion/{prefix2}_future_gt.npy", allow_pickle=True).item()
future_xyz2 = future_gt2['xyz']

print(f"\n=== 转弯clip修复后 (Frame 1000, ADE=33m) ===")
print(f"未来位移: {np.linalg.norm(future_xyz2[-1][:2] - future_xyz2[0][:2]):.2f}m")
print(f"Y方向位移: {future_xyz2[-1][1] - future_xyz2[0][1]:.2f}m (大转弯)")
