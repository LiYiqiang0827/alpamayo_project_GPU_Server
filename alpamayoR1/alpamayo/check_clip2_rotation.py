#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

clip = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"
DATA_DIR = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data"

print("=== 检查t0时刻的车辆朝向 ===\n")

# 读取索引
index_df = pd.read_csv(f"{DATA_DIR}/inference_index.csv")

# 读取原始egomotion
ego_path = f"/data01/vla/data/data_sample_chunk0/labels/egomotion/{clip}.egomotion.parquet"
ego_df = pd.read_parquet(ego_path)

for infer_idx in [0, 100, 200]:
    ego_idx = index_df.iloc[infer_idx]['ego_idx']
    t0_row = ego_df.iloc[ego_idx]
    
    t0_xyz = t0_row[['x', 'y', 'z']].values
    t0_quat = t0_row[['qx', 'qy', 'qz', 'qw']].values
    t0_rot = R.from_quat(t0_quat).as_matrix()
    
    # 旋转矩阵的列向量表示局部坐标轴在世界坐标系中的方向
    # 第一列: 局部X轴在世界坐标系中的方向
    # 第二列: 局部Y轴（前进方向）在世界坐标系中的方向
    # 第三列: 局部Z轴在世界坐标系中的方向
    
    print(f"Frame {infer_idx} (ego_idx={ego_idx}):")
    print(f"  t0世界坐标: [{t0_xyz[0]:.2f}, {t0_xyz[1]:.2f}, {t0_xyz[2]:.2f}]")
    print(f"  局部X轴方向: [{t0_rot[0,0]:.2f}, {t0_rot[1,0]:.2f}, {t0_rot[2,0]:.2f}]")
    print(f"  局部Y轴方向(前进): [{t0_rot[0,1]:.2f}, {t0_rot[1,1]:.2f}, {t0_rot[2,1]:.2f}]")
    print(f"  局部Z轴方向: [{t0_rot[0,2]:.2f}, {t0_rot[1,2]:.2f}, {t0_rot[2,2]:.2f}]")
    
    # 计算偏航角
    yaw = np.arctan2(t0_rot[1,0], t0_rot[0,0]) * 180 / np.pi
    print(f"  偏航角(yaw): {yaw:.1f}度 (0=东, 90=北, 180=西, -90=南)")
    print()
