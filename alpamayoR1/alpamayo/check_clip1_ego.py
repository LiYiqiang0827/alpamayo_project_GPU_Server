import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

clip1 = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"

ego_df = pd.read_parquet(f"/data01/vla/data/data_sample_chunk0/labels/egomotion/{clip1}.egomotion.parquet")

# Clip1的ego_000011对应ego_df的哪一行？
# 从索引表看，ego_file_prefix=ego_000011对应infer_idx=11
# 让我找ego_000011对应的时间戳

df_idx = pd.read_csv(f"/data01/vla/data/data_sample_chunk0/infer/{clip1}/data/inference_index.csv")
row = df_idx[df_idx['ego_file_prefix'] == 'ego_000011'].iloc[0]
t0_us = row['t0_timestamp']
print(f"Clip1: ego_00011 -> t0_timestamp={t0_us}")

# 在ego_df中找到对应行
matching = ego_df[ego_df['timestamp'] == t0_us]
if len(matching) > 0:
    ego_idx = matching.index[0]
    print(f"对应的 ego_idx: {ego_idx}")
    
    t0_row = ego_df.iloc[ego_idx]
    t0_xyz = t0_row[['x', 'y', 'z']].values
    t0_quat = t0_row[['qx', 'qy', 'qz', 'qw']].values
    
    print(f"t0世界坐标: {t0_xyz}")
    
    # 历史轨迹
    hist_world = ego_df.iloc[ego_idx-16:ego_idx][['x', 'y', 'z']].values
    print(f"历史世界坐标最后点: {hist_world[-1]}")
    
    # 转换
    t0_rot = R.from_quat(t0_quat).as_matrix()
    hist_local = (hist_world - t0_xyz) @ t0_rot.T
    print(f"历史局部坐标最后点: {hist_local[-1]}")
else:
    print("找不到匹配的时间戳！")
    print(f"ego_df timestamp范围: [{ego_df['timestamp'].min()}, {ego_df['timestamp'].max()}]")
