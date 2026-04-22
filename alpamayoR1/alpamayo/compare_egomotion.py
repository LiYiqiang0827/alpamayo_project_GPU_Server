import pandas as pd
import numpy as np

# 对比两个clip的索引表
clip1 = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"  # 之前成功的
clip2 = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"  # 转弯场景

for clip in [clip1, clip2]:
    print(f"\n{'='*60}")
    print(f"Clip: {clip}")
    print(f"{'='*60}")
    
    # 读取索引表
    df_idx = pd.read_csv(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data/inference_index.csv")
    print(f"索引表行数: {len(df_idx)}")
    print(f"\n前3行:")
    print(df_idx.head(3))
    print(f"\n列名: {df_idx.columns.tolist()}")
    
    # 检查egomotion文件
    import os
    ego_files = sorted([f for f in os.listdir(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data/egomotion") if f.endswith('_future_gt.npy')])
    print(f"\n真值文件数: {len(ego_files)}")
    
    # 检查第一帧的真值
    future = np.load(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data/egomotion/{ego_files[0]}", allow_pickle=True).item()
    gt = future['xyz']
    print(f"\n第一帧真值 ({ego_files[0]}):")
    print(f"  起点: {gt[0]}")
    print(f"  终点: {gt[-1]}")
    print(f"  X位移: {gt[-1,0] - gt[0,0]:.3f}m")
    print(f"  Y位移: {gt[-1,1] - gt[0,1]:.3f}m")
    
    # 检查历史轨迹
    hist_file = ego_files[0].replace('future_gt', 'history_local')
    hist = np.load(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data/egomotion/{hist_file}", allow_pickle=True).item()
    hist_xyz = hist['xyz']
    print(f"\n第一帧历史轨迹 ({hist_file}):")
    print(f"  第1点: {hist_xyz[0]}")
    print(f"  最后点: {hist_xyz[-1]}")
    print(f"  X位移: {hist_xyz[-1,0] - hist_xyz[0,0]:.3f}m")
    print(f"  Y位移: {hist_xyz[-1,1] - hist_xyz[0,1]:.3f}m")
