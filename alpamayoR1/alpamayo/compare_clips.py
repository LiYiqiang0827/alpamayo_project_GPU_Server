import pandas as pd
import numpy as np

# 对比两个clip的ego_df时间戳
clip1 = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
clip2 = "46003675-4b4e-4c0f-ae54-3f7622bddf6a"

for clip in [clip1, clip2]:
    print(f"\n{'='*60}")
    print(f"Clip: {clip}")
    print(f"{'='*60}")
    
    ego_df = pd.read_parquet(f"/data01/vla/data/data_sample_chunk0/labels/egomotion/{clip}.egomotion.parquet")
    
    print(f"总帧数: {len(ego_df)}")
    print(f"时间戳范围: [{ego_df['timestamp'].min()}, {ego_df['timestamp'].max()}]")
    print(f"前5个时间戳: {ego_df['timestamp'].head().values}")
    print(f"时间间隔: {np.diff(ego_df['timestamp'].head(10).values)}")
    
    # 检查索引表
    try:
        df_idx = pd.read_csv(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data/inference_index.csv")
        t0 = df_idx.iloc[0]['t0_timestamp']
        print(f"\n索引表t0_timestamp: {t0}")
        print(f"infer_idx=0 对应 ego_idx: {df_idx.iloc[0]['ego_idx']}")
    except:
        print("\n索引表不存在")
