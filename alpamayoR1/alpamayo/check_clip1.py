import pandas as pd

clip1 = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"

df = pd.read_csv(f"/data01/vla/data/data_sample_chunk0/infer/{clip1}/data/inference_index.csv")
print("Clip1 索引表前5行:")
print(df[['infer_idx', 't0_timestamp', 'ego_file_prefix']].head())

# 检查infer_idx=11对应哪个ego文件
print(f"\ninfer_idx=11 -> ego_file_prefix: {df.iloc[0]['ego_file_prefix']}")

# 读取对应的egomotion
import numpy as np
hist = np.load(f"/data01/vla/data/data_sample_chunk0/infer/{clip1}/data/egomotion/ego_000011_history_local.npy", allow_pickle=True).item()
print(f"\nego_000011_history_local 最后点: {hist['xyz'][-1]}")

future = np.load(f"/data01/vla/data/data_sample_chunk0/infer/{clip1}/data/egomotion/ego_000011_future_gt.npy", allow_pickle=True).item()
print(f"ego_000011_future_gt 起点: {future['xyz'][0]}")
