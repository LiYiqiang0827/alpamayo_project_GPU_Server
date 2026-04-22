import pandas as pd
import numpy as np

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"

# 旧结果
old_results = pd.read_csv(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/result/continuous_inference_results.csv")
row_138 = old_results[old_results['infer_idx'] == 138].iloc[0]
print("=== 旧数据 (infer_idx=138) ===")
print(f"t0_timestamp: {row_138['t0_timestamp']}")
print(f"ADE: {row_138['ade']}")

# 新数据 (infer_idx=0，但时间戳应该类似)
new_idx = pd.read_csv(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data/inference_index.csv")
print(f"\n=== 新数据 ===")
print(f"infer_idx=0: t0_timestamp={new_idx.iloc[0]['t0_timestamp']}")

# 找到新数据中时间戳最接近138的
closest = (new_idx['t0_timestamp'] - row_138['t0_timestamp']).abs().idxmin()
print(f"最接近138的新数据: infer_idx={new_idx.iloc[closest]['infer_idx']}, t0={new_idx.iloc[closest]['t0_timestamp']}")

# 对比egomotion
import numpy as np
old_pred = np.load(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/result/pred_000138.npy")
print(f"\n旧预测终点: {old_pred[-1]}")

# 新的egomotion
new_future = np.load(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data/egomotion/ego_000000_future_gt.npy", allow_pickle=True).item()
print(f"新真值终点: {new_future['xyz'][-1][:2]}")
