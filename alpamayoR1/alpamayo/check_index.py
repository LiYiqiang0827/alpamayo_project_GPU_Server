import numpy as np
import pandas as pd

# 检查索引对应
df = pd.read_csv("/data01/vla/data/data_sample_chunk0/infer/46003675-4b4e-4c0f-ae54-3f7622bddf6a/data/inference_index.csv")
print("=== 索引对应 ===")
print(df[["infer_idx", "ego_idx"]].head(10))

# infer_idx=0 对应 ego_idx=27
infer_idx = 0
ego_idx = df.iloc[infer_idx]["ego_idx"]
print(f"\ninfer_idx={infer_idx} 对应 ego_idx={ego_idx}")

# 加载pred_000000
pred = np.load(f"/data01/vla/data/data_sample_chunk0/infer/46003675-4b4e-4c0f-ae54-3f7622bddf6a/result/pred_{infer_idx:06d}.npy")
print(f"\n预测 (pred_{infer_idx:06d}) 起点: {pred[0]}")

# 加载对应的真值 - 文件名用infer_idx，但数据对应ego_idx
saved = np.load(f"/data01/vla/data/data_sample_chunk0/infer/46003675-4b4e-4c0f-ae54-3f7622bddf6a/data/egomotion/ego_{infer_idx:06d}_future_gt.npy", allow_pickle=True).item()
print(f"真值 (ego_{infer_idx:06d}) 起点: {saved['xyz'][0]}")

# 计算误差
diff = np.linalg.norm(pred - saved['xyz'][:, :2], axis=1)
print(f"ADE: {diff.mean():.3f}m")

# 现在检查infer_idx=27对应的ego_idx=?
ego_idx_27 = df.iloc[27]["ego_idx"]
print(f"\n\ninfer_idx=27 对应 ego_idx={ego_idx_27}")

# 如果文件名用infer_idx命名，那应该对得上
# 但之前的检查显示ego_000000对应的数据不是从0开始的

# 让我直接检查egomotion文件命名
import os
files = sorted([f for f in os.listdir("/data01/vla/data/data_sample_chunk0/infer/46003675-4b4e-4c0f-ae54-3f7622bddf6a/data/egomotion") if f.endswith("_future_gt.npy")])[:5]
print(f"\n前5个真值文件: {files}")
