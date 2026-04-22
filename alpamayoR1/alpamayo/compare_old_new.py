import numpy as np

# 对比新旧预处理的数据
clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"

# 新数据 (v3)
hist_new = np.load(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data/egomotion/ego_000000_history_local.npy", allow_pickle=True).item()
future_new = np.load(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data/egomotion/ego_000000_future_gt.npy", allow_pickle=True).item()

print("=== 新数据 (v3 - 10Hz插值) ===")
print(f"历史轨迹 (16步):")
print(f"  第1点: {hist_new['xyz'][0]}")
print(f"  最后点: {hist_new['xyz'][-1]}")
print(f"  Y范围: [{hist_new['xyz'][:,1].min():.4f}, {hist_new['xyz'][:,1].max():.4f}]")

print(f"\n真值轨迹 (64步):")
print(f"  第1点: {future_new['xyz'][0]}")
print(f"  最后点: {future_new['xyz'][-1]}")

# 旧数据 (如果还存在)
import os
if os.path.exists(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/result/pred_000138.npy"):
    # 读取旧的历史数据（从之前的预处理）
    # 注意：旧数据已经被覆盖了，只能看预测结果对比
    pred_old = np.load(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/result/pred_000138.npy")
    print(f"\n=== 旧预测 (之前有0.8m ADE) ===")
    print(f"预测终点: {pred_old[-1]}")

# 检查历史轨迹的旋转矩阵
print(f"\n=== 旋转矩阵检查 ===")
print(f"历史第1点旋转矩阵:\n{hist_new['rotation_matrix'][0]}")
print(f"历史最后点旋转矩阵:\n{hist_new['rotation_matrix'][-1]}")
