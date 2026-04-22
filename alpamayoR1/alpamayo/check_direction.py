import numpy as np
import pandas as pd

# 检查前几帧的数据
base = "/data01/vla/data/data_sample_chunk0/infer/46003675-4b4e-4c0f-ae54-3f7622bddf6a"

print("=== 检查第0帧 (infer_idx=0) ===")

# 加载预测
pred = np.load(f"{base}/result/pred_000000.npy")
print(f"预测轨迹 (pred_000000):")
print(f"  形状: {pred.shape}")
print(f"  起点: [{pred[0,0]:.4f}, {pred[0,1]:.4f}]")
print(f"  终点: [{pred[-1,0]:.4f}, {pred[-1,1]:.4f}]")
print(f"  X范围: [{pred[:,0].min():.3f}, {pred[:,0].max():.3f}]")
print(f"  Y范围: [{pred[:,1].min():.3f}, {pred[:,1].max():.3f}]")

# 加载真值
future = np.load(f"{base}/data/egomotion/ego_000000_future_gt.npy", allow_pickle=True).item()
gt = future['xyz'][:, :2]
print(f"\n真值轨迹 (future_gt):")
print(f"  形状: {gt.shape}")
print(f"  起点: [{gt[0,0]:.4f}, {gt[0,1]:.4f}]")
print(f"  终点: [{gt[-1,0]:.4f}, {gt[-1,1]:.4f}]")
print(f"  X范围: [{gt[:,0].min():.3f}, {gt[:,0].max():.3f}]")
print(f"  Y范围: [{gt[:,1].min():.3f}, {gt[:,1].max():.3f}]")

# 加载历史轨迹
hist = np.load(f"{base}/data/egomotion/ego_000000_history_local.npy", allow_pickle=True).item()
hist_xyz = hist['xyz']
print(f"\n历史轨迹 (history_local):")
print(f"  形状: {hist_xyz.shape}")
print(f"  前3点:\n{hist_xyz[:3]}")
print(f"  最后3点:\n{hist_xyz[-3:]}")
print(f"  X范围: [{hist_xyz[:,0].min():.3f}, {hist_xyz[:,0].max():.3f}]")
print(f"  Y范围: [{hist_xyz[:,1].min():.3f}, {hist_xyz[:,1].max():.3f}]")

# 计算ADE
diff = np.linalg.norm(pred - gt, axis=1)
print(f"\n误差分析:")
print(f"  每帧误差: {diff[:10]}")
print(f"  ADE: {diff.mean():.3f}m")

# 检查方向 - 车是向前走还是向右走？
print(f"\n=== 方向分析 ===")
print(f"预测X位移: {pred[-1,0] - pred[0,0]:.3f}m")
print(f"预测Y位移: {pred[-1,1] - pred[0,1]:.3f}m")
print(f"真值X位移: {gt[-1,0] - gt[0,0]:.3f}m")
print(f"真值Y位移: {gt[-1,1] - gt[0,1]:.3f}m")

# 从历史轨迹看车往哪个方向走
print(f"\n历史轨迹最后一点: {hist_xyz[-1]}")
print(f"历史轨迹位移: X={hist_xyz[-1,0] - hist_xyz[0,0]:.3f}, Y={hist_xyz[-1,1] - hist_xyz[0,1]:.3f}")
