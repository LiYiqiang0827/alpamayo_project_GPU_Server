import numpy as np

# 检查模型输出和真值的范围
base = "/data01/vla/data/data_sample_chunk0/infer/46003675-4b4e-4c0f-ae54-3f7622bddf6a"

# 加载第一帧的预测和真值
pred = np.load(f"{base}/result/pred_000000.npy")
future = np.load(f"{base}/data/egomotion/ego_000000_future_gt.npy", allow_pickle=True).item()

print("=== 预测轨迹 (模型输出) ===")
print(f"形状: {pred.shape}")
print(f"前5点:\n{pred[:5]}")
print(f"最后5点:\n{pred[-5:]}")
print(f"X范围: [{pred[:,0].min():.3f}, {pred[:,0].max():.3f}]")
print(f"Y范围: [{pred[:,1].min():.3f}, {pred[:,1].max():.3f}]")

print("\n=== 真值轨迹 (future_gt) ===")
gt = future["xyz"]
print(f"形状: {gt.shape}")
print(f"前5点:\n{gt[:5]}")
print(f"最后5点:\n{gt[-5:]}")
print(f"X范围: [{gt[:,0].min():.3f}, {gt[:,0].max():.3f}]")
print(f"Y范围: [{gt[:,1].min():.3f}, {gt[:,1].max():.3f}]")

print("\n=== 误差分析 ===")
diff = np.linalg.norm(pred - gt[:, :2], axis=1)
print(f"每帧误差: {diff[:10]}")
print(f"ADE: {diff.mean():.3f}m")
