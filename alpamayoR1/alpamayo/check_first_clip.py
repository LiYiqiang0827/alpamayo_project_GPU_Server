import numpy as np
base = "/data01/vla/data/data_sample_chunk0/infer/01d3588e-bca7-4a18-8e74-c6cfe9e996db"
pred = np.load(f"{base}/result/pred_000138.npy")
future = np.load(f"{base}/data/egomotion/ego_000138_future_gt.npy", allow_pickle=True).item()
print("=== 第一个clip (之前成功的) ===")
print(f"预测前5点:\n{pred[:5]}")
print(f"预测X范围: [{pred[:,0].min():.3f}, {pred[:,0].max():.3f}]")
print(f"预测Y范围: [{pred[:,1].min():.3f}, {pred[:,1].max():.3f}]")
gt = future["xyz"]
print(f"\n真值前5点:\n{gt[:5]}")
print(f"真值X范围: [{gt[:,0].min():.3f}, {gt[:,0].max():.3f}]")
print(f"真值Y范围: [{gt[:,1].min():.3f}, {gt[:,1].max():.3f}]")
diff = np.linalg.norm(pred - gt[:, :2], axis=1)
print(f"ADE: {diff.mean():.3f}m")
