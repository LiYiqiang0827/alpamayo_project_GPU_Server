import numpy as np

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"

# 检查新生成的数据
future = np.load(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data/egomotion/ego_000000_future_gt.npy", allow_pickle=True).item()
hist = np.load(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data/egomotion/ego_000000_history_local.npy", allow_pickle=True).item()

print("=== Clip1 (重新预处理后) ===")
print(f"历史最后点: {hist['xyz'][-1]}")
print(f"真值起点: {future['xyz'][0]}")
print(f"真值终点: {future['xyz'][-1]}")
print(f"真值X位移: {future['xyz'][-1,0] - future['xyz'][0,0]:.3f}m")
print(f"真值Y位移: {future['xyz'][-1,1] - future['xyz'][0,1]:.3f}m")
