import numpy as np

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"

hist = np.load(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data/egomotion/ego_000000_history_local.npy", allow_pickle=True).item()
future = np.load(f"/data01/vla/data/data_sample_chunk0/infer/{clip}/data/egomotion/ego_000000_future_gt.npy", allow_pickle=True).item()

print("=== Clip1 (v3预处理后) ===")
print(f"历史轨迹 (16步, 1.6s):")
print(f"  第1点 (t0-1.6s): {hist['xyz'][0]}")
print(f"  最后点 (t0-0.1s): {hist['xyz'][-1]}")
print(f"  X位移: {hist['xyz'][-1,0] - hist['xyz'][0,0]:.3f}m")
print(f"  Y位移: {hist['xyz'][-1,1] - hist['xyz'][0,1]:.3f}m")

print(f"\n真值轨迹 (64步, 6.4s):")
print(f"  第1点 (t0+0.1s): {future['xyz'][0]}")
print(f"  最后点 (t0+6.4s): {future['xyz'][-1]}")
print(f"  X位移: {future['xyz'][-1,0] - future['xyz'][0,0]:.3f}m")
print(f"  Y位移: {future['xyz'][-1,1] - future['xyz'][0,1]:.3f}m")

# 历史最后点应该在原点后方（Y负方向）
print(f"\n=== 验证 ===")
print(f"历史最后点 Y坐标: {hist['xyz'][-1,1]:.4f}")
print(f"真值第1点 Y坐标: {future['xyz'][0,1]:.4f}")
print(f"车在向前走吗？历史Y < 真值Y? {hist['xyz'][-1,1] < future['xyz'][0,1]}")

# 计算ADE（模拟，因为没有预测）
print(f"\n历史轨迹长度: {len(hist['xyz'])} 步")
print(f"真值轨迹长度: {len(future['xyz'])} 步")
