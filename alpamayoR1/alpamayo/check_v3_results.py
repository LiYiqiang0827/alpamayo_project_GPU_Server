#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

clip = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"

# 检查v3结果
result_dir = f"/data01/vla/data/data_sample_chunk0/infer/{clip}/result"

# 读取几个结果文件
print("=== v3推理结果 (按10Hz采样的历史数据) ===\n")

for i in range(0, 40, 10):
    pred_file = f"{result_dir}/pred_{i*50:06d}.npy"
    if os.path.exists(pred_file):
        pred = np.load(pred_file, allow_pickle=True).item()
        ade = pred.get('ade', 'N/A')
        min_ade = pred.get('minADE', 'N/A')
        print(f"Frame {i*50}: minADE={min_ade:.2f}m" if isinstance(min_ade, (int, float)) else f"Frame {i*50}: minADE={min_ade}")

# 计算整体统计
ades = []
for f in os.listdir(result_dir):
    if f.startswith('pred_') and f.endswith('.npy'):
        pred = np.load(f"{result_dir}/{f}", allow_pickle=True).item()
        if 'minADE' in pred:
            ades.append(pred['minADE'])

if ades:
    print(f"\n统计: 均值={np.mean(ades):.2f}m, 中位数={np.median(ades):.2f}m, 最小={np.min(ades):.2f}m, 最大={np.max(ades):.2f}m")
