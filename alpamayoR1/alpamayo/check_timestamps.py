#!/usr/bin/env python3
import numpy as np

HISTORY_STEPS = 16
TIME_STEP = 0.1

print("=== 历史时间戳计算 ===\n")

hist_offsets = np.arange(-(HISTORY_STEPS-1) * int(TIME_STEP * 1e6), 
                          int(TIME_STEP * 1e6 / 2), 
                          int(TIME_STEP * 1e6)).astype(np.int64)

print(f"HISTORY_STEPS = {HISTORY_STEPS}")
print(f"TIME_STEP = {TIME_STEP}s = {int(TIME_STEP * 1e6)}us")
print(f"\n历史offsets (us):")
print(hist_offsets)
print(f"\n历史offsets (s):")
print(hist_offsets / 1e6)
print(f"\n共 {len(hist_offsets)} 个点")
print(f"时间范围: {hist_offsets[0]/1e6:.1f}s ~ {hist_offsets[-1]/1e6:.1f}s (相对于t0)")

# 正确的计算应该是
correct_offsets = np.arange(-(HISTORY_STEPS-1) * int(TIME_STEP * 1e6), 
                             int(TIME_STEP * 1e6),  # 包含t0
                             int(TIME_STEP * 1e6)).astype(np.int64)

print(f"\n=== 正确的offsets (包含t0) ===")
print(correct_offsets / 1e6)
print(f"共 {len(correct_offsets)} 个点")
