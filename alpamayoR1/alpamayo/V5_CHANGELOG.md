# Alpamayo V5 版本 - 严格时间戳对齐预处理

**创建时间**: 2026-03-19  
**核心改进**: 时间戳精确对齐 + 逐点验证 + 坐标系转换修复

## 📁 脚本位置

| 脚本 | 路径 | 说明 |
|-----|------|-----|
| 预处理脚本 | `~/alpamayo/preprocess_strict.py` | 严格时间戳对齐预处理 |
| 推理脚本 | `~/alpamayo/run_inference_new_strict.py` | V5 版本推理（支持新索引格式） |
| 自动化脚本 | `~/alpamayo/auto_process.py` | 一键处理（需适配 V5） |

## 🎯 核心改动点

### 1. 时间戳对齐策略（重大改进）

**旧版 (V4)**:
- 按固定帧号间隔取 egomotion（每 10 帧 = 100ms）
- 假设 100Hz 数据严格均匀分布
- 问题：实际数据有 jitter，导致时间偏差累积

**新版 (V5)**:
- 对每个目标时间点（如 t-100ms, t-200ms...），**在全部 egomotion 中查找最接近的帧**
- 容差：±10ms（超过则标记该推理帧无效）
- 优势：不受 jitter 影响，每个点都精确对齐

### 2. 图像帧选择策略

**f3 (t=0, 当前时刻)**:
- 只查找 ≤ 目标时间的帧（**不能看未来**）
- 保证推理时不泄露未来信息

**f0-f2 (历史帧)**:
- 可以左右查找最接近的帧
- 容差：±33ms（约 1 帧，30Hz）

### 3. 边界处理

- 计算 4 个相机的**共同时间范围**
- 只保留在此范围内的 egomotion 帧
- 严格处理启动延迟差异

### 4. 坐标系转换修复（关键 Bug 修复）

**Bug**: 四元数列索引错误（把 timestamp 当成 qx）

**修复后**:
```python
# 正确的列索引
hist_xyz_world = history[:, 5:8]   # x, y, z (columns 5,6,7)
hist_quat = history[:, 1:5]        # qx, qy, qz, qw (columns 1,2,3,4)
```

### 5. 数据格式统一

**V5 预处理输出格式**:
```
data/
├── inference_index_strict.csv    # 索引文件
├── egomotion/
│   ├── frame_{id:06d}_history.npy      # (16, 11) - numpy array
│   └── frame_{id:06d}_future_gt.npy    # (64, 4) - numpy array
└── camera_images/
    ├── camera_cross_left_120fov/*.jpg
    ├── camera_front_wide_120fov/*.jpg
    ├── camera_cross_right_120fov/*.jpg
    └── camera_front_tele_30fov/*.jpg
```

## 📊 效果对比

| 指标 | V4 (旧方法) | V5 (新方法) | 提升 |
|-----|------------|------------|-----|
| **ADE 均值** | 4.6 m | **1.23 m** | **3.7x** |
| **有效帧数** | ~1800 (宽松) | **1120** (严格) | 质量优先 |
| **时间对齐精度** | 固定帧号 | **逐点 ±10ms** | 更准确 |
| **坐标系转换** | 有 Bug | **已修复** | 正确 |

## 🔧 使用方式

### 1. 预处理数据
```bash
python3 ~/alpamayo/preprocess_strict.py
# 输出: data/inference_index_strict.csv + egomotion/ + camera_images/
```

### 2. 运行推理
```bash
python3 ~/alpamayo/run_inference_new_strict.py \
    --clip 01d3588e-bca7-4a18-8e74-c6cfe9e996db \
    --num_frames 100 \
    --traj 1
```

### 3. 查看结果
```bash
cat /data01/vla/data/data_sample_chunk0/infer/{clip_id}/result_strict/inference_results_strict.csv
```

## ⚙️ 关键参数

| 参数 | 值 | 说明 |
|-----|---|------|
| `HISTORY_STEPS` | 16 | 1.6 秒历史 |
| `FUTURE_STEPS` | 64 | 6.4 秒未来 |
| `TIME_STEP` | 0.1s | 100ms 间隔 |
| `MAX_EGO_DIFF_MS` | 10ms | egomotion 时间容差 |
| `MAX_IMAGE_DIFF_MS` | 33ms | 图像帧时间容差 |
| `IMG_TIME_OFFSETS_MS` | [300,200,100,0] | f0,f1,f2,f3 相对偏移 |

## 📝 注意事项

1. **必须使用 V5 推理脚本**：旧脚本不支持新的 `inference_index_strict.csv` 格式
2. **坐标系转换已内置**：V5 推理脚本自动处理局部坐标系转换
3. **数据质量优先**：过滤条件较严格，确保每个推理帧都可靠

## 🔄 后续优化方向

- [ ] 自动化批量处理多个 clips
- [ ] 支持多 GPU 并行推理
- [ ] 可视化结果对比（V4 vs V5）
