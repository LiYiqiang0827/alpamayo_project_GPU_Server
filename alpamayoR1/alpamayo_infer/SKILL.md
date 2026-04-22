# Alpamayo-R1 Skill

NVIDIA Alpamayo-R1 自动驾驶视觉-语言-动作模型项目支持。

## 项目概述

**Alpamayo-R1** 是 NVIDIA 开发的 10B 参数自动驾驶推理模型，结合视觉、语言和动作预测，输出带因果推理链的轨迹预测。

### 核心能力
- **多模态输入**: 4个摄像头视频流 + 自车运动历史
- **轨迹预测**: 6.4秒未来轨迹 (64个路点 @ 10Hz)
- **因果推理**: 生成 Chain-of-Causation (CoC) 推理链
- **模型架构**: 视觉-语言-动作 (VLA) 模型

## 路径配置

| 类型 | 路径 | 说明 |
|------|------|------|
| **项目代码** | `~/mikelee/alpamayo-main/` | 主代码库 |
| **模型权重** | `/data01/vla/models--nvidia--Alpamayo-R1-10B/` | 21GB SafeTensors格式 |
| **训练数据** | `/data01/vla/data/data_sample_chunk0/` | 摄像头+ego数据 |
| **标定数据** | `/data01/vla/data/calibration/` | 相机标定参数 |
| **推理脚本** | `~/mikelee/alpamayo-main/notebooks/inference.ipynb` | Jupyter演示 |
| **数据加载器** | `~/mikelee/alpamayo-main/src/alpamayo_r1/load_physical_aiavdataset.py` | 数据预处理 |
| **批量预处理** | `~/mikelee/alpamayo-main/src/alpamayo_r1/preprocess_inference_data_gpu.py` | GPU加速数据预处理 |
| **数据验证** | `~/mikelee/alpamayo-main/src/alpamayo_r1/verify_inference_data.py` | 验证预处理数据完整性 |
| **推理工作目录** | `~/alpamayo/` | 本地脚本和记录 |
| **预处理数据** | `/data01/vla/data/data_sample_chunk0/infer/{clip_id}/` | 批量推理数据目录 |

## 数据结构

### 摄像头数据 (30Hz)
```
data_sample_chunk0/camera/
├── {clip_id}.camera_cross_left_120fov.{mp4|timestamps.parquet|blurred_boxes.parquet}
├── {clip_id}.camera_front_wide_120fov.{...}
├── {clip_id}.camera_cross_right_120fov.{...}
└── {clip_id}.camera_front_tele_30fov.{...}
```

### Ego Motion 数据 (10Hz)
```
data_sample_chunk0/labels/egomotion/
└── {clip_id}.egomotion.parquet
    - timestamp, qx/qy/qz/qw (四元数)
    - x/y/z (位置), vx/vy/vz (速度)
    - ax/ay/az (加速度), curvature (曲率)
```

### 模型输入格式
```python
{
    "image_frames": (4, 4, 3, H, W),        # 4摄像头 × 4帧
    "ego_history_xyz": (1, 1, 16, 3),       # 16步历史位置
    "ego_history_rot": (1, 1, 16, 3, 3),    # 16步旋转矩阵
    "ego_future_xyz": (1, 1, 64, 3),        # 64步未来位置 (真值)
    "ego_future_rot": (1, 1, 64, 3, 3),     # 64步未来旋转 (真值)
}
```

## 关键概念

### 坐标系转换
模型要求输入在 **t0时刻自车局部坐标系**：
```
xyz_local = R_t0^{-1} @ (xyz_world - t0_xyz)
rot_local = R_t0^{-1} @ R_world
```

### 时间对齐
- **摄像头**: 30Hz，取最近4帧 (t0-0.3s 到 t0)
- **Ego**: 10Hz，历史16步 (t0-1.5s 到 t0)，未来64步 (t0+0.1s 到 t0+6.4s)

## 常用命令

### 连接GPU服务器
```bash
ssh gpu-server
```

### 查看数据
```bash
# 数据大小
du -sh /data01/vla/data/*

# 模型权重
ls -la /data01/vla/models--nvidia--Alpamayo-R1-10B/blobs/
```

### 运行推理
```python
cd ~/mikelee/alpamayo-main
jupyter notebook notebooks/inference.ipynb
```

## 依赖

- PyTorch + CUDA
- `physical_ai_av` 数据集接口
- `scipy` (旋转矩阵计算)
- `einops` (张量重排)
- `mediapy` (可视化)

## 版本更新记录

### V5 版本 (2026-03-19) - 严格时间戳对齐

**核心改进**: ADE 从 4.6m → **1.23m** (3.7倍提升)

**关键改动**:
1. **逐点时间戳对齐** - 每个 100ms 目标点独立查找最接近帧（±10ms 容差）
2. **修复坐标系 Bug** - 四元数列索引错误（timestamp 列误当 qx）
3. **严格边界处理** - 4 相机共同时间范围 + 未来信息保护
4. **图像帧策略** - f3 只向后查找（不能看未来）

**新脚本**:
| 脚本 | 路径 | 说明 |
|-----|------|-----|
| 预处理 | `~/alpamayo_infer/preprocess_strict.py` | 严格时间戳对齐预处理 |
| 推理 | `~/alpamayo_infer/run_inference_new_strict.py` | V5 版本推理 (支持 `--step`) |
| **多GPU推理** | `~/alpamayo_infer/run_inference_multi_gpu.py` | **8卡并行推理，自动显存分配，崩溃重启3次** |
| 组合帧 | `~/alpamayo_infer/batch_create_combined_v5.py` | 生成可视化组合帧 |
| 视频 | `~/alpamayo_infer/step4_create_video_v5.py` | 生成MP4视频 |
| 一键脚本 | `~/alpamayo_infer/auto_process.py` | 完整自动化流程 |
| 改动记录 | `~/alpamayo_infer/V5_CHANGELOG.md` | 详细改动说明 |

**数据格式变更**:
- **索引**: `inference_index_strict.csv` (1054+ 有效帧)
- **Egomotion**: `frame_{id:06d}_history.npy` (16步, numpy array)
- **未来真值**: `frame_{id:06d}_future_gt.npy` (64步, numpy array)
- **图像**: `camera_images/{camera}/{idx:06d}.jpg`

**使用方法**:

```bash
# === 方式1: 一键自动化 (推荐，默认并行推理) ===
# 默认使用并行推理 (多GPU)，不生成视频
python3 ~/alpamayo_infer/auto_process.py \
    --clip 054da32b-9f3d-4074-93ab-044036b679f8 \
    --num_frames 100 --step 5 --traj 1

# 并行推理 + 生成视频
python3 ~/alpamayo_infer/auto_process.py \
    --clip 054da32b-9f3d-4074-93ab-044036b679f8 \
    --num_frames 100 --step 5 --traj 1 \
    --video

# 单卡推理 (禁用并行)
python3 ~/alpamayo_infer/auto_process.py \
    --clip 054da32b-9f3d-4074-93ab-044036b679f8 \
    --no-parallel \
    --num_frames 100 --step 5 --traj 1

# 一键脚本参数
#   --clip           Clip ID (必需)
#   --chunk          Chunk号 (默认0)
#   --num_frames     推理帧数 (默认20)
#   --traj           轨迹数 1/3/6 (默认1)
#   --step           采样步长，每N帧推理一帧 (默认1)
#   --video          生成视频 (默认False)
#   --parallel       使用并行推理 (默认True)
#   --no-parallel    禁用并行推理，使用单卡
#   --skip-preprocess 跳过预处理（如果已预处理过）
#   --no-download    不下载视频到本地

# === 方式2: 多GPU并行推理 (独立使用) ===
# 自动检测GPU，按显存分配实例（35G+/70G+），崩溃自动重启3次
# 输出与非并行化版本一致：result_strict/inference_results_strict.csv
python3 ~/alpamayo_infer/run_inference_multi_gpu.py \
    --clip 054da32b-9f3d-4074-93ab-044036b679f8 \
    --num_frames 1000 \
    --step 1 \
    --traj 1

# === 方式3: 分步执行 ===
# 预处理
~/miniconda3/bin/python ~/alpamayo_infer/preprocess_strict.py --clip <clip_id>

# 单卡推理
source ~/mikelee/alpamayo-main/.venv/bin/activate
python3 ~/alpamayo_infer/run_inference_new_strict.py \
    --clip 01d3588e-bca7-4a18-8e74-c6cfe9e996db \
    --num_frames 100 --step 5 --traj 1

# 生成组合帧
~/miniconda3/bin/python3 ~/alpamayo_infer/batch_create_combined_v5.py --clip <clip_id>

# 生成视频
~/miniconda3/bin/python3 ~/alpamayo_infer/step4_create_video_v5.py --clip <clip_id>
```

**关键参数**:
- `MAX_EGO_DIFF_MS = 10` - egomotion 时间容差
- `MAX_IMAGE_DIFF_MS = 33` - 图像帧时间容差

---

## 批量推理数据预处理

### 预处理脚本

**`preprocess_inference_data_gpu.py`** - GPU加速数据预处理

```bash
# 运行预处理 (GPU服务器)
cd ~/mikelee/alpamayo-main/src/alpamayo_r1
python3 preprocess_inference_data_gpu.py
```

**功能**:
- GPU硬件解码视频 (h264_cuvid)，比CPU快~100倍
- 生成JPG格式图片 (节省72%空间 vs PNG)
- 提取并转换egomotion数据 (世界坐标 → t0局部坐标)
- 生成统一索引表 `inference_index.csv`

**预处理流程**:
1. **Step 1**: 建立时间对齐索引 (`infer_data_index.csv`)
   - 以egomotion为主轴，对齐4个相机帧
   - 筛选时间差 ≤50ms 的有效帧

2. **Step 2**: GPU解码与数据准备
   - 4相机并行GPU解码
   - 生成JPG图片到 `camera_images/`
   - 提取egomotion到 `egomotion/`
   - 生成索引表 `inference_index.csv`

### 预处理数据目录结构

```
/data01/vla/data/data_sample_chunk0/infer/{clip_id}/
├── data/
│   ├── camera_images/              # 4相机 × 605帧 JPG
│   │   ├── camera_cross_left_120fov/000000.jpg
│   │   ├── camera_front_wide_120fov/
│   │   ├── camera_cross_right_120fov/
│   │   └── camera_front_tele_30fov/
│   ├── egomotion/                  # 推理帧轨迹数据
│   │   ├── ego_000011_history_world.npy   # (16,3) 世界坐标
│   │   ├── ego_000011_history_local.npy   # (16,3) t0局部坐标
│   │   ├── ego_000011_future_gt.npy       # (64,3) 未来真值
│   │   └── ego_000011_t0.json             # t0时刻信息
│   ├── inference_index.csv         # 核心索引表
│   └── infer_data_index.csv        # 原始时间戳索引
└── result/                         # 推理结果
```

### 数据验证

```bash
# 验证预处理数据完整性
python3 verify_inference_data.py
```

**验证内容**:
- 索引表存在性和格式
- 16张图片/帧存在性和可读性
- 4个egomotion文件/帧存在性和格式
- 数据形状验证 (历史16步, 未来64步)

### 性能对比

| 指标 | 旧版 (PNG+CPU) | 新版 (JPG+GPU) | 提升 |
|------|---------------|---------------|------|
| 视频解码 | ~10-15 fps | ~1500 fps | **~100x** |
| 图片大小 | ~3.0 GB | 843 MB | **节省72%** |
| 总处理时间 | ~15-20分钟 | ~2分钟 | **~10x** |

### 使用预处理数据

```python
import pandas as pd
import numpy as np
from PIL import Image

# 加载索引表
df = pd.read_csv('inference_index.csv')
row = df.iloc[0]

# 加载图片 (16张)
img_left_f0 = Image.open(f"data/{row['cam_left_f0']}")
# ...

# 加载egomotion
prefix = row['ego_file_prefix']
history = np.load(f"egomotion/{prefix}_history_local.npy", allow_pickle=True).item()
future = np.load(f"egomotion/{prefix}_future_gt.npy", allow_pickle=True).item()

# 数据格式
# history['xyz']: (16, 3) - 16步历史位置
# history['rotation_matrix']: (16, 3, 3) - 旋转矩阵
# future['xyz']: (64, 3) - 64步未来真值
```

## 完整批量推理流程

> 详细步骤见: `knowledge/alpamayo-r1-workflow.md`

### 快速开始 (4步完成单个clip批量推理)

#### Step 0: 创建目录结构
```bash
mkdir -p /data01/vla/data/data_sample_chunk0/infer/{clip_id}/{data,result}
```

#### Step 1: 生成高质量索引
**脚本**: `~/alpamayo/preprocess_high_quality.py`
```bash
python3 preprocess_high_quality.py
```
**功能**:
- GPU硬件解码4个相机视频 → `camera_images/`
- 提取并转换egomotion → `egomotion/`
- 时间对齐 + 自动过滤边界帧 → `inference_index_high_quality.csv`

**过滤效果**:
- 高质量帧: ~1866/2010 (92.8%)
- 图像帧时间差: <17ms

#### Step 2: 批量推理
**脚本**: `~/alpamayo/continuous_inference.py`
```bash
python3 continuous_inference.py
```
**配置**:
```python
NUM_TRAJ_SAMPLES = 1  # 1/3/6条轨迹
STEP = 5              # 每5帧推理一次
```
**输出**:
- `continuous_inference_results.csv` - CoC + ADE + 推理时间
- `pred_*.npy` - 轨迹预测

**性能参考**:
| 轨迹数 | GPU时间 | minADE |
|--------|---------|--------|
| 1条 | ~1.1s | ~2-3m |
| 3条 | ~2.1s | ~1.5m |
| 6条 | ~3.5s | ~0.7m |

#### Step 3: 批量创建组合帧 (JPG格式)
**脚本**: `~/alpamayo/batch_create_combined_v2.py`
```bash
python3 batch_create_combined_v2.py --clip <clip_id> [--chunk 0]
```
**布局**:
```
┌─────────────────────────────────────────┐
│         16宫格 (4相机 × 4帧)            │  1280×720
├────────────────────────┬────────────────┤
│   CoC + ADE 信息       │    轨迹图      │
│   左下 740×1080        │    右下        │
│   文字更大！           │    540×1080    │
│                        │   (宽高比1:2)  │
└────────────────────────┴────────────────┘
      总尺寸: 1280×1800
```
**坐标轴**: X-15~15m, Y0~60m

**输出格式**: **JPG (quality=90)** - 比 PNG 节省 ~70% 空间
- 单帧大小: ~250 KB (JPG) vs ~800 KB (PNG)
- 374帧总大小: ~99 MB vs ~300 MB

**输出目录**: `{result_dir}/combined_frames_v2/`

#### Step 4: 生成组合视频 (MP4)
**脚本**: `~/alpamayo/step4_create_video_from_frames.py`
```bash
python3 step4_create_video_from_frames.py --clip <clip_id> [--chunk 0]
```
**技术参数**:
- 编码器: libx264 (H.264)
- 帧率: 10 FPS
- 分辨率: 1280×1800
- 质量: CRF=23

**压缩效果**: 
| 项目 | 大小 | 压缩比 |
|------|------|--------|
| 源帧 (JPG, 374帧) | ~99 MB | - |
| 输出视频 | ~11 MB | **9x** |

**输出文件**: `{result_dir}/combined_video_{clip_id}.mp4`

#### 打包传回本地 (可选)
```bash
# 下载视频到本地
scp gpu-server:/data01/vla/data/data_sample_chunk0/infer/{clip_id}/result/combined_video_*.mp4 ~/alpamayo/
```

### 完整流程文档
- **详细步骤**: `knowledge/alpamayo-r1-workflow.md`
- **文件夹结构**: 见上文"预处理数据目录结构"
- **进阶用法**: 轨迹数量对比、时间对齐优化、边界帧过滤

---

## 多GPU并行推理

### 并行化改进总结

**核心改进**:
1. **自动GPU检测与分配** - 动态检测可用GPU，显存≥70G分配2实例，≥35G分配1实例
2. **进程级隔离** - 每个worker独立进程，通过 `CUDA_VISIBLE_DEVICES` 绑定GPU
3. **统一输出格式** - 并行/单卡推理输出完全一致，`result_strict/inference_results_strict.csv`
4. **崩溃自动重启** - 每个worker崩溃后自动重启，最多3次

**性能提升**:
| 场景 | 单卡时间 | 并行时间 | 加速比 |
|------|---------|---------|--------|
| 200帧 | ~400秒 | ~76秒 | 5.3x |
| 1120帧 | ~40分钟 | ~6分钟 | 6.7x |

**脚本**: `~/alpamayo_infer/run_inference_multi_gpu.py`

---

## ⚠️ 重要：本地模型权重使用规范

**必须使用本地缓存模型，禁止从HuggingFace下载**

### 正确做法

```python
# 必须在任何transformers导入之前设置
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 然后导入模型相关库
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# 使用本地模型路径
MODEL_PATH = "/data01/vla/models--nvidia--Alpamayo-R1-10B/snapshots/22fab1399111f50b52bfbe5d8b809f39bd4c2fe1"
model = AlpamayoR1.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")
```

### 多进程推理脚本模板

```python
def inference_worker_entry(args_dict):
    """Worker入口 - 必须在开头设置GPU和离线模式"""
    # 每个进程都要设置
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_dict['gpu_id'])
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # 现在才导入模型库
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    
    model = AlpamayoR1.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")
    # ... 推理逻辑
```

### 错误示例

```python
# ❌ 错误：没有设置TRANSFORMERS_OFFLINE
import torch
from transformers import AutoModel  # 这会尝试连接HuggingFace，导致失败
```

### 关键点

1. **`TRANSFORMERS_OFFLINE=1`** 必须在任何transformers相关导入之前设置
2. **每个子进程**都需要单独设置环境变量
3. **模型路径**使用本地绝对路径 `/data01/vla/models--nvidia--Alpamayo-R1-10B/...`
4. **虚拟环境**使用 `~/mikelee/alpamayo-main/.venv/`

---

## 参考资料

- HuggingFace: https://huggingface.co/nvidia/Alpamayo-R1-10B
- 数据集: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
- 本地知识库: 
  - `MEMORY.md` (Alpamayo-R1 自动驾驶项目 章节)
  - `knowledge/alpamayo-r1-inference.md` (批量推理详细文档)
  - `knowledge/alpamayo-r1-workflow.md` (完整流程文档)
