# Alpamayo-R1 批量推理完整流程

> 版本: 2026-03-19 (V5 严格版本)  
> 适用范围: 单个clip的连续批量推理

---

## 快速开始 (一键脚本)

### 使用 `auto_process.py` (推荐，默认并行推理)

**脚本位置**: `~/alpamayo_infer/auto_process.py`

#### 基础用法 - 并行推理（默认）
```bash
python3 ~/alpamayo_infer/auto_process.py \
    --clip 054da32b-9f3d-4074-93ab-044036b679f8 \
    --num_frames 100 \
    --step 5 \
    --traj 1
```

#### 单卡推理 - 禁用并行
```bash
python3 ~/alpamayo_infer/auto_process.py \
    --clip 054da32b-9f3d-4074-93ab-044036b679f8 \
    --no-parallel \
    --num_frames 100 \
    --step 5 \
    --traj 1
```

#### 完整流程 - 推理 + 视频
```bash
python3 ~/alpamayo_infer/auto_process.py \
    --clip 054da32b-9f3d-4074-93ab-044036b679f8 \
    --num_frames 100 \
    --step 5 \
    --traj 1 \
    --video
```

#### 参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--clip` | Clip ID (必需) | - |
| `--chunk` | Chunk 号 | 0 |
| `--num_frames` | 推理帧数 | 20 |
| `--traj` | 轨迹数量 (1/3/6) | 1 |
| `--step` | 采样步长 (每N帧推理一帧) | 1 |
| `--video` | 是否生成视频 | False |
| `--parallel` | 使用并行推理 | True |
| `--no-parallel` | 禁用并行推理，使用单卡 | False |
| `--skip-preprocess` | 跳过预处理 | False |
| `--no-download` | 不下载视频到本地 | False |

#### 执行流程
1. **Step 1**: V5 严格数据预处理 (`preprocess_strict.py`)
2. **Step 2**: V5 严格推理
   - 默认: 并行推理 (`run_inference_multi_gpu.py`)
   - `--no-parallel`: 单卡推理 (`run_inference_new_strict.py`)
   - **两种模式输出目录一致**: `result_strict/inference_results_strict.csv`
3. **Step 3**: 生成组合帧 (`batch_create_combined_v5.py`) - 仅 `--video` 时执行
4. **Step 4**: 生成视频 (`step4_create_video_v5.py`) - 仅 `--video` 时执行
5. **下载**: 自动 scp 视频到本地 - 仅 `--video` 且非 `--no-download` 时执行

---

## 多GPU并行推理 (独立使用)

### 使用 `run_inference_multi_gpu.py`

**脚本位置**: `~/alpamayo_infer/run_inference_multi_gpu.py`

**功能**:
- **自动GPU检测**: 检测所有可用GPU及其显存
- **智能实例分配**: 显存≥70G跑2实例，≥35G跑1实例
- **崩溃自动重启**: 每个worker崩溃后自动重启，最多3次
- **统一输出**: 输出与非并行化版本完全一致

**输出位置**:
```
result_strict/
├── inference_results_strict.csv  # 推理结果
└── pred_*.npy                    # 轨迹预测
```

#### 用法
```bash
python3 ~/alpamayo_infer/run_inference_multi_gpu.py \
    --clip 054da32b-9f3d-4074-93ab-044036b679f8 \
    --num_frames 1000 \
    --step 1 \
    --traj 1
```

#### 参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--clip` | Clip ID (必需) | - |
| `--chunk` | Chunk 号 | 0 |
| `--num_frames` | 推理帧数 | 100 |
| `--traj` | 轨迹数量 (1/3/6) | 1 |
| `--step` | 采样步长 | 1 |
| `--top_p` | 采样top_p | 0.98 |
| `--temp` | 采样temperature | 0.6 |
| `--max_len` | 最大生成长度 | 256 |

#### 性能预估
| GPU配置 | 可用实例 | 1000帧理论时间 |
|---------|---------|---------------|
| 8×A100 (80G) 空闲 | 16 | ~70秒 (单帧1.1s÷16) |
| 8×A100 (80G) 轻度使用 | 8-12 | ~90-140秒 |
| 4×A100 (80G) 空闲 | 8 | ~140秒 |

#### 显存阈值配置
```python
VRAM_THRESHOLD_SINGLE = 35  # GB, 单实例阈值
VRAM_THRESHOLD_DOUBLE = 70  # GB, 双实例阈值
MAX_RESTARTS = 3            # 最大重启次数
```

---

## 一、数据预处理 (V5 严格版本)

### 1.1 V5 严格预处理

**脚本**: `~/alpamayo_infer/preprocess_strict.py`

**功能**:
- GPU硬件解码视频 (h264_cuvid)
- **逐点时间戳对齐**: 每个100ms目标点独立查找最接近帧（±10ms容差）
- **严格边界处理**: 4相机共同时间范围 + 未来信息保护
- 自动过滤边界帧

**输入**:
- `{clip_id}.camera_*.mp4` (4个相机视频)
- `{clip_id}.egomotion.parquet`
- `{clip_id}.timestamps.parquet`

**输出**:
```
data/
├── camera_images/                    # 解码后的JPG
│   ├── camera_cross_left_120fov/
│   ├── camera_front_wide_120fov/
│   ├── camera_cross_right_120fov/
│   └── camera_front_tele_30fov/
├── egomotion/                        # 轨迹数据
│   ├── frame_000000_history.npy      # (16,11) 历史轨迹
│   └── frame_000000_future_gt.npy    # (64,4) 未来真值
└── inference_index_strict.csv        # 严格索引
```

**索引表列**:
- `frame_id`: 帧ID
- `ego_idx`: ego索引
- `timestamp`: t0时间戳
- `cam_*_f{0-3}_idx`: 4相机×4帧的图像索引

**过滤结果**:
- 典型有效帧: 1054+ 帧
- egomotion时间容差: <10ms
- 图像帧时间容差: <33ms

---

## 二、批量推理 (V5 严格版本)

### 2.1 V5 严格推理

**脚本**: `~/alpamayo_infer/run_inference_new_strict.py`

**配置**:
```bash
python3 run_inference_new_strict.py \
    --clip <clip_id> \
    --num_frames 100 \      # 推理帧数
    --step 5 \              # 每5帧推理一次
    --traj 1                # 轨迹数量 (1/3/6)
```

**输入**:
- `inference_index_strict.csv`
- `camera_images/`
- `egomotion/`

**输出**:
```
result_strict/
├── inference_results_strict.csv  # 推理结果
└── pred_*.npy                    # 轨迹预测 (64,3)
```

**结果CSV列**:
- `frame_id`: 帧ID
- `ego_idx`: ego索引
- `ade`: ADE误差
- `inference_time_ms`: GPU推理时间
- `coc_text`: CoC推理文本

**性能**:
| 轨迹数 | GPU时间 | minADE |
|--------|---------|--------|
| 1条 | ~1.1s | ~2-3m |
| 3条 | ~2.1s | ~1.5m |
| 6条 | ~3.5s | ~0.7m |

---

## 三、可视化 (V5 版本)

### 3.1 批量创建组合帧 (V5)

**脚本**: `~/alpamayo_infer/batch_create_combined_v5.py`

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

**坐标轴范围**:
- X: -15m ~ +15m
- Y: 0m ~ 60m

**输出**:
```
result_strict/combined_frames/
└── frame_*.jpg  (JPG格式, quality=90)
```

### 3.2 生成视频 (V5)

**脚本**: `~/alpamayo_infer/step4_create_video_v5.py`

**技术参数**:
- 编码器: libx264 (H.264)
- 帧率: 10 FPS
- 分辨率: 1280×1800
- 质量: CRF=23

**输出**:
```
result_strict/combined_video_<clip_id>.mp4
```

**压缩效果**:
| 项目 | 大小 | 压缩比 |
|------|------|--------|
| 源帧 (JPG, 100帧) | ~25 MB | - |
| 输出视频 | ~6 MB | **4x** |

---

## 四、完整流程步骤 (手动执行)

如果不想使用一键脚本，可以分步执行：

### Step 0: 创建目录结构
```bash
mkdir -p /data01/vla/data/data_sample_chunk0/infer/{clip_id}/{data,result}
```

### Step 1: V5 严格预处理
```bash
~/miniconda3/bin/python3 ~/alpamayo_infer/preprocess_strict.py \
    --clip <clip_id> --chunk 0
```

### Step 2: V5 严格推理
```bash
source ~/mikelee/alpamayo-main/.venv/bin/activate
python3 ~/alpamayo_infer/run_inference_new_strict.py \
    --clip <clip_id> \
    --num_frames 100 --step 5 --traj 1
```

### Step 3: 生成组合帧
```bash
~/miniconda3/bin/python3 ~/alpamayo_infer/batch_create_combined_v5.py \
    --clip <clip_id> --chunk 0
```

### Step 4: 生成视频
```bash
~/miniconda3/bin/python3 ~/alpamayo_infer/step4_create_video_v5.py \
    --clip <clip_id> --chunk 0
```

### Step 5: 下载到本地
```bash
scp gpu-server:/data01/vla/data/data_sample_chunk0/infer/{clip_id}/result_strict/combined_video_*.mp4 \
    ~/alpamayo_infer/
```

---

## 五、文件位置汇总

| 文件/目录 | 位置 |
|-----------|------|
| 原始数据 | `/data01/vla/data/data_sample_chunk0/` |
| 一键脚本 | `~/alpamayo_infer/auto_process.py` |
| 预处理脚本 | `~/alpamayo_infer/preprocess_strict.py` |
| 单GPU推理 | `~/alpamayo_infer/run_inference_new_strict.py` |
| 多GPU推理 | `~/alpamayo_infer/run_inference_multi_gpu.py` |
| 组合帧脚本 | `~/alpamayo_infer/batch_create_combined_v5.py` |
| 视频脚本 | `~/alpamayo_infer/step4_create_video_v5.py` |
| 预处理后数据 | `/data01/vla/data/data_sample_chunk0/infer/{clip_id}/data/` |
| **推理结果** | `/data01/vla/data/data_sample_chunk0/infer/{clip_id}/result_strict/` |
| - 结果CSV | `result_strict/inference_results_strict.csv` |
| - 预测文件 | `result_strict/pred_*.npy` |
| - 组合帧 | `result_strict/combined_frames/` |
| - 视频 | `result_strict/combined_video_*.mp4` |
| 本地结果 | `~/alpamayo_infer/` |

---

## 六、V5 关键改进点

1. **逐点时间戳对齐**: 每个100ms目标点独立找最接近帧（±10ms容差）
2. **修复坐标系 Bug**: 四元数列索引错误（timestamp列误当qx）
3. **严格边界处理**: 4相机共同时间范围 + 未来信息保护
4. **图像帧策略**: f3只向后查找（不能看未来）
5. **一键自动化**: `auto_process.py` 支持 `--video` 参数控制是否生成视频
6. **多GPU并行**: `run_inference_multi_gpu.py` 支持8卡并行，自动显存分配
7. **GPU硬件解码**: 比CPU快~100倍
8. **高质量索引**: 时间对齐<17ms，有效帧比例高

---

## 七、多GPU并行推理详解

### 并行化架构

```
┌─────────────────────────────────────────────────────────────┐
│                      主控进程 (Master)                        │
├─────────────────────────────────────────────────────────────┤
│  1. GPU检测 → 获取可用GPU + 显存 → 计算实例数 (1-2个/GPU)      │
│  2. 任务分配 → 把N帧分配到M个GPU进程                          │
│  3. 进程池管理 → 启动M个InferenceWorker进程                   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │ Worker 0│          │ Worker 1│          │ Worker M│
   │ GPU 0   │          │ GPU 0/1 │          │ GPU 7   │
   └────┬────┘          └────┬────┘          └────┬────┘
        │                    │                    │
   ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
   │ 数据加载 │          │ 数据加载 │          │ 数据加载 │
   │ (预加载) │          │ (预加载) │          │ (预加载) │
   ├─────────┤          ├─────────┤          ├─────────┤
   │ 推理主程 │          │ 推理主程 │          │ 推理主程 │
   │ (GPU计算)│          │ (GPU计算)│          │ (GPU计算)│
   └─────────┘          └─────────┘          └─────────┘
```

### 显存分配策略

| GPU显存 | 分配实例数 | 说明 |
|---------|-----------|------|
| ≥70 GB | 2个 | 可运行2个模型实例 |
| ≥35 GB | 1个 | 可运行1个模型实例 |
| <35 GB | 0个 | 显存不足，跳过 |

### 性能实测

| 配置 | 帧数 | 单卡时间 | 并行时间 | 加速比 |
|------|------|---------|---------|--------|
| 8×A100 (80G) | 200 | ~400秒 | ~76秒 | 5.3x |
| 8×A100 (80G) | 1120 | ~40分钟 | ~6分钟 | 6.7x |

### 关键配置参数

```python
# 显存阈值 (GB)
VRAM_THRESHOLD_SINGLE = 35  # 单实例阈值
VRAM_THRESHOLD_DOUBLE = 70  # 双实例阈值

# 重启次数
MAX_RESTARTS = 3  # 每个worker最多重启3次
```

---

## 八、推理脚本开发规范

### ⚠️ 必须使用本地模型权重

**环境变量设置（必须在任何导入之前）**:

```python
#!/usr/bin/env python3
import os
import sys

# 第一步：设置离线模式（必须在任何transformers导入之前）
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 第二步：添加模型路径
sys.path.insert(0, '/home/user/mikelee/alpamayo-main/src')

# 第三步：导入其他库
import torch
import numpy as np
# ... 其他导入

# 第四步：加载本地模型
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

MODEL_PATH = "/data01/vla/models--nvidia--Alpamayo-R1-10B/snapshots/22fab1399111f50b52bfbe5d8b809f39bd4c2fe1"
model = AlpamayoR1.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)
```

### 多进程Worker模板

```python
def inference_worker_entry(args_dict):
    """
    Worker入口函数 - 多GPU并行推理
    必须在开头设置GPU和离线模式
    """
    # 设置当前进程的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_dict['gpu_id'])
    
    # 设置离线模式（每个进程都要设置）
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # 现在才导入模型库
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    from scipy.spatial.transform import Rotation as R
    
    # 加载模型
    model = AlpamayoR1.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    
    # 推理逻辑...
```

### 常见错误

**错误1**: 没有设置 `TRANSFORMERS_OFFLINE`
```python
# ❌ 错误
import torch
from transformers import AutoModel  # 会尝试连接HuggingFace

# ✅ 正确
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
```

**错误2**: 子进程没有设置环境变量
```python
# ❌ 错误：只在主进程设置
os.environ["TRANSFORMERS_OFFLINE"] = "1"
p = Process(target=worker)  # 子进程不会继承环境变量

# ✅ 正确：在worker函数开头设置
def worker():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # ...
```

**错误3**: 使用miniconda3的Python运行推理脚本
```bash
# ❌ 错误：miniconda3没有安装模型依赖
~/miniconda3/bin/python3 run_inference.py

# ✅ 正确：使用虚拟环境的Python
source ~/mikelee/alpamayo-main/.venv/bin/activate
python3 run_inference.py
```

---

*最后更新: 2026-03-19*
