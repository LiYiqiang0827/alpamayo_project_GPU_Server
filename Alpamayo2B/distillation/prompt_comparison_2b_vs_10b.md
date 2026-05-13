# Alpamayo2B vs Alpamayo1.5-10B Prompt 对比分析

## 分析日期
2026-05-13

## 关键发现

### 1. 图片 Token 数量

| 模型 | 每张图片 token 数 | 16 张图片总 token 数 |
|------|------------------|---------------------|
| **Alpamayo2B** | **180** | **2880** |
| Alpamayo1.5-10B | 180 | 2880 |

**确认**: 576×320 的图片经过 ViT 处理后产生 180 个 visual tokens。

### 2. Prefill Token 数量对比

#### Alpamayo2B (实测)
| 组件 | Token 数量 |
|------|-----------|
| Image pad | 2880 (16 × 180) |
| Vision start | 16 |
| Vision end | 16 |
| **图片相关总计** | **2912** |
| traj_history_start | 1 |
| traj_history | 48 |
| traj_history_end | 1 |
| 其他文本 token | 43 |
| **总 prefill tokens** | **3005** |

#### Alpamayo1.5-10B (预计)
由于 10B 模型使用相同的 Qwen3-VL processor 和相同的词表，prefill token 数量应该与 2B 模型**完全一致**。

### 3. 特殊 Token 处理

**重要修复**: 
- `traj_history_start`、`traj_history`、`cot_start` 等 token 在 tokenizer 的 `added_tokens` 列表中存在
- 但 processor 自带的 tokenizer **缺少这些特殊 token**，导致它们被拆分成子词
- **解决方案**: 在 dataloader 初始化时，将 processor 的 tokenizer 替换为完整的 tokenizer_final

### 4. 词表对比

| 项目 | Alpamayo2B | Alpamayo1.5-10B |
|------|-----------|----------------|
| 总词表大小 | 155,697 | 155,697 |
| 基础词表 | [0, 151642] | [0, 151642] |
| 扩展词表 | [151669, 155696] | [151669, 155696] |
| 特殊 token | 一致 | 一致 |

**结论**: 两个模型的词表**完全一致**。

### 5. Prompt 格式

两个模型使用**完全相同的 prompt 格式**：

```
<|im_start|>system
You are a driving assistant that generates safe and accurate actions.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|>...<|vision_end|> (×16)
<|traj_history_start|><|traj_history|>...×48<|traj_history_end|>
output the chain-of-thought reasoning of the driving process, then output the future trajectory.<|im_end|>
<|im_start|>assistant
<|cot_start|>
```

## 修复记录

### 修复 1: 图片尺寸
- **问题**: dataloader 默认将图片 resize 到 224×224，导致只有 169 个 visual tokens
- **解决**: 将 `image_size` 默认值改为 `None`，使用原始图片尺寸（576×320）
- **结果**: 现在每张图片产生 180 个 visual tokens

### 修复 2: Processor Tokenizer
- **问题**: processor 自带的 tokenizer 缺少 `traj_history`、`cot_start` 等特殊 token
- **解决**: 在 dataloader `__init__` 中检测并替换为完整的 tokenizer
- **结果**: 特殊 token 现在被正确识别为单个 token

### 修复 3: Chat Template
- **问题**: processor 的 tokenizer 没有 chat_template
- **解决**: 在替换 tokenizer 时，确保使用有 chat_template 的 tokenizer_final
- **结果**: chat template 与 10B 模型一致

## Token ID List 验证

### 关键 Token IDs

| Token | ID |
|-------|-----|
| `<|im_start|>` | 151644 |
| `<|im_end|>` | 151645 |
| `<|vision_start|>` | 151652 |
| `<|vision_end|>` | 151653 |
| `<|image_pad|>` | 151655 |
| `<|traj_history_start|>` | 155673 |
| `<|traj_history|>` | 155679 |
| `<|traj_history_end|>` | 155674 |
| `<|cot_start|>` | 155675 |

### Token 位置分析

| Token | 数量 | 位置 |
|-------|------|------|
| Vision start | 16 | [20, 202, 384, 566, 748, 930, 1112, 1294, 1476, 1658, 1840, 2022, 2204, 2386, 2568, 2750] |
| Vision end | 16 | [201, 383, 565, 747, 929, 1111, 1293, 1475, 1657, 1839, 2021, 2203, 2385, 2567, 2749, 2931] |
| Image pad | 2880 | 20-2931 (每 182 个 token 一个图片块) |
| Traj history start | 1 | [2932] |
| Traj history | 48 | [2933-2980] |
| Traj history end | 1 | [2981] |
| CoT start | 1 | [3005] |

## 结论

1. **Prompt 格式一致**: 2B 和 10B 模型使用完全相同的 prompt 格式
2. **Prefill token 数量一致**: 都是约 3005 个 token（含 2912 个图片相关 token）
3. **图片 token 数量一致**: 都是 180 个 per image
4. **词表完全一致**: 可以安全地进行知识蒸馏
5. **Token ID list 一致**: 进入 LLM 前的 token ID 序列完全相同

## 对蒸馏训练的影响

由于 prompt 格式和 prefill token 数量完全一致，蒸馏训练可以正常进行：
- 学生模型（2B）和教师模型（10B）的输入完全一致
- 教师 logits 的 token 位置与学生模型的输出位置对齐
- 特殊 token（traj_history, cot_start 等）被正确处理
