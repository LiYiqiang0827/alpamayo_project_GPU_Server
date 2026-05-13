"""
Alpamayo2B 蒸馏数据集加载器
=============================
用于 Alpamayo1.5-10B → Alpamayo2B 的 LLM 知识蒸馏

主要功能：
1. 加载教师模型推理结果（CoT文本 + logits）
2. 加载学生模型输入（图片 + 历史轨迹）
3. 支持温度缩放的 soft target 生成
4. 兼容 Qwen3VL 模型的输入格式

作者: 小胖龟
创建时间: 2026-05-12
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ==================== 配置 ====================
CAMERAS = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]

TIME_FRAMES = [0, 1, 2, 3]  # 每相机4帧历史


class AlpamayoDistillationDataset(Dataset):
    """
    Alpamayo2B 蒸馏数据集
    
    每个样本包含：
        - 输入: 16张图片 (4相机×4帧) + 历史轨迹
        - 教师输出: CoT文本 + logits (用于蒸馏)
        - 学生标签: CoT token IDs
    
    支持从 Parquet 格式的 logits 文件读取教师模型输出
    """
    
    def __init__(
        self,
        infer_result_csv: str,
        teacher_logits_dir: str,
        tokenizer,
        processor,
        data_root: str = "/data01/mikelee/data",
        temperature: float = 2.0,
        max_seq_length: int = 512,
        image_size: Tuple[int, int] = (224, 224),
        use_small_images: bool = True,
        cache_images: bool = False,
        max_cache_gb: int = 100,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            infer_result_csv: infer_results_all.csv 路径
            teacher_logits_dir: 教师模型 logits 目录 (Parquet文件)
            tokenizer: 用于编码 CoT 文本
            processor: Qwen3VL processor (处理图片和文本)
            data_root: 数据根目录
            temperature: 蒸馏温度，用于计算 soft target
            max_seq_length: 最大序列长度
            image_size: 图片 resize 大小
            use_small_images: 是否使用 _small.jpg 后缀
            cache_images: 是否缓存图片到内存
            max_cache_gb: 图片缓存最大内存 (GB)
            split: 数据集划分 - "train", "val", "test"
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
        """
        self.infer_result_csv = infer_result_csv
        self.teacher_logits_dir = Path(teacher_logits_dir)
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_root = Path(data_root)
        self.temperature = temperature
        self.max_seq_length = max_seq_length
        self.image_size = image_size
        self.use_small_images = use_small_images
        self.cache_images = cache_images
        
        # 图片后缀
        self.image_suffix = "_small.jpg" if use_small_images else ".jpg"
        
        # 加载推理结果索引
        all_samples = self._load_infer_results()
        
        # 按 clip 划分数据集
        self.samples = self._split_by_clip(
            all_samples, 
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        
        # 图片缓存（可选）
        self.image_cache = {} if cache_images else None
        self.max_cache_bytes = max_cache_gb * 1024 * 1024 * 1024
        self.current_cache_bytes = 0
        
        logger.info(f"Dataset initialized: {len(self.samples)} samples ({split})")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Image cache: {'enabled' if cache_images else 'disabled'}")
    
    def _load_infer_results(self) -> List[Dict[str, Any]]:
        """
        加载推理结果 CSV，构建样本列表
        
        Returns:
            样本列表，每个样本包含 (chunk_id, clip_id, frame_id, cot_text, logits_path)
        """
        logger.info(f"Loading inference results from {self.infer_result_csv}...")
        
        infer_df = pd.read_csv(self.infer_result_csv)
        samples = []
        missing_logits = 0
        
        for idx, row in tqdm(infer_df.iterrows(), total=len(infer_df), desc="Loading samples"):
            chunk_id = int(row['chunk_id'])
            clip_id = row['clip_name']
            frame_id = int(row['frame_number'])
            cot_text = row.get('cot_result', '')
            
            # 构建 logits 文件路径
            # 格式: chunk{chunk_id:04d}_{clip_id}_{frame_id:06d}_logits.parquet
            logits_filename = f"chunk{chunk_id:04d}_{clip_id}_{frame_id:06d}_logits.parquet"
            logits_path = self.teacher_logits_dir / logits_filename
            
            # 检查 logits 文件是否存在
            if not logits_path.exists():
                # 尝试不带前导零的格式
                logits_filename = f"chunk{chunk_id}_{clip_id}_{frame_id}_logits.parquet"
                logits_path = self.teacher_logits_dir / logits_filename
            
            if not logits_path.exists():
                missing_logits += 1
                continue
            
            samples.append({
                'chunk_id': chunk_id,
                'clip_id': clip_id,
                'frame_id': frame_id,
                'cot_text': cot_text,
                'logits_path': str(logits_path),
            })
        
        if missing_logits > 0:
            logger.warning(f"Missing logits for {missing_logits}/{len(infer_df)} samples")
        
        logger.info(f"Loaded {len(samples)} valid samples with logits")
        return samples
    
    def _split_by_clip(
        self,
        samples: List[Dict[str, Any]],
        split: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> List[Dict[str, Any]]:
        """
        按 clip 划分数据集
        
        避免数据泄露：同一段视频的所有帧只出现在一个划分中
        
        Args:
            samples: 所有样本列表
            split: "train", "val", "test"
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
            
        Returns:
            划分后的样本列表
        """
        import random
        
        # 收集所有唯一的 clip
        clip_to_samples = defaultdict(list)
        for idx, sample in enumerate(samples):
            clip_to_samples[sample['clip_id']].append(idx)
        
        all_clips = sorted(list(clip_to_samples.keys()))
        num_clips = len(all_clips)
        
        # 设置随机种子并打乱
        random.seed(seed)
        random.shuffle(all_clips)
        
        # 计算划分边界
        train_end = int(num_clips * train_ratio)
        val_end = train_end + int(num_clips * val_ratio)
        
        # 划分 clips
        if split == "train":
            split_clips = set(all_clips[:train_end])
        elif split == "val":
            split_clips = set(all_clips[train_end:val_end])
        elif split == "test":
            split_clips = set(all_clips[val_end:])
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'")
        
        # 收集对应样本
        split_samples = []
        for clip_id in split_clips:
            for idx in clip_to_samples[clip_id]:
                split_samples.append(samples[idx])
        
        logger.info(
            f"Split '{split}': {len(split_clips)}/{num_clips} clips, "
            f"{len(split_samples)}/{len(samples)} samples"
        )
        
        return split_samples
    
    def _load_teacher_logits(self, logits_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        加载教师模型的 logits 并计算 soft target 和 hard target
        
        Args:
            logits_path: Parquet 文件路径
            
        Returns:
            teacher_logits: (seq_len, vocab_size) 原始 logits
            teacher_soft: (seq_len, vocab_size) 温度缩放后的概率分布
            teacher_hard: (seq_len,) hard token IDs (argmax结果)
        """
        # 读取 Parquet 文件
        df = pd.read_parquet(logits_path)
        
        # 提取 logits 和 token IDs
        # Parquet 格式: 每行一个 token，包含 token_idx, token_id, hard_token_id, logits(list)
        logits_list = []
        hard_token_ids = []
        
        for _, row in df.iterrows():
            # logits 列可能是 list 或 numpy array
            logits = row['logits']
            if isinstance(logits, str):
                # 如果是字符串，需要解析
                logits = eval(logits)
            logits = np.array(logits, dtype=np.float32)
            logits_list.append(logits)
            
            # 使用 hard_token_id 作为 hard target (logits的argmax结果)
            hard_token_id = int(row.get('hard_token_id', row.get('token_id', 0)))
            hard_token_ids.append(hard_token_id)
        
        # 转换为 tensor
        teacher_logits = torch.from_numpy(np.stack(logits_list))  # (seq_len, vocab_size)
        teacher_hard = torch.tensor(hard_token_ids, dtype=torch.long)  # (seq_len,)
        
        # 温度缩放计算 soft target
        teacher_soft = self._compute_soft_target(teacher_logits, self.temperature)
        
        return teacher_logits, teacher_soft, teacher_hard
    
    def _compute_soft_target(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        使用温度缩放计算 soft target
        
        Args:
            logits: (seq_len, vocab_size) 原始 logits
            temperature: 温度参数
            
        Returns:
            soft_target: (seq_len, vocab_size) 概率分布
        """
        # 温度缩放
        scaled_logits = logits / temperature
        
        # Softmax 计算概率分布
        soft_target = torch.softmax(scaled_logits, dim=-1)
        
        return soft_target
    
    def _load_images(self, chunk_id: int, clip_id: str, frame_id: int) -> List[Image.Image]:
        """
        加载16张图片（4相机×4帧）
        
        Args:
            chunk_id: chunk ID
            clip_id: clip ID
            frame_id: frame ID
            
        Returns:
            images: List[PIL.Image] 16张图片
        """
        clip_dir = self.data_root / f"data_sample_chunk{chunk_id}" / "infer" / clip_id / "data"
        
        # 读取 inference_index_strict.csv 获取图片索引
        index_csv = clip_dir / "inference_index_strict.csv"
        image_indices = {}
        
        if index_csv.exists():
            try:
                index_df = pd.read_csv(index_csv)
                frame_rows = index_df[index_df['frame_id'] == frame_id]
                if len(frame_rows) > 0:
                    frame_row = frame_rows.iloc[0]
                    for cam in CAMERAS:
                        for t in TIME_FRAMES:
                            col_name = f"{cam}_f{t}_idx"
                            if col_name in frame_row.index:
                                image_indices[(cam, t)] = int(frame_row[col_name])
            except Exception as e:
                logger.warning(f"Failed to load index for {clip_id}/{frame_id}: {e}")
        
        # 加载图片
        images = []
        for cam in CAMERAS:
            for t in TIME_FRAMES:
                # 获取图片索引
                if (cam, t) in image_indices:
                    img_idx = image_indices[(cam, t)]
                else:
                    # fallback: 使用 frame_id
                    img_idx = frame_id
                
                img_path = clip_dir / "camera_images" / cam / f"{img_idx:06d}{self.image_suffix}"
                
                # 尝试加载图片
                img = self._load_image(img_path)
                images.append(img)
        
        return images
    
    def _load_image(self, img_path: Path) -> Image.Image:
        """
        加载单张图片，支持缓存
        
        Args:
            img_path: 图片路径
            
        Returns:
            PIL.Image
        """
        # 检查缓存
        if self.cache_images and img_path in self.image_cache:
            return self.image_cache[img_path].copy()
        
        # 加载图片
        if img_path.exists():
            img = Image.open(img_path).convert('RGB')
        else:
            # fallback: 黑色图片
            logger.warning(f"Image not found: {img_path}, using black image")
            img = Image.new('RGB', self.image_size, color='black')
        
        # Resize
        img = img.resize(self.image_size, Image.BILINEAR)
        
        # 缓存（如果启用）
        if self.cache_images:
            img_array = np.array(img)
            size = img_array.nbytes
            
            # LRU 淘汰
            while (self.current_cache_bytes + size > self.max_cache_bytes and 
                   len(self.image_cache) > 0):
                oldest = next(iter(self.image_cache))
                self.current_cache_bytes -= self.image_cache[oldest].nbytes
                del self.image_cache[oldest]
            
            self.image_cache[img_path] = img_array
            self.current_cache_bytes += size
        
        return img
    
    def _load_history(self, chunk_id: int, clip_id: str, frame_id: int) -> np.ndarray:
        """
        加载历史轨迹
        
        Args:
            chunk_id: chunk ID
            clip_id: clip ID
            frame_id: frame ID
            
        Returns:
            history: (16, 11) numpy array
        """
        history_path = (
            self.data_root / f"data_sample_chunk{chunk_id}" / "infer" / clip_id / "data" /
            "egomotion" / f"frame_{frame_id:06d}_history.npy"
        )
        
        if history_path.exists():
            history = np.load(history_path)
        else:
            logger.warning(f"History not found: {history_path}, using zeros")
            history = np.zeros((16, 11), dtype=np.float32)
        
        return history
    
    def _build_prompt(self, cot_text: str) -> str:
        """
        构建 Qwen3VL 的 prompt
        
        Args:
            cot_text: CoT 文本
            
        Returns:
            prompt: 完整的 prompt 字符串
        """
        # 图片占位符
        image_placeholders = ""
        for _ in range(16):  # 16张图片
            image_placeholders += "<|vision_start|><|image_pad|><|vision_end|>"
        
        # 构建消息
        messages = [
            {
                "role": "system",
                "content": "You are a driving assistant that generates safe and accurate actions."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{image_placeholders}<|traj_history_start|><|traj_history|>*48<|traj_history_end|>output the chain-of-thought reasoning of the driving process, then output the future trajectory."
                    }
                ]
            },
            {
                "role": "assistant",
                "content": f"<|cot_start|>{cot_text}<|cot_end|>"
            }
        ]
        
        # 应用 chat template
        # 使用tokenizer而不是processor，因为processor可能没有chat_template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        return prompt
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取一个训练样本
        
        Returns:
            {
                'input_ids': torch.Tensor,           # (seq_len,)
                'attention_mask': torch.Tensor,      # (seq_len,)
                'labels': torch.Tensor,              # (seq_len,)
                'pixel_values': torch.Tensor,        # (16, 3, H, W)
                'image_grid_thw': torch.Tensor,      # (16, 3)
                'history': torch.Tensor,             # (16, 11)
                'teacher_logits': torch.Tensor,      # (seq_len, vocab_size)
                'teacher_soft': torch.Tensor,        # (seq_len, vocab_size)
            }
        """
        sample = self.samples[idx]
        chunk_id = sample['chunk_id']
        clip_id = sample['clip_id']
        frame_id = sample['frame_id']
        cot_text = sample['cot_text']
        logits_path = sample['logits_path']
        
        # 1. 加载图片
        images = self._load_images(chunk_id, clip_id, frame_id)
        
        # 2. 加载历史轨迹
        history = self._load_history(chunk_id, clip_id, frame_id)
        history_tensor = torch.from_numpy(history).float()
        
        # 3. 构建 prompt
        prompt = self._build_prompt(cot_text)
        
        # 4. 使用 processor 处理输入
        try:
            # 使用 processor 同时处理文本和图片
            # 注意：processor 会自动将 <|image_pad|> 扩展为正确的 token 数量
            # 不启用 truncation，避免截断图片 token
            inputs = self.processor(
                text=[prompt],
                images=images,
                return_tensors="pt",
                padding=True,
            )
            
            # 移除 batch 维度
            inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # 确保 image_grid_thw 是 long 类型
            if 'image_grid_thw' in inputs:
                inputs['image_grid_thw'] = inputs['image_grid_thw'].long()
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            # 返回空样本
            return self._get_empty_sample()
        
        # 5. 设置 labels（和 input_ids 相同）
        if 'input_ids' in inputs:
            inputs['labels'] = inputs['input_ids'].clone()
        
        # 6. 加载教师 logits
        try:
            teacher_logits, teacher_soft, teacher_hard = self._load_teacher_logits(logits_path)
            inputs['teacher_logits'] = teacher_logits
            inputs['teacher_soft'] = teacher_soft
            inputs['teacher_hard'] = teacher_hard
        except Exception as e:
            logger.error(f"Error loading teacher logits for sample {idx}: {e}")
            # 如果没有 logits，返回空 tensor
            vocab_size = len(self.tokenizer)
            seq_len = inputs['input_ids'].shape[0] if 'input_ids' in inputs else 1
            inputs['teacher_logits'] = torch.zeros(seq_len, vocab_size)
            inputs['teacher_soft'] = torch.zeros(seq_len, vocab_size)
            inputs['teacher_hard'] = torch.zeros(seq_len, dtype=torch.long)
        
        # 7. 添加历史轨迹
        inputs['history'] = history_tensor
        
        return inputs
    
    def _get_empty_sample(self) -> Dict[str, torch.Tensor]:
        """返回空样本（用于错误处理）"""
        return {
            'input_ids': torch.tensor([0]),
            'attention_mask': torch.tensor([1]),
            'labels': torch.tensor([0]),
            'pixel_values': torch.zeros(16, 3, *self.image_size),
            'image_grid_thw': torch.zeros(16, 3, dtype=torch.long),
            'history': torch.zeros(16, 11),
            'teacher_logits': torch.zeros(1, len(self.tokenizer)),
            'teacher_soft': torch.zeros(1, len(self.tokenizer)),
            'teacher_hard': torch.zeros(1, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    自定义 collate 函数，处理变长序列
    
    对 input_ids, attention_mask, labels, teacher_logits, teacher_soft, teacher_hard 进行 padding
    """
    # 找到最大长度
    max_len = max(item['input_ids'].shape[0] for item in batch)
    vocab_size = batch[0]['teacher_logits'].shape[-1]
    
    # 初始化列表
    input_ids = []
    attention_masks = []
    labels = []
    teacher_logits_list = []
    teacher_soft_list = []
    teacher_hard_list = []
    
    for item in batch:
        seq_len = item['input_ids'].shape[0]
        pad_len = max_len - seq_len
        
        # Padding
        input_ids.append(torch.cat([
            item['input_ids'],
            torch.full((pad_len,), 0, dtype=torch.long)
        ]))
        
        attention_masks.append(torch.cat([
            item['attention_mask'],
            torch.zeros(pad_len, dtype=torch.long)
        ]))
        
        labels.append(torch.cat([
            item['labels'],
            torch.full((pad_len,), -100, dtype=torch.long)
        ]))
        
        # Teacher logits padding
        if pad_len > 0:
            pad_logits = torch.full((pad_len, vocab_size), 0.0, dtype=torch.float32)
            teacher_logits_list.append(torch.cat([item['teacher_logits'], pad_logits]))
            teacher_soft_list.append(torch.cat([item['teacher_soft'], pad_logits]))
            teacher_hard_list.append(torch.cat([
                item['teacher_hard'],
                torch.full((pad_len,), -100, dtype=torch.long)
            ]))
        else:
            teacher_logits_list.append(item['teacher_logits'])
            teacher_soft_list.append(item['teacher_soft'])
            teacher_hard_list.append(item['teacher_hard'])
    
    # Stack
    result = {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.stack(labels),
        'teacher_logits': torch.stack(teacher_logits_list),
        'teacher_soft': torch.stack(teacher_soft_list),
        'teacher_hard': torch.stack(teacher_hard_list),
    }
    
    # 处理图片相关（假设所有样本相同）
    if 'pixel_values' in batch[0]:
        result['pixel_values'] = torch.stack([item['pixel_values'] for item in batch])
    
    if 'image_grid_thw' in batch[0]:
        image_grid_thw = batch[0]['image_grid_thw']
        if isinstance(image_grid_thw, torch.Tensor):
            # 确保形状是 (batch_size*num_images, 3)
            # 如果输入是 (num_images, 3)，则重复batch次
            if image_grid_thw.dim() == 2:
                # (num_images, 3) -> (batch_size*num_images, 3)
                result['image_grid_thw'] = image_grid_thw.repeat(len(batch), 1)
            elif image_grid_thw.dim() == 3:
                # (batch, num_images, 3)
                result['image_grid_thw'] = image_grid_thw
            else:
                result['image_grid_thw'] = image_grid_thw.unsqueeze(0).repeat(len(batch), 1)
        else:
            result['image_grid_thw'] = [item['image_grid_thw'] for item in batch]
    
    # 历史轨迹
    if 'history' in batch[0]:
        result['history'] = torch.stack([item['history'] for item in batch])
    
    return result


def create_distillation_dataloader(
    infer_result_csv: str,
    teacher_logits_dir: str,
    tokenizer,
    processor,
    batch_size: int = 1,
    num_workers: int = 4,
    temperature: float = 2.0,
    shuffle: bool = True,
    split: str = "train",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    **kwargs
) -> DataLoader:
    """
    创建蒸馏 DataLoader
    
    Args:
        infer_result_csv: 推理结果 CSV 路径
        teacher_logits_dir: 教师 logits 目录
        tokenizer: tokenizer
        processor: Qwen3VL processor
        batch_size: 批次大小
        num_workers: 数据加载线程数
        temperature: 蒸馏温度
        shuffle: 是否 shuffle
        split: 数据集划分 - "train", "val", "test"
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        
    Returns:
        DataLoader
    """
    dataset = AlpamayoDistillationDataset(
        infer_result_csv=infer_result_csv,
        teacher_logits_dir=teacher_logits_dir,
        tokenizer=tokenizer,
        processor=processor,
        temperature=temperature,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        **kwargs
    )


# ==================== 测试 ====================
if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer, AutoProcessor
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer-csv", default="/gpfs-data/mikelee/infer_result/infer_result_20260507_195923/infer_results_all.csv")
    parser.add_argument("--logits-dir", default="/gpfs-data/mikelee/infer_result/infer_result_20260507_195923/logits")
    parser.add_argument("--tokenizer-path", default="/data01/mikelee/weight/alpamayo2B/tokenizer_final")
    parser.add_argument("--model-path", default="/data01/mikelee/weight/alpamayo2B")
    parser.add_argument("--test-sample", type=int, default=0, help="测试样本索引")
    args = parser.parse_args()
    
    print("Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    print("Creating dataset...")
    dataset = AlpamayoDistillationDataset(
        infer_result_csv=args.infer_csv,
        teacher_logits_dir=args.logits_dir,
        tokenizer=tokenizer,
        processor=processor,
        temperature=2.0,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试加载一个样本
    print(f"\nLoading sample {args.test_sample}...")
    sample = dataset[args.test_sample]
    
    print("\nSample keys:", sample.keys())
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    
    # 测试 DataLoader
    print("\nTesting DataLoader...")
    dataloader = create_distillation_dataloader(
        infer_result_csv=args.infer_csv,
        teacher_logits_dir=args.logits_dir,
        tokenizer=tokenizer,
        processor=processor,
        batch_size=2,
        temperature=2.0,
        split="train",
    )
    
    batch = next(iter(dataloader))
    print("\nBatch keys:", batch.keys())
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # 测试数据集划分
    print("\n" + "=" * 50)
    print("Testing dataset splits...")
    print("=" * 50)
    
    for split_name in ["train", "val", "test"]:
        dataset_split = AlpamayoDistillationDataset(
            infer_result_csv=args.infer_csv,
            teacher_logits_dir=args.logits_dir,
            tokenizer=tokenizer,
            processor=processor,
            temperature=2.0,
            split=split_name,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )
        print(f"\n{split_name}: {len(dataset_split)} samples")
        
        # 检查是否有数据泄露
        if split_name != "train":
            # 获取训练集的 clip
            train_dataset = AlpamayoDistillationDataset(
                infer_result_csv=args.infer_csv,
                teacher_logits_dir=args.logits_dir,
                tokenizer=tokenizer,
                processor=processor,
                temperature=2.0,
                split="train",
                seed=42,
            )
            train_clips = set(s['clip_id'] for s in train_dataset.samples)
            split_clips = set(s['clip_id'] for s in dataset_split.samples)
            overlap = train_clips & split_clips
            print(f"  Overlap with train: {len(overlap)} clips")
            assert len(overlap) == 0, f"Data leakage detected! {len(overlap)} clips overlap with train"
