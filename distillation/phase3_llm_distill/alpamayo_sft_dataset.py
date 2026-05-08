"""
Alpamayo SFT DataLoader
=======================
设计思路：
1. 一次性加载所有 inference_index_strict.csv，建立 (chunk_id, clip_id, frame_id) → 索引映射
2. 利用 1TB 内存，提前加载图片数据到内存缓存
3. 直接通过 dict 查找，无需重复读取 CSV
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)


class AlpamayoIndexManager:
    """
    管理所有 chunk 的索引映射 + CoC 文本
    一次性加载，以 (chunk_id, clip_id, frame_id) 为 key 存储所有信息
    """
    
    def __init__(
        self,
        infer_result_csv: str,
        data_root: str = "/data01/mikelee/data",
        chunk_base_name: str = "data_sample_chunk"
    ):
        """
        Args:
            infer_result_csv: infer_results_all.csv 路径
            data_root: 数据根目录
            chunk_base_name: chunk 目录名前缀
        """
        self.infer_result_csv = infer_result_csv
        self.data_root = Path(data_root)
        self.chunk_base_name = chunk_base_name
        
        # 合并的大索引字典: {(chunk_id, clip_id, frame_id) -> {image_indices, ego_idx, cot_result}}
        self.index_map: Dict[Tuple[int, str, int], Dict[str, Any]] = {}
        
        # clip -> chunk 映射: {(chunk_id, clip_id) -> csv_path}
        self.clip_csv_paths: Dict[Tuple[int, str], str] = {}
        
        logger.info(f"Loading infer results from {infer_result_csv}...")
        self._load_infer_results()
        
    def _load_infer_results(self):
        """读取 infer_results_all.csv，获取所有需要加载的 (chunk_id, clip_id) 并加载 CoC 文本"""
        df = pd.read_csv(self.infer_result_csv)
        
        # 一次性把所有 CoC 文本存进 index_map
        logger.info(f"Loading {len(df)} CoC texts into memory...")
        for _, row in df.iterrows():
            chunk_id = int(row['chunk_id'])
            clip_name = row['clip_name']
            frame_id = int(row['frame_number'])
            cot_result = row['cot_result']
            
            key = (chunk_id, clip_name, frame_id)
            self.index_map[key] = {
                'cot_result': cot_result,
            }
        
        # 提取唯一的 (chunk_id, clip_name) 对，加载索引
        unique_clips = df[['chunk_id', 'clip_name']].drop_duplicates()
        
        logger.info(f"Found {len(unique_clips)} unique (chunk, clip) pairs")
        
        # 加载每个 clip 的 index CSV
        for _, row in unique_clips.iterrows():
            chunk_id = int(row['chunk_id'])
            clip_name = row['clip_name']
            
            chunk_dir = self.data_root / f"{self.chunk_base_name}{chunk_id}" / "infer" / clip_name / "data"
            index_csv = chunk_dir / "inference_index_strict.csv"
            
            if not index_csv.exists():
                logger.warning(f"Index CSV not found: {index_csv}")
                continue
            
            self.clip_csv_paths[(chunk_id, clip_name)] = str(index_csv)
        
        logger.info(f"Index manager initialized with {len(self.clip_csv_paths)} clips")
    
    def load_all_indices(self) -> Dict[Tuple[int, str, int], Dict[str, Any]]:
        """
        一次性加载所有 clip 的 index 数据，补充到 index_map
        返回: {(chunk_id, clip_id, frame_id) -> {image_indices, ego_idx, cot_result}}
        """
        logger.info("Loading all index data into memory...")
        
        for (chunk_id, clip_name), csv_path in self.clip_csv_paths.items():
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                frame_id = int(row['frame_id'])
                ego_idx = int(row['ego_idx'])
                
                key = (chunk_id, clip_name, frame_id)
                
                # 跳过没有 CoC 的帧
                if key not in self.index_map:
                    continue
                
                # 收集4个相机的16个图片索引
                image_indices = {}
                cameras = [
                    'camera_cross_left_120fov',
                    'camera_front_wide_120fov', 
                    'camera_cross_right_120fov',
                    'camera_front_tele_30fov'
                ]
                
                for cam in cameras:
                    for i in range(4):  # f0, f1, f2, f3
                        col = f"{cam}_f{i}_idx"
                        if col in row:
                            image_indices[f"{cam}_f{i}"] = int(row[col])
                
                # 补充索引信息
                self.index_map[key]['ego_idx'] = ego_idx
                self.index_map[key]['image_indices'] = image_indices
                self.index_map[key]['chunk_id'] = chunk_id
                self.index_map[key]['clip_name'] = clip_name
        
        logger.info(f"Loaded {len(self.index_map)} frame indices into memory")
        return self.index_map
    
    def get_info(self, chunk_id: int, clip_id: str, frame_id: int) -> Optional[Dict[str, Any]]:
        """快速查找所有信息（索引+CoC）"""
        return self.index_map.get((chunk_id, clip_id, frame_id))


class ImageCache:
    """
    图片内存缓存
    利用大内存提前加载图片数据
    """
    
    def __init__(self, data_root: str, max_memory_gb: int = 800):
        """
        Args:
            data_root: 数据根目录
            max_memory_gb: 最大使用内存 (GB)
        """
        self.data_root = Path(data_root)
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.current_memory = 0
        
        # 缓存: {(chunk_id, clip_name, camera, frame_idx) -> np.array}
        self.cache: Dict[Tuple, np.ndarray] = {}
        
        # 访问统计用于 LRU
        self.access_order: List[Tuple] = []
        
    def preload_clip_images(
        self, 
        chunk_id: int, 
        clip_name: str, 
        camera: str,
        image_indices: List[int]
    ):
        """
        预加载整个 clip 的某个相机所有图片到内存
        
        Args:
            chunk_id: chunk ID
            clip_name: clip ID  
            camera: 相机名称
            image_indices: 需要加载的图片索引列表
        """
        camera_dir = (
            self.data_root / f"data_sample_chunk{chunk_id}" / 
            "infer" / clip_name / "data" / "camera_images" / camera
        )
        
        for idx in image_indices:
            key = (chunk_id, clip_name, camera, idx)
            if key in self.cache:
                continue
                
            img_path = camera_dir / f"{idx:06d}_small.jpg"
            if not img_path.exists():
                continue
                
            # 读取图片
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # 估算大小
            size = img_array.nbytes
            if self.current_memory + size > self.max_memory_bytes:
                # LRU 淘汰
                self._evict_lru(size)
            
            self.cache[key] = img_array
            self.current_memory += size
            self.access_order.append(key)
    
    def get_image(
        self, 
        chunk_id: int, 
        clip_name: str, 
        camera: str, 
        frame_idx: int
    ) -> Optional[np.ndarray]:
        """获取图片，先查缓存"""
        key = (chunk_id, clip_name, camera, frame_idx)
        
        if key in self.cache:
            # 更新访问顺序
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        # 未命中，加载
        img_path = (
            self.data_root / f"data_sample_chunk{chunk_id}" / 
            "infer" / clip_name / "data" / "camera_images" / camera /
            f"{frame_idx:06d}_small.jpg"
        )
        
        if not img_path.exists():
            return None
            
        img = Image.open(img_path)
        img_array = np.array(img)
        
        size = img_array.nbytes
        if self.current_memory + size > self.max_memory_bytes:
            self._evict_lru(size)
            
        self.cache[key] = img_array
        self.current_memory += size
        self.access_order.append(key)
        
        return img_array
    
    def _evict_lru(self, needed_size: int):
        """LRU 淘汰"""
        while self.access_order and self.current_memory + needed_size > self.max_memory_bytes:
            oldest = self.access_order.pop(0)
            if oldest in self.cache:
                self.current_memory -= self.cache[oldest].nbytes
                del self.cache[oldest]


class AlpamayoSFTDataset(Dataset):
    """
    Alpamayo SFT 数据集
    
    每个样本:
        输入: 16张图片 (4相机×4帧) + 历史轨迹
        输出: CoC 文本 token IDs
    """
    
    # 相机列表
    CAMERAS = [
        'camera_cross_left_120fov',
        'camera_front_wide_120fov',
        'camera_cross_right_120fov', 
        'camera_front_tele_30fov'
    ]
    
    def __init__(
        self,
        infer_result_csv: str,
        tokenizer,  # 用于 tokenize CoC 文本
        data_root: str = "/data01/mikelee/data",
        max_cache_gb: int = 800,
        preload_all: bool = False,
        image_size: Tuple[int, int] = (224, 224),
    ):
        """
        Args:
            infer_result_csv: infer_results_all.csv 路径
            data_root: 数据根目录
            tokenizer: tokenizer 用于编码 CoC
            max_cache_gb: 图片缓存最大内存 (GB)
            preload_all: 是否预加载所有图片到内存
            image_size: 图片 resize 大小
        """
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.image_size = image_size
        
        # 加载索引管理器（合并了 index + CoC）
        self.index_manager = AlpamayoIndexManager(infer_result_csv, data_root)
        self.index_manager.load_all_indices()
        
        # 所有有效样本的 key 列表
        self.sample_keys = list(self.index_manager.index_map.keys())
        logger.info(f"Found {len(self.sample_keys)} valid samples")
        
        # 图片缓存
        self.image_cache = ImageCache(data_root, max_cache_gb)
        
        # 预加载所有图片（如果内存足够）
        if preload_all:
            self._preload_all_images()
    
    def _preload_all_images(self):
        """预加载所有图片到内存"""
        logger.info("Preloading all images into memory... This may take a while.")
        
        # 按 clip 组织，先统计
        clip_to_frames = defaultdict(list)
        for key in self.sample_keys:
            chunk_id, clip_name, frame_id = key
            clip_to_frames[(chunk_id, clip_name)].append(frame_id)
        
        for (chunk_id, clip_name), frame_ids in clip_to_frames.items():
            logger.info(f"Preloading clip {clip_name} with {len(frame_ids)} frames...")
            
            # 统计需要的图片数
            unique_images = set()
            for frame_id in frame_ids:
                info = self.index_manager.get_info(chunk_id, clip_name, frame_id)
                if info:
                    for cam in self.CAMERAS:
                        for i in range(4):
                            unique_images.add((cam, info['image_indices'].get(f'{cam}_f{i}')))
            
            logger.info(f"  {len(unique_images)} unique images to load")
            
            # 按相机加载
            for cam in self.CAMERAS:
                cam_images = [idx for c, idx in unique_images if c == cam]
                self.image_cache.preload_clip_images(chunk_id, clip_name, cam, cam_images)
        
        logger.info(f"Image cache: {self.image_cache.current_memory / 1e9:.2f} GB used")
    
    def __len__(self) -> int:
        return len(self.sample_keys)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回一个训练样本
        
        Returns:
            {
                'images': torch.Tensor,  # (16, C, H, W) 16张图片
                'history': torch.Tensor,  # (16, 11) 历史轨迹
                'labels': torch.Tensor,   # (seq_len,) CoC token IDs
                'input_ids': torch.Tensor, # (seq_len,) 相同，作为输入参考
            }
        """
        chunk_id, clip_name, frame_id = self.sample_keys[idx]
        
        # 获取索引信息（包含 image_indices, ego_idx, cot_result）
        index_info = self.index_manager.get_info(chunk_id, clip_name, frame_id)
        if index_info is None:
            raise ValueError(f"Index not found for {chunk_id}_{clip_name}_{frame_id}")
        
        # 加载16张图片
        images = []
        for cam in self.CAMERAS:
            for i in range(4):
                frame_idx = index_info['image_indices'].get(f'{cam}_f{i}')
                if frame_idx is None:
                    # fallback: 用0
                    frame_idx = 0
                
                img_array = self.image_cache.get_image(
                    chunk_id, clip_name, cam, frame_idx
                )
                
                if img_array is None:
                    # fallback: 黑色图片
                    img_array = np.zeros((*self.image_size, 3), dtype=np.uint8)
                
                # Resize
                img = Image.fromarray(img_array)
                img = img.resize(self.image_size, Image.BILINEAR)
                img_array = np.array(img).transpose(2, 0, 1)  # HWC -> CHW
                images.append(img_array)
        
        images = np.stack(images, axis=0)  # (16, 3, H, W)
        
        # 加载历史轨迹
        ego_idx = index_info['ego_idx']
        history_path = (
            self.data_root / f"data_sample_chunk{chunk_id}" / 
            "infer" / clip_name / "data" / "egomotion" /
            f"frame_{ego_idx:06d}_history.npy"
        )
        
        if history_path.exists():
            history = np.load(history_path)
        else:
            history = np.zeros((16, 11), dtype=np.float32)
        
        # 获取 CoC 文本（直接从 index_info 获取）
        coc_text = index_info.get('cot_result', '')
        
        # Tokenize
        tokens = self.tokenizer(
            coc_text,
            return_tensors='pt',
            padding='max_length',
            max_length=256,
            truncation=True
        )
        
        input_ids = tokens['input_ids'].squeeze(0)
        labels = input_ids.clone()  # SFT 中 labels = input_ids
        
        return {
            'images': torch.from_numpy(images).float(),
            'history': torch.from_numpy(history).float(),
            'input_ids': input_ids,
            'labels': labels,
            'cot_text': coc_text  # 用于 debug
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """DataLoader collate 函数"""
    return {
        'images': torch.stack([b['images'] for b in batch]),
        'history': torch.stack([b['history'] for b in batch]),
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
    }


def create_alpamayo_dataloader(
    infer_result_csv: str,
    tokenizer,
    batch_size: int = 8,
    num_workers: int = 8,
    max_cache_gb: int = 800,
    preload_all: bool = False,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """
    创建 Alpamayo SFT DataLoader
    
    Args:
        infer_result_csv: infer_results_all.csv 路径
        tokenizer: tokenizer
        batch_size: 批次大小
        num_workers: 数据加载线程数
        max_cache_gb: 缓存最大内存
        preload_all: 是否预加载所有图片
        shuffle: 是否 shuffle
    
    Returns:
        DataLoader
    """
    dataset = AlpamayoSFTDataset(
        infer_result_csv=infer_result_csv,
        tokenizer=tokenizer,
        max_cache_gb=max_cache_gb,
        preload_all=preload_all
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs
    )


# 使用示例
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # 配置
    INFER_RESULT_CSV = "/data01/mikelee/infer_result/infer_result_20260424_161448/infer_results_all.csv"
    DATA_ROOT = "/data01/mikelee/data"
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/data01/mikelee/weight/Cosmos-Reason2-2B",
        trust_remote_code=True
    )
    
    # 创建 dataset（不预加载图片，先用 on-demand 模式测试）
    dataset = AlpamayoSFTDataset(
        infer_result_csv=INFER_RESULT_CSV,
        data_root=DATA_ROOT,
        tokenizer=tokenizer,
        max_cache_gb=800,
        preload_all=False  # 先不用预加载，测试 on-demand
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试取一个样本
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Images shape: {sample['images'].shape}")
    print(f"History shape: {sample['history'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"CoC text: {sample['cot_text']}")
    
    # 创建 dataloader
    dataloader = create_alpamayo_dataloader(
        infer_result_csv=INFER_RESULT_CSV,
        tokenizer=tokenizer,
        batch_size=4,
        num_workers=4,
        preload_all=False
    )
    
    # 遍历一个 batch
    batch = next(iter(dataloader))
    print(f"\nBatch images shape: {batch['images'].shape}")
    print(f"Batch history shape: {batch['history'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
