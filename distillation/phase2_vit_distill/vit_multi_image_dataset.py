from inference_index_dataset import InferenceIndexDataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import torch
import pandas as pd
import numpy as np


class ViTMultiImageDataset:
    """
    Dataset for ViT Distillation that loads 16 images per frame.
    
    Uses InferenceIndexDataset to manage CSV files and returns:
    - pixel_values: [16, 3, H, W] - 16 images for one frame
    - metadata: frame_id, clip_id, chunk, etc.
    """
    
    def __init__(self, base_path, chunks, num_samples=100000, 
                 image_size=224, seed=42):
        self.index_ds = InferenceIndexDataset(base_path, chunks, verbose=False)
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Camera name mapping
        self.cameras = [
            'camera_cross_left_120fov',
            'camera_front_wide_120fov', 
            'camera_cross_right_120fov',
            'camera_front_tele_30fov'
        ]
        
        self.current_sample = self.index_ds.sample(num_samples, seed=seed)
        print(f"[ViTMultiImageDataset] Loaded {len(self.current_sample):,} frames")
    
    def resample(self, new_seed=None):
        """Resample frames for new epoch."""
        if new_seed is not None:
            self.seed = new_seed
        else:
            self.seed += 1
        self.current_sample = self.index_ds.sample(self.num_samples, seed=self.seed)
        print(f"[ViTMultiImageDataset] Resampled {len(self.current_sample):,} frames (seed={self.seed})")
    
    def __len__(self):
        return len(self.current_sample)
    
    def __getitem__(self, idx):
        row = self.current_sample.iloc[idx]
        
        # Build clip directory path
        chunk_dir = Path(self.index_ds.base_path) / f"data_sample_chunk{int(row['chunk'])}" / "infer"
        clip_dir = chunk_dir / str(row['clip_id']) / "data"
        
        # Load 16 images (4 cameras × 4 frames)
        images = []
        for cam in self.cameras:
            for f in range(4):  # f0, f1, f2, f3
                idx_col = f"{cam}_f{f}_idx"
                img_idx = int(row[idx_col])
                
                # Image path: camera_images/{camera_name}/{idx:06d}_small.jpg
                img_path = clip_dir / "camera_images" / cam.replace('camera_', '') / f"{img_idx:06d}_small.jpg"
                
                try:
                    img = Image.open(img_path).convert('RGB')
                except:
                    img = Image.new('RGB', (self.image_size, self.image_size), color='gray')
                
                img_tensor = self.transform(img)
                images.append(img_tensor)
        
        # Stack to [16, 3, H, W]
        pixel_values = torch.stack(images, dim=0)
        
        return {
            "pixel_values": pixel_values,  # [16, 3, H, W]
            "frame_id": int(row['frame_id']),
            "clip_id": str(row['clip_id']),
            "chunk": int(row['chunk']),
        }


def build_vit_dataloaders(config_dict, seed=42):
    """Build train/val datasets for ViT training.
    
    Args:
        config_dict: Config with data_path, train_chunks, val_chunks, etc.
        seed: Random seed
    
    Returns:
        train_dataset, val_dataset
    """
    base_path = config_dict.get('data_path', '/data01/mikelee/data/')
    if base_path.endswith('/infer/'):
        base_path = base_path.rsplit('/', 2)[0]
    
    train_ds = ViTMultiImageDataset(
        base_path=base_path,
        chunks=config_dict.get('train_chunks', list(range(27))),
        num_samples=config_dict.get('samples_per_epoch', 100000),
        image_size=224,
        seed=seed,
    )
    
    val_ds = ViTMultiImageDataset(
        base_path=base_path,
        chunks=config_dict.get('val_chunks', [27, 28, 29]),
        num_samples=config_dict.get('val_samples', 2000),
        image_size=224,
        seed=seed + 999,
    )
    
    return train_ds, val_ds


# Test
if __name__ == "__main__":
    import json
    
    with open('config_train.json') as f:
        config = json.load(f)
    
    print("Building train dataset...")
    train_ds = build_vit_dataloaders(config, seed=42)[0]
    print(f"Train: {len(train_ds):,} frames")
    
    print("\nLoading first sample...")
    sample = train_ds[0]
    print(f"pixel_values shape: {sample['pixel_values'].shape}")  # [16, 3, 224, 224]
    print(f"frame_id: {sample['frame_id']}")
    print(f"clip_id: {sample['clip_id']}")
    print(f"chunk: {sample['chunk']}")
    
    print("\nBuilding val dataset...")
    val_ds = build_vit_dataloaders(config, seed=42)[1]
    print(f"Val: {len(val_ds):,} frames")
