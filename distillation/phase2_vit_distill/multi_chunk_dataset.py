"""
Multi-Chunk Dataset for ViT Distillation
========================================
- Training: random 100k frames from chunk0-chunk26 per epoch
- Validation: random 2k frames from chunk27-chunk29 (fixed or random per eval)
"""

import os, glob, random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class MultiChunkImageDataset(Dataset):
    """Dataset that loads images from multiple chunks with random sampling per epoch."""
    
    def __init__(self, base_data_path: str, chunks: list, 
                 num_samples: int = 100000, transform=None, 
                 image_size: int = 224, seed: int = 42):
        """
        Args:
            base_data_path: e.g. /data01/mikelee/data/
            chunks: list of chunk indices, e.g. [0, 1, 2, ..., 26]
            num_samples: number of samples to randomly select per epoch
            transform: torchvision transforms
            image_size: resize size
            seed: random seed for reproducibility
        """
        self.base_path = Path(base_data_path)
        self.chunks = chunks
        self.num_samples = num_samples
        self.transform = transform
        self.image_size = image_size
        self.seed = seed
        
        # Collect ALL image paths from all chunks
        self.all_images = []
        for chunk_idx in chunks:
            chunk_dir = self.base_path / f"data_sample_chunk{chunk_idx}" / "infer"
            if chunk_dir.exists():
                pattern = str(chunk_dir / "**" / "*.jpg")
                images = glob.glob(pattern, recursive=True)
                self.all_images.extend(images)
        
        print(f"[Dataset] Chunks {chunks}: {len(self.all_images):,} total images")
        
        # Randomly sample for this epoch
        self.epoch_images = self._sample_epoch()
    
    def _sample_epoch(self):
        """Randomly sample num_samples images for this epoch."""
        rng = random.Random(self.seed)
        if len(self.all_images) <= self.num_samples:
            return self.all_images
        return rng.sample(self.all_images, self.num_samples)
    
    def resample(self, new_seed=None):
        """Resample for a new epoch. Call at the start of each epoch."""
        if new_seed is not None:
            self.seed = new_seed
        else:
            self.seed += 1  # increment seed for different sampling
        self.epoch_images = self._sample_epoch()
        print(f"[Dataset] Resampled: {len(self.epoch_images):,} images (seed={self.seed})")
    
    def __len__(self):
        return len(self.epoch_images)
    
    def __getitem__(self, idx):
        img_path = self.epoch_images[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            # Return a gray image if loading fails
            img = Image.new('RGB', (self.image_size, self.image_size), color='gray')
        
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])(img)
        
        return {"pixel_values": img, "path": img_path}


def build_dataloaders(config_dict, seed=42):
    """Build training and validation dataloaders.
    
    Args:
        config_dict: training config dict with data_path, batch_size_per_gpu, etc.
        seed: random seed
    
    Returns:
        train_dataset, val_dataset
    """
    base_path = config_dict.get('data_path', '/data01/mikelee/data/')
    if base_path.endswith('/infer/'):
        base_path = os.path.dirname(os.path.dirname(base_path))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Training: chunk0-chunk26 (27 chunks)
    train_chunks = config_dict.get('train_chunks', list(range(27)))
    train_dataset = MultiChunkImageDataset(
        base_data_path=base_path,
        chunks=train_chunks,
        num_samples=config_dict.get('samples_per_epoch', 100000),
        transform=transform,
        seed=seed,
    )
    
    # Validation: chunk27-chunk29 (3 chunks)
    val_chunks = config_dict.get('val_chunks', [27, 28, 29])
    val_dataset = MultiChunkImageDataset(
        base_data_path=base_path,
        chunks=val_chunks,
        num_samples=config_dict.get('val_samples', 2000),
        transform=transform,
        seed=seed + 999,  # different seed to avoid overlap
    )
    
    return train_dataset, val_dataset
