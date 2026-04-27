"""
InferenceIndexDataset
=====================
Manages inference_index_strict.csv files across multiple chunks.
Builds a combined DataFrame and provides random sampling.

Usage:
    index_ds = InferenceIndexDataset(chunks=[0,1,2,...,26])
    df = index_ds.sample(n=100000)  # Random sample 100k frames
"""

import os, glob, pandas as pd
from pathlib import Path
from typing import List, Optional


class InferenceIndexDataset:
    """Dataset that manages inference_index_strict.csv across multiple chunks."""
    
    def __init__(self, base_path: str, chunks: List[int], verbose: bool = True):
        """
        Args:
            base_path: Base data path, e.g. /data01/mikelee/data/
            chunks: List of chunk indices, e.g. [0, 1, 2, ..., 26]
            verbose: Print progress
        """
        self.base_path = Path(base_path)
        self.chunks = chunks
        self.verbose = verbose
        
        # Build combined DataFrame
        self.df = self._build_dataframe()
        
        if verbose:
            print(f"[InferenceIndexDataset] Chunks {chunks}: {len(self.df):,} total frames")
    
    def _build_dataframe(self) -> pd.DataFrame:
        """Scan all chunks and collect all inference_index_strict.csv files."""
        all_records = []
        
        for chunk_idx in self.chunks:
            chunk_dir = self.base_path / f"data_sample_chunk{chunk_idx}" / "infer"
            if not chunk_dir.exists():
                if self.verbose:
                    print(f"  Warning: {chunk_dir} not found, skipping")
                continue
            
            # Find all clip directories
            clip_dirs = [d for d in chunk_dir.iterdir() if d.is_dir()]
            
            for clip_dir in clip_dirs:
                csv_path = clip_dir / "data" / "inference_index_strict.csv"
                if not csv_path.exists():
                    csv_path = clip_dir / "inference_index_strict.csv"  # Try alternate location
                
                if csv_path.exists():
                    try:
                        df_clip = pd.read_csv(csv_path)
                        # Add metadata columns
                        df_clip['chunk'] = chunk_idx
                        df_clip['clip_id'] = clip_dir.name
                        df_clip['clip_dir'] = str(clip_dir)
                        all_records.append(df_clip)
                    except Exception as e:
                        if self.verbose:
                            print(f"  Error reading {csv_path}: {e}")
        
        if not all_records:
            raise ValueError(f"No inference_index_strict.csv found in chunks {self.chunks}")
        
        combined = pd.concat(all_records, ignore_index=True)
        return combined
    
    def sample(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Randomly sample n frames from the dataset.
        
        Args:
            n: Number of frames to sample
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with n sampled rows
        """
        if seed is not None:
            sampled = self.df.sample(n=n, random_state=seed)
        else:
            sampled = self.df.sample(n=n)
        
        return sampled.reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def get_clip_info(self, clip_id: str) -> pd.DataFrame:
        """Get all frames for a specific clip."""
        return self.df[self.df['clip_id'] == clip_id]
    
    def get_chunk_info(self, chunk_idx: int) -> pd.DataFrame:
        """Get all frames for a specific chunk."""
        return self.df[self.df['chunk'] == chunk_idx]


# ==============================================================================
# Convenience function
# ==============================================================================

def build_index_dataset(config_dict, split: str = "train"):
    """Build InferenceIndexDataset from config.
    
    Args:
        config_dict: Config dict with data_path, train_chunks, val_chunks
        split: "train" or "val"
    
    Returns:
        InferenceIndexDataset
    """
    base_path = config_dict.get('data_path', '/data01/mikelee/data/')
    if base_path.endswith('/infer/'):
        base_path = os.path.dirname(os.path.dirname(base_path))
    
    if split == "train":
        chunks = config_dict.get('train_chunks', list(range(27)))
    else:
        chunks = config_dict.get('val_chunks', [27, 28, 29])
    
    return InferenceIndexDataset(base_path=base_path, chunks=chunks)


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    import json
    
    # Test with config
    with open('config_train.json') as f:
        config = json.load(f)
    
    print("Building train dataset...")
    train_ds = build_index_dataset(config, "train")
    print(f"Train: {len(train_ds):,} frames")
    
    print("\nSampling 100 frames...")
    sample = train_ds.sample(100, seed=42)
    print(f"Sample: {len(sample)} rows")
    print(sample[['frame_id', 'ego_idx', 'clip_id', 'chunk']].head())
    
    print("\nBuilding val dataset...")
    val_ds = build_index_dataset(config, "val")
    print(f"Val: {len(val_ds):,} frames")
