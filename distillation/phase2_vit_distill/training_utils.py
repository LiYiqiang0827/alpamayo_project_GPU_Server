import os
import json
import time
from pathlib import Path

def setup_training_output(config):
    """Setup timestamped training output directory with logging.
    
    Creates:
        /gpfs-data/mikelee/distillation_output/
            checkpoint_20260427_165703/          <- timestamped root
                training.log                     <- main log file
                training_config.json             <- saved config
                best_model/                      <- best checkpoint
                    checkpoint_best.pt
                epoch_0/                         <- epoch checkpoints
                    checkpoint_step_5000.pt
                    checkpoint_step_10000.pt
                epoch_1/
                    ...
    """
    # Create timestamped directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(config['output_dir']) / f"checkpoint_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Update config to use this directory
    config['output_dir'] = str(output_root)
    
    # Save config
    config_path = output_root / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return output_root


def setup_logger(log_file):
    """Setup file logger that writes to both file and console."""
    import logging
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Logger
    logger = logging.getLogger('vit_distillation')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_training_header(logger, config):
    """Log training header with all metadata."""
    header = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║           Alpamayo 1.5 ViT Vision Encoder Distillation                        ║
║           Teacher: Alpamayo-1.5-10B  →  Student: Cosmos-Reason2-2B          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

[Model Configuration]
  Teacher Model:  Alpamayo-1.5-10B (NVIDIA)
    Path: {teacher_path}
    Architecture: 27 layers, hidden=1152, output=4096
    Deepstack layers: [8, 16, 24]
    
  Student Model:  Cosmos-Reason2-2B / Qwen3.5-VL-2B
    Base Path: {student_path}
    Architecture: 24 layers, hidden=1024, intermediate=4096, output=2048
    Deepstack layers: [5, 11, 17]
    Per-frame images: 16 (4 cameras × 4 temporal frames)

[Training Configuration]
  Dataset:
    Train chunks: {train_chunks} (chunk 0-26)
    Val chunks: {val_chunks} (chunk 27-29)
    Samples per epoch: {samples_per_epoch:,} frames
    Val samples: {val_samples:,} frames
    Total train images: ~{total_train_images:,} (all frames across chunks)
  
  Hyperparameters:
    Learning rate: {lr}
    Batch size per GPU: {batch_size}
    Number of GPUs: {num_gpus}
    Epochs: {num_epochs}
    Max steps: {max_steps:,}
    Warmup steps: {warmup_steps:,}
    Save interval: {save_interval} steps
    Eval interval: {eval_interval} steps
    
  Loss Weights:
    alpha_final: {alpha_final}
    alpha_deepstack: {alpha_deepstack}
    alpha_norm: {alpha_norm}

[Training Start]
  Timestamp: {timestamp}
  Output directory: {output_dir}
  
═══════════════════════════════════════════════════════════════════════════════
""".format(
        teacher_path=config['teacher_model_path'],
        student_path=config['student_model_path'],
        train_chunks=config.get('train_chunks', list(range(27))),
        val_chunks=config.get('val_chunks', [27, 28, 29]),
        samples_per_epoch=config.get('samples_per_epoch', 100000),
        val_samples=config.get('val_samples', 2000),
        total_train_images=372258,  # From our dataset scan
        lr=config.get('learning_rate', 0.0001),
        batch_size=config.get('batch_size_per_gpu', 4),
        num_gpus=config.get('num_gpus', 3),
        num_epochs=config.get('num_epochs', 10),
        max_steps=config.get('max_steps', 100000),
        warmup_steps=config.get('warmup_steps', 1000),
        save_interval=config.get('save_interval', 5000),
        eval_interval=config.get('eval_interval', 5000),
        alpha_final=config.get('alpha_final', 1.0),
        alpha_deepstack=config.get('alpha_deepstack', 0.5),
        alpha_norm=config.get('alpha_norm', 0.01),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        output_dir=config['output_dir']
    )
    
    logger.info(header)


def save_best_checkpoint(student_state, projector_state, optimizer_state, 
                        step, metrics, output_dir, rank=0):
    """Save best checkpoint to best_model/ directory."""
    if rank != 0:
        return
    
    best_dir = Path(output_dir) / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    
    import torch
    checkpoint = {
        'step': step,
        'student_state_dict': student_state,
        'projector_state_dict': projector_state,
        'optimizer_state_dict': optimizer_state,
        'metrics': metrics,
        'timestamp': time.strftime("%Y%m%d_%H%M%S"),
    }
    
    torch.save(checkpoint, best_dir / "checkpoint_best.pt")


class TrainingTracker:
    """Track training metrics and identify best checkpoint."""
    
    def __init__(self):
        self.best_val_loss = float('inf')
        self.best_step = 0
        self.history = []
    
    def update(self, step, train_loss, val_metrics):
        """Update tracker with new metrics."""
        val_loss = val_metrics.get('eval_final_mse', float('inf'))
        
        record = {
            'step': step,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
        }
        self.history.append(record)
        
        # Check if this is the best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_step = step
            return True  # Is best
        return False
    
    def get_summary(self):
        """Get training summary."""
        if not self.history:
            return "No training history"
        
        return f"""
Training Summary:
  Total steps: {self.history[-1]['step']}
  Best step: {self.best_step}
  Best val loss: {self.best_val_loss:.6f}
  Final train loss: {self.history[-1]['train_loss']:.6f}
  Final val loss: {self.history[-1]['val_loss']:.6f}
"""
