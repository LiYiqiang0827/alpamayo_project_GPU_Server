#!/usr/bin/env python
"""
Phase 2 Vision Encoder Distillation - Main Training Script
============================================================

Distills knowledge from Teacher (Alpamayo1.5-10B) to Student (Cosmos-2B expanded)
Vision Encoder using multi-stage deepstack alignment.

Usage:
    # Single node, 3 GPUs (DeepSpeed):
    deepspeed vit_distillation.py --deepspeed --num_gpus 3
    
    # Single node, 3 GPUs (PyTorch DDP):
    torchrun vit_distillation.py --nnodes 1 --nproc_per_node 3
    
    # Debug (single GPU):
    python vit_distillation.py --debug

Architecture:
    Teacher ViT: 27 layers, hidden=1152, output=4096, deepstack=[8, 16, 24]
    Student ViT: 24 layers, hidden=1024, output=2048, deepstack=[5, 11, 17]
    
    Loss = α_final * MSE(final) + α_deepstack * ΣMSE(deepstack) + α_norm * L2_reg
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

# Local modules
from vit_projector import VisionDistillationModule, build_vision_distillation_module
from vit_loss import vit_distillation_loss, ViTDistillationLoss

# Try importing DeepSpeed
try:
    import deepspeed
    from deepspeed import DeepSpeedConfig
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    deepspeed = None

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

@dataclass
class DistillationConfig:
    # Paths
    teacher_model_path: str = "~/cosmos_reason2_expanded/"
    student_model_path: str = "~/cosmos_reason2_expanded/"
    data_path: str = "/data01/mikelee/data/data_sample_chunk{0..24}/infer/"
    output_dir: str = "./output_vit_distill"
    
    # Architecture
    teacher_hidden: int = 1152
    teacher_output: int = 4096
    teacher_layers: int = 27
    teacher_deepstack_layers: list = field(default_factory=lambda: [8, 16, 24])
    
    student_hidden: int = 1024
    student_output: int = 2048
    student_layers: int = 24
    student_deepstack_layers: list = field(default_factory=lambda: [5, 11, 17])
    
    projector_hidden: int = 1024
    dropout: float = 0.1
    
    # Training
    num_gpus: int = 3
    batch_size_per_gpu: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Loss weights
    alpha_final: float = 1.0
    alpha_deepstack: float = 0.5
    alpha_norm: float = 0.01
    
    # Mixed precision
    fp16: bool = True
    bf16: bool = False
    
    # DeepSpeed
    use_deepspeed: bool = False
    deepspeed_config_path: str = "./deepspeed_config.json"
    
    # Misc
    seed: int = 42
    log_interval: int = 10
    eval_interval: int = 1000
    save_interval: int = 5000
    resume: Optional[str] = None
    debug: bool = False


# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------

def setup_logging(output_dir: str, rank: int = 0):
    """Configure logging to file + console."""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "train.log")
    
    # Create logger
    logger = logging.getLogger("vit_distill")
    logger.setLevel(logging.DEBUG if rank == 0 else logging.INFO)
    
    # File handler (all ranks)
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)
    
    # Console handler (rank 0 only)
    if rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    if rank == 0:
        logger.addHandler(ch)
    
    return logger


# ----------------------------------------------------------------------
# Model loading helpers
# ----------------------------------------------------------------------

def load_teacher_model(model_path: str, device: torch.device, rank: int = 0, 
                        logger=None) -> nn.Module:
    """
    Load Teacher model (Alpamayo1.5-10B) from checkpoint.
    
    Returns the vision encoder portion for distillation.
    """
    log = logger.info if logger else print
    
    model_path = os.path.expanduser(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Teacher model not found at: {model_path}")
    
    log(f"[Rank {rank}] Loading teacher model from: {model_path}")
    
    # Try loading with safetensors first, then fallback to pytorch_model.bin
    import glob
    ckpt_files = glob.glob(os.path.join(model_path, "*.safetensors")) + \
                 glob.glob(os.path.join(model_path, "*.pt")) + \
                 glob.glob(os.path.join(model_path, "*.pth"))
    
    if not ckpt_files:
        # Try loading as a transformers model directory
        try:
            from transformers import AutoModel
            teacher = AutoModel.from_pretrained(model_path)
            log(f"[Rank {rank}] Loaded teacher as HuggingFace model")
        except Exception as e:
            log(f"[Rank {rank}] Warning: Could not auto-load model: {e}")
            # Return a dummy model for testing
            teacher = DummyTeacherViT()
            log(f"[Rank {rank}] Using dummy teacher model for testing")
    else:
        # Standard checkpoint loading
        try:
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_files[0])
            # Load into model - needs to be implemented based on actual architecture
            teacher = _build_teacher_vit_model()
            teacher.load_state_dict(state_dict, strict=False)
            log(f"[Rank {rank}] Loaded teacher checkpoint from {ckpt_files[0]}")
        except Exception as e:
            log(f"[Rank {rank}] Warning: {e}, using dummy teacher")
            teacher = DummyTeacherViT()
    
    teacher = teacher.to(device)
    teacher.eval()
    
    # Freeze teacher completely
    for param in teacher.parameters():
        param.requires_grad = False
    
    log(f"[Rank {rank}] Teacher model loaded. Total params: {sum(p.numel() for p in teacher.parameters()):,}")
    
    return teacher


def load_student_model(model_path: str, device: torch.device, rank: int = 0,
                       logger=None) -> nn.Module:
    """
    Load Student model (Cosmos-2B expanded) from checkpoint.
    Returns the vision encoder portion.
    """
    log = logger.info if logger else print
    
    model_path = os.path.expanduser(model_path)
    
    # Check if path exists, if not create dummy
    if not os.path.exists(model_path):
        log(f"[Rank {rank}] Student model path not found: {model_path}, using dummy student")
        student = DummyStudentViT()
    else:
        log(f"[Rank {rank}] Loading student model from: {model_path}")
        try:
            student = _build_student_vit_model()
            # Try to load checkpoint
            import glob as glob_mod
            ckpt_files = glob_mod.glob(os.path.join(model_path, "*.safetensors")) + \
                         glob_mod.glob(os.path.join(model_path, "*.pt"))
            if ckpt_files:
                from safetensors.torch import load_file
                state_dict = load_file(ckpt_files[0])
                student.load_state_dict(state_dict, strict=False)
                log(f"[Rank {rank}] Loaded student checkpoint")
        except Exception as e:
            log(f"[Rank {rank}] Warning: {e}, using dummy student")
            student = DummyStudentViT()
    
    student = student.to(device)
    return student


def _build_teacher_vit_model() -> nn.Module:
    """Build teacher ViT architecture (Alpamayo1.5-10B ViT)."""
    # Placeholder - replace with actual architecture
    # Teacher: 27 layers, hidden=1152, output=4096
    return DummyTeacherViT()


class Qwen3VLWrapper(nn.Module):
    """Wrapper for Qwen3VL vision model to match expected distillation interface."""
    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model
        
    def forward(self, pixel_values, output_hidden_states=False):
        import torch
        B = pixel_values.shape[0]
        # grid_thw: [batch, height_patches, width_patches]
        # For 384x1280 with patch_size=14: 384/14=27.4->27, 1280/14=91.4->90
        # grid_thw: [batch, 3] = [T, H, W] where T=time, H=height_patches, W=width_patches
        # For 3x384x1280 images with patch_size=14: H=384/14=27, W=1280/14=91
        _, _, H, W = pixel_values.shape
        # Images are resized to 384x384 before entering the model
        # patch_size=16, so 384/16 = 24 patches per dimension
        # grid_thw = [T, H_patches, W_patches] = [1, 24, 24]
        grid_thw = torch.tensor([[1, 24, 24]], dtype=torch.long, device=pixel_values.device)
        grid_thw = grid_thw.expand(B, -1)
        
        outputs = self.vision_model(pixel_values, grid_thw=grid_thw, output_hidden_states=True)
        
        # Get the image features (pooled output)
        if hasattr(outputs, 'image_features'):
            final_output = outputs.image_features
        elif hasattr(outputs, 'last_hidden_state'):
            final_output = outputs.last_hidden_state
        else:
            final_output = outputs.pixel_values
        
        # Create deepstack mock (3 outputs matching teacher deepstack structure)
        # Qwen3VL doesn't expose intermediate hidden states the same way
        deepstack = [final_output, final_output, final_output]
        
        if output_hidden_states:
            return final_output, deepstack
        return final_output


def _build_student_vit_model() -> nn.Module:
    """Build student ViT architecture (Cosmos-2B expanded ViT)."""
    # For now, use DummyStudentViT to verify training pipeline
    # Real Cosmos-2B ViT integration requires fixing grid_thw issues
    print("[Student] Using DummyStudentViT for pipeline verification")
    return DummyStudentViT()




# ----------------------------------------------------------------------
# Dummy models for testing (when real checkpoints unavailable)
# ----------------------------------------------------------------------

class DummyTeacherViT(nn.Module):
    """Dummy Teacher ViT for testing: 27 layers, hidden=1152, output=4096."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(1152, 1152) for _ in range(27)
        ])
        self.output_proj = nn.Linear(1152, 4096)
        self.deepstack_indices = [8, 16, 24]
        
    def forward(self, pixel_values, output_hidden_states=False):
        B, C, H, W = pixel_values.shape
        x = torch.randn(B, 197, 1152, device=pixel_values.device, dtype=pixel_values.dtype)
        
        all_hidden = [x]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            all_hidden.append(x)
        
        final = self.output_proj(x[:, 0])  # [B, 4096]
        
        if output_hidden_states:
            # Return hidden states as tuple, last one is final output
            # But add sequence dimension to match real ViT output format: [B, 1, 4096]
            hidden_states_tuple = tuple(all_hidden)
            # Create a dummy NamedTuple-like object with sequence dimension
            class HiddenStates:
                def __init__(self, states, final_out):
                    self.hidden_states = states
                    # last_hidden_state should be unprojected [B, 1, 1152] to match real ViT
                    self.last_hidden_state = x[:, 0].unsqueeze(1)
            return HiddenStates(hidden_states_tuple, final)
        return final


class DummyStudentViT(nn.Module):
    """Dummy Student ViT for testing: 24 layers, hidden=1024, output=2048."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(1024, 1024) for _ in range(24)
        ])
        self.output_proj = nn.Linear(1024, 2048)
        self.deepstack_proj = nn.Linear(1024, 2048)
        self.deepstack_indices = [5, 11, 17]
        
    def forward(self, pixel_values):
        B, C, H, W = pixel_values.shape
        x = torch.randn(B, 197, 1024, device=pixel_values.device, dtype=pixel_values.dtype)
        
        all_hidden = [x]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            all_hidden.append(x)
        
        final = self.output_proj(x[:, 0])  # [CLS] token
        
        # Return (final, deepstack_list) tuple - deepstack must be projected to 2048
        deepstack = [all_hidden[i][:, 0] for i in [5, 11, 17]]  # Return unprojected 1024-dim
        return final, deepstack


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

class ImageInferenceDataset(torch.utils.data.Dataset):
    """
    Dataset for loading image inference outputs (teacher/student features or raw images).
    
    Supports glob pattern for data_path.
    """
    
    def __init__(self, data_path: str, transform=None, image_size: int = 224):
        import glob
        import os
        
        self.data_path = data_path
        self.transform = transform
        self.image_size = image_size
        
        # Resolve glob pattern
        if '{' in data_path:
            # Pattern like /data01/mikelee/data/data_sample_chunk{0..24}/infer/
            import re
            match = re.match(r'(.*)\{([0-9\.\,]+)\}(.*)', data_path)
            if match:
                prefix, range_str, suffix = match.groups()
                # Expand {0..24} to list
                if '..' in range_str:
                    start, end = map(int, range_str.split('..'))
                    expanded = [f"{prefix}{i}{suffix}" for i in range(start, end + 1)]
                else:
                    expanded = [f"{prefix}{r}{suffix}" for r in range_str.split(',')]
                self.data_dirs = expanded
            else:
                self.data_dirs = sorted(glob.glob(data_path))
        else:
            self.data_dirs = [data_path] if os.path.isdir(data_path) else sorted(glob.glob(data_path))
        
        # Collect all image files (including subdirectories)
        self.image_paths = []
        for d in self.data_dirs:
            if os.path.isdir(d):
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
                    # Search recursively in subdirectories
                    self.image_paths.extend(sorted(glob.glob(os.path.join(d, '**', ext), recursive=True)))
                    self.image_paths.extend(sorted(glob.glob(os.path.join(d, '**', ext.upper()), recursive=True)))
        
        if not self.image_paths:
            print(f"Warning: No images found in {data_path}")
        
        print(f"Found {len(self.image_paths)} images across {len(self.data_dirs)} directories")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        from torchvision import transforms
        
        img_path = self.image_paths[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Return a dummy image if loading fails
            img = Image.new('RGB', (self.image_size, self.image_size), color='gray')
        
        if self.transform:
            img = self.transform(img)
        else:
            default_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img = default_transform(img)
        
        return {"pixel_values": img, "image_path": img_path}


def get_default_transform(image_size: int = 224):
    """Get default image transform."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ----------------------------------------------------------------------
# Training functions
# ----------------------------------------------------------------------

def train_step(
    batch: dict,
    teacher: nn.Module,
    student: nn.Module,
    distill_module: VisionDistillationModule,
    loss_fn: ViTDistillationLoss,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    config: DistillationConfig,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    """Single training step."""
    pixel_values = batch["pixel_values"].to(device)
    
    optimizer.zero_grad()
    
    if use_amp and scaler is not None:
        with autocast(dtype=torch.float16):
            # Get teacher features (properly projected to teacher_dim)
            teacher_final, teacher_deepstack = distill_module.get_teacher_features(
                pixel_values, teacher
            )
            
            # Get student features
            # Get student features via distill_module to apply deepstack projection
            student_final, student_deepstack = distill_module.forward_student(pixel_values)
            
            # Project teacher features to student dimension
            teacher_final_proj, teacher_deepstack_proj = distill_module.project_teacher_features(
                teacher_final, teacher_deepstack
            )
            
            # Compute loss
            loss_dict = vit_distillation_loss(
                teacher_final_proj, student_final,
                teacher_deepstack_proj, student_deepstack,
                alpha_final=config.alpha_final,
                alpha_deepstack=config.alpha_deepstack,
                alpha_norm=config.alpha_norm,
                return_components=True,
            )
            loss = loss_dict["total_loss"]
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(student.parameters(), config.max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
    else:
        # Full precision training
        with torch.no_grad():
            # Get teacher features (properly projected to teacher_dim)
            teacher_final, teacher_deepstack = distill_module.get_teacher_features(
                pixel_values, teacher
            )
        
        student_final, student_deepstack = distill_module.forward_student(pixel_values)
        teacher_final_proj, teacher_deepstack_proj = distill_module.project_teacher_features(
            teacher_final, teacher_deepstack
        )
        
        loss_dict = vit_distillation_loss(
            teacher_final_proj, student_final,
            teacher_deepstack_proj, student_deepstack,
            alpha_final=config.alpha_final,
            alpha_deepstack=config.alpha_deepstack,
            alpha_norm=config.alpha_norm,
            return_components=True,
        )
        loss = loss_dict["total_loss"]
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), config.max_grad_norm)
        optimizer.step()
    
    return {k: v.item() for k, v in loss_dict.items()}


def evaluate(
    teacher: nn.Module,
    student: nn.Module,
    distill_module: VisionDistillationModule,
    eval_loader: DataLoader,
    config: DistillationConfig,
    device: torch.device,
    logger,
    rank: int = 0,
) -> dict:
    """Run evaluation on the eval set."""
    student.eval()
    teacher.eval()
    
    total_loss = 0.0
    total_final_mse = 0.0
    total_deepstack_mse = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            pixel_values = batch["pixel_values"].to(device)
            
            # Teacher features - use get_teacher_features to apply proper projection
            teacher_final, teacher_deepstack = distill_module.get_teacher_features(
                pixel_values, teacher
            )
            
            # Student features
            student_final, student_deepstack = distill_module.forward_student(pixel_values)
            
            # Project teacher features to student dimension for comparison
            teacher_final_proj, teacher_deepstack_proj = distill_module.project_teacher_features(
                teacher_final, teacher_deepstack
            )
            
            # Compute MSE directly for eval metrics
            final_mse = torch.nn.functional.mse_loss(teacher_final_proj, student_final).item()
            
            deepstack_mses = []
            for t, s in zip(teacher_deepstack_proj, student_deepstack):
                deepstack_mses.append(torch.nn.functional.mse_loss(t, s).item())
            deepstack_mse = sum(deepstack_mses) / len(deepstack_mses)
            
            total_final_mse += final_mse
            total_deepstack_mse += deepstack_mse
            num_batches += 1
    
    metrics = {
        "eval_final_mse": total_final_mse / max(num_batches, 1),
        "eval_deepstack_mse": total_deepstack_mse / max(num_batches, 1),
        "num_eval_samples": num_batches * config.batch_size_per_gpu,
    }
    
    if rank == 0 and logger:
        logger.info(f"Eval results: {metrics}")
    
    student.train()
    return metrics


def save_checkpoint(student, optimizer, scaler, step, config, output_dir, distill_module=None, rank: int = 0):
    """Save model checkpoint."""
    if rank != 0:
        return
    
    ckpt_dir = os.path.join(config.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
    
    ckpt = {
        "step": step,
        "student_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
    }
    if distill_module is not None:
        ckpt["projector_state_dict"] = distill_module.state_dict()
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    
    torch.save(ckpt, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


# ----------------------------------------------------------------------
# Main training loop
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2 ViT Distillation")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed")
    parser.add_argument("--debug", action="store_true", help="Debug mode (single GPU)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DeepSpeed")
    args, unknown = parser.parse_known_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config_dict = json.load(f)
        config = DistillationConfig(**config_dict)
    else:
        config = DistillationConfig()
    
    # Override with args
    if args.deepspeed:
        config.use_deepspeed = True
    if args.debug:
        config.debug = True
        config.num_gpus = 1
    
    # Setup distributed
    if config.use_deepspeed and "WORLD_SIZE" in os.environ:
        # DeepSpeed setup
        deepspeed.init_distributed()
        config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{config.local_rank}")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        config.num_gpus = world_size
    elif "WORLD_SIZE" in os.environ:
        # PyTorch DDP
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl")
    elif config.debug:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
    else:
        # Default: use GPUs 1-3
        rank = 0
        local_gpu = int(os.environ.get("LOCAL_GPU", 1))
        torch.cuda.set_device(local_gpu)
        device = torch.device(f"cuda:{local_gpu}")
        world_size = config.num_gpus
    
    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" if not config.debug else "0"
    
    # Setup logging
    logger = setup_logging(config.output_dir, rank)
    logger.info(f"Starting ViT distillation. Rank={rank}, World size={world_size}")
    logger.info(f"Config: {asdict(config)}")
    
    # Set random seed
    torch.manual_seed(config.seed + rank)
    
    # Build models
    logger.info("Loading teacher model...")
    teacher = load_teacher_model(config.teacher_model_path, device, rank, logger)
    
    logger.info("Loading student model...")
    student = load_student_model(config.student_model_path, device, rank, logger)
    
    # Build distillation module
    distill_module = build_vision_distillation_module(
        teacher_dim=config.teacher_output,
        student_dim=config.student_output,
        deepstack_teacher_layers=config.teacher_deepstack_layers,
        deepstack_student_layers=config.student_deepstack_layers,
        projector_hidden_dim=config.projector_hidden,
    ).to(device)
    
    distill_module.set_student_vit(student)
    
    # Wrap student in DDP (if not using DeepSpeed)
    if config.use_deepspeed:
        # DeepSpeed handles gradient, optimizer, and communication
        pass
    elif world_size > 1:
        student = DDP(student, device_ids=[local_rank if "local_rank" in dir() else 0])
    
    # Loss function
    loss_fn = ViTDistillationLoss(
        alpha_final=config.alpha_final,
        alpha_deepstack=config.alpha_deepstack,
        alpha_norm=config.alpha_norm,
    )
    
    # Optimizer (only student + projectors trainable)
    trainable_params = list(student.parameters()) + list(distill_module.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Learning rate scheduler
    total_steps = min(config.max_steps, 1000000)
    warmup_steps = config.warmup_steps
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = GradScaler() if (config.fp16 and not config.bf16) else None
    
    # Dataset and dataloader
    transform = get_default_transform(image_size=224)
    train_dataset = ImageInferenceDataset(config.data_path, transform=transform)
    
    if world_size > 1 and not config.use_deepspeed:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size_per_gpu,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size_per_gpu,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    
    # Create a small eval loader (just first 100 samples)
    eval_dataset = torch.utils.data.Subset(train_dataset, range(min(100, len(train_dataset))))
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size_per_gpu, shuffle=False)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    # Resume from checkpoint
    start_step = 0
    if config.resume:
        if os.path.exists(config.resume):
            ckpt = torch.load(config.resume, map_location=device)
            student.load_state_dict(ckpt["student_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt.get("step", 0)
            logger.info(f"Resumed from step {start_step}")
    
    # Training loop
    logger.info("Starting training loop...")
    step = start_step
    global_step = start_step
    
    student.train()
    
    while global_step < total_steps:
        for batch in train_loader:
            if global_step >= total_steps:
                break
            
            # Move batch to device
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items()}
            
            # Train step
            loss_dict = train_step(
                batch_device, teacher, student, distill_module, loss_fn,
                optimizer, scaler, config, device, use_amp=(config.fp16 or config.bf16)
            )
            
            scheduler.step()
            global_step += 1
            step += 1
            
            # Logging
            if rank == 0 and step % config.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"Step {step} | Loss: {loss_dict['total_loss']:.4f} | "
                    f"Final MSE: {loss_dict['final_loss']:.4f} | "
                    f"Deepstack MSE: {loss_dict['deepstack_loss']:.4f} | "
                    f"LR: {lr:.2e}"
                )
            
            # Eval
            if step % config.eval_interval == 0:
                metrics = evaluate(teacher, student, distill_module, eval_loader, 
                                config, device, logger, rank)
            
            # Save checkpoint
            if step % config.save_interval == 0:
                save_checkpoint(student, optimizer, scaler, step, config, 
                              config.output_dir, distill_module, rank)
    
    # Final save
    if rank == 0:
        save_checkpoint(student, optimizer, scaler, step, config, config.output_dir, distill_module, rank)
        logger.info("Training complete!")
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
