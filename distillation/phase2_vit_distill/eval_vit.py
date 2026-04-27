#!/usr/bin/env python
"""
Phase 2 Vision Encoder Distillation - Evaluation Script
=========================================================

Evaluates the distilled student ViT against the teacher using:
- MSE loss on final output
- MSE loss on deepstack features
- Feature alignment check (cosine similarity)
- Per-layer deepstack analysis

Usage:
    python eval_vit.py --checkpoint ./output_vit_distill/checkpoints/checkpoint_step_5000.pt
    
    # Compare two checkpoints:
    python eval_vit.py --checkpoint1 ckpt1.pt --checkpoint2 ckpt2.pt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Local modules
from vit_projector import VisionDistillationModule, build_vision_distillation_module
from vit_distillation import (
    load_teacher_model, load_student_model, 
    ImageInferenceDataset, get_default_transform,
    DummyTeacherViT, DummyStudentViT,
)


def compute_mse_metrics(
    teacher_final: torch.Tensor,
    student_final: torch.Tensor,
    teacher_deepstack: list[torch.Tensor],
    student_deepstack: list[torch.Tensor],
) -> dict:
    """Compute comprehensive MSE metrics between teacher and student features."""
    
    # Final output MSE
    final_mse = F.mse_loss(teacher_final, student_final).item()
    
    # Deepstack MSE per layer
    deepstack_mse = []
    for i, (t, s) in enumerate(zip(teacher_deepstack, student_deepstack)):
        mse = F.mse_loss(t, s).item()
        deepstack_mse.append(mse)
    
    avg_deepstack_mse = sum(deepstack_mse) / len(deepstack_mse) if deepstack_mse else 0.0
    
    # Per-dimension MSE (final output)
    per_dim_mse = F.mse_loss(teacher_final, student_final, reduction='none').mean(dim=0)
    
    # Mean/Std of per-dimension MSE
    dim_mse_mean = per_dim_mse.mean().item()
    dim_mse_std = per_dim_mse.std().item()
    
    return {
        "final_mse": final_mse,
        "deepstack_mse_avg": avg_deepstack_mse,
        "deepstack_mse_per_layer": deepstack_mse,
        "per_dim_mse_mean": dim_mse_mean,
        "per_dim_mse_std": dim_mse_std,
    }


def compute_alignment_metrics(
    teacher_final: torch.Tensor,
    student_final: torch.Tensor,
    teacher_deepstack: list[torch.Tensor],
    student_deepstack: list[torch.Tensor],
) -> dict:
    """Compute feature alignment metrics (cosine similarity, correlation)."""
    
    # Final output cosine similarity
    cos_sim_final = F.cosine_similarity(teacher_final, student_final, dim=-1)
    cos_sim_final_mean = cos_sim_final.mean().item()
    cos_sim_final_std = cos_sim_final.std().item()
    
    # Deepstack cosine similarity per layer
    deepstack_cos_sim = []
    for i, (t, s) in enumerate(zip(teacher_deepstack, student_deepstack)):
        cos_sim = F.cosine_similarity(t, s, dim=-1)
        deepstack_cos_sim.append({
            "layer": i,
            "cos_sim_mean": cos_sim.mean().item(),
            "cos_sim_std": cos_sim.std().item(),
        })
    
    # Average cosine similarity across deepstack
    avg_cos_sim = sum(d["cos_sim_mean"] for d in deepstack_cos_sim) / len(deepstack_cos_sim) if deepstack_cos_sim else 0.0
    
    return {
        "final_cos_sim_mean": cos_sim_final_mean,
        "final_cos_sim_std": cos_sim_final_std,
        "deepstack_cos_sim_avg": avg_cos_sim,
        "deepstack_cos_sim_per_layer": deepstack_cos_sim,
    }


def compute_feature_statistics(
    teacher_features: torch.Tensor,
    student_features: torch.Tensor,
    name: str = "features",
) -> dict:
    """Compute statistical properties of features."""
    return {
        f"{name}_teacher_mean": teacher_features.mean().item(),
        f"{name}_teacher_std": teacher_features.std().item(),
        f"{name}_teacher_min": teacher_features.min().item(),
        f"{name}_teacher_max": teacher_features.max().item(),
        f"{name}_student_mean": student_features.mean().item(),
        f"{name}_student_std": student_features.std().item(),
        f"{name}_student_min": student_features.min().item(),
        f"{name}_student_max": student_features.max().item(),
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    teacher_model_path: str,
    data_path: str,
    output_dir: str = "./eval_results",
    batch_size: int = 16,
    num_eval_samples: int = 500,
    device: torch.device = None,
    config: dict = None,
) -> dict:
    """
    Evaluate a student checkpoint against the teacher.
    
    Returns a dict of all evaluation metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Default config if not provided
    if config is None:
        config = {
            "teacher_output": 4096,
            "student_output": 2048,
            "teacher_deepstack_layers": [8, 16, 24],
            "projector_hidden_dim": 1024,
        }
    
    # Load teacher model
    print("Loading teacher model...")
    teacher = load_teacher_model(teacher_model_path, device, logger=None)
    teacher.eval()
    
    # Load student model
    print("Loading student model...")
    student = load_student_model(
        checkpoint_path.replace("/checkpoint_step_", "/"),  # parent dir
        device, logger=None
    )
    
    # Load checkpoint into student
    if os.path.exists(checkpoint_path):
        print(f"Loading student weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "student_state_dict" in ckpt:
            state_dict = ckpt["student_state_dict"]
        else:
            state_dict = ckpt
        # Strip DDP "module." prefix from checkpoint keys
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        student.load_state_dict(state_dict)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using base student model")
    
    student.eval()
    
    # Build distillation module
    distill_module = build_vision_distillation_module(
        teacher_dim=config.get("teacher_output", 4096),
        student_dim=config.get("student_output", 2048),
        deepstack_teacher_layers=config.get("teacher_deepstack_layers", [8, 16, 24]),
        projector_hidden_dim=config.get("projector_hidden_dim", 1024),
    ).to(device)
    
    distill_module.set_student_vit(student)
    
    # Dataset
    print(f"Loading dataset from: {data_path}")
    transform = get_default_transform(image_size=224)
    dataset = ImageInferenceDataset(data_path, transform=transform)
    
    # Limit number of samples for faster eval
    num_samples = min(num_eval_samples, len(dataset))
    dataset = torch.utils.data.Subset(dataset, range(num_samples))
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Evaluating on {num_samples} samples in {len(dataloader)} batches...\n")
    
    # Collect metrics across all batches
    all_final_mse = []
    all_deepstack_mse = []
    all_final_cos_sim = []
    all_deepstack_cos_sim = []
    
    all_final_stats = []
    all_deepstack_stats = {i: [] for i in range(3)}
    
    print("Using real teacher model for evaluation")
    print("Computing teacher-student alignment metrics...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            
            # Get teacher features (final + deepstack at layers [8, 16, 24])
            teacher_final, teacher_deepstack = distill_module.get_teacher_features(pixel_values, teacher)
            # Project teacher features to student dimension (4096 -> 2048)
            teacher_final_proj, teacher_deepstack_proj = distill_module.project_teacher_features(
                teacher_final, teacher_deepstack
            )
            # Get student features
            student_final, student_deepstack = distill_module.forward_student(pixel_values)
            
            # MSE metrics
            mse_metrics = compute_mse_metrics(
                teacher_final_proj, student_final,
                teacher_deepstack_proj, student_deepstack,
            )
            all_final_mse.append(mse_metrics["final_mse"])
            all_deepstack_mse.append(mse_metrics["deepstack_mse_avg"])
            
            # Cosine similarity
            cos_sim_final = F.cosine_similarity(teacher_final_proj, student_final, dim=-1)
            all_final_cos_sim.append(cos_sim_final.mean().item())
            
            for i, (t, s) in enumerate(zip(teacher_deepstack_proj, student_deepstack)):
                cos_sim = F.cosine_similarity(t, s, dim=-1).mean().item()
                all_deepstack_cos_sim.append(cos_sim)
            
            # Feature statistics
            final_stats = compute_feature_statistics(teacher_final_proj, student_final, "final")
            all_final_stats.append(final_stats)
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}: "
                      f"final_mse={mse_metrics['final_mse']:.6f}, "
                      f"cos_sim={cos_sim_final.mean().item():.4f}")
    
    # Aggregate results
    results = {
        "checkpoint_path": checkpoint_path,
        "num_eval_samples": num_samples,
        "num_batches": len(dataloader),
        
        # MSE Summary
        "final_mse_mean": np.mean(all_final_mse),
        "final_mse_std": np.std(all_final_mse),
        "final_mse_min": np.min(all_final_mse),
        "final_mse_max": np.max(all_final_mse),
        
        "deepstack_mse_mean": np.mean(all_deepstack_mse),
        "deepstack_mse_std": np.std(all_deepstack_mse),
        
        # Cosine Similarity Summary
        "final_cos_sim_mean": np.mean(all_final_cos_sim),
        "final_cos_sim_std": np.std(all_final_cos_sim),
        
        "deepstack_cos_sim_mean": np.mean(all_deepstack_cos_sim),
        "deepstack_cos_sim_std": np.std(all_deepstack_cos_sim),
        
        # Per-layer deepstack analysis
        "deepstack_per_layer": {},
    }
    
    # Per-layer deepstack MSE
    for layer_idx in range(3):
        layer_mses = all_deepstack_mse  # Already per-layer
        layer_cos = [all_deepstack_cos_sim[i::3] for i in range(3)]  # Strided
        results["deepstack_per_layer"][f"layer_{layer_idx}"] = {
            "cos_sim_mean": np.mean(layer_cos[layer_idx]) if layer_idx < len(layer_cos) else 0,
            "cos_sim_std": np.std(layer_cos[layer_idx]) if layer_idx < len(layer_cos) else 0,
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Eval Samples: {num_samples}")
    print(f"\n--- MSE (lower is better) ---")
    print(f"  Final Output MSE:   {results['final_mse_mean']:.6f} ± {results['final_mse_std']:.6f}")
    print(f"  Deepstack MSE:      {results['deepstack_mse_mean']:.6f} ± {results['deepstack_mse_std']:.6f}")
    print(f"\n--- Cosine Similarity (higher = better alignment, target >0.9) ---")
    print(f"  Final Output:       {results['final_cos_sim_mean']:.4f} ± {results['final_cos_sim_std']:.4f}")
    print(f"  Deepstack Avg:      {results['deepstack_cos_sim_mean']:.4f} ± {results['deepstack_cos_sim_std']:.4f}")
    
    print(f"\n--- Per-Layer Deepstack Alignment ---")
    for layer_idx in range(3):
        layer_info = results["deepstack_per_layer"].get(f"layer_{layer_idx}", {})
        print(f"  Layer {layer_idx}: cos_sim = {layer_info.get('cos_sim_mean', 0):.4f}")
    
    # Quality assessment
    print(f"\n--- Quality Assessment ---")
    if results["final_cos_sim_mean"] > 0.95:
        print("  ✅ Excellent alignment (cos_sim > 0.95)")
    elif results["final_cos_sim_mean"] > 0.90:
        print("  ✅ Good alignment (cos_sim > 0.90)")
    elif results["final_cos_sim_mean"] > 0.80:
        print("  ⚠️  Moderate alignment (0.80 < cos_sim < 0.90) - may need more training")
    else:
        print("  ❌ Poor alignment (cos_sim < 0.80) - distillation may have failed")
    
    print(f"{'='*60}\n")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    ckpt_name = Path(checkpoint_path).stem
    results_path = os.path.join(output_dir, f"eval_results_{ckpt_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate distilled ViT student model")
    parser.add_argument("--checkpoint", type=str, 
                       default="./output_vit_distill/checkpoints/checkpoint_step_5000.pt",
                       help="Path to student checkpoint to evaluate")
    parser.add_argument("--checkpoint1", type=str, default=None, help="First checkpoint for comparison")
    parser.add_argument("--checkpoint2", type=str, default=None, help="Second checkpoint for comparison")
    parser.add_argument("--teacher_model_path", type=str, 
                       default="/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B/",
                       help="Path to teacher model")
    parser.add_argument("--data_path", type=str,
                       default="/data01/mikelee/data/data_sample_chunk{0..24}/infer/",
                       help="Path to evaluation data")
    parser.add_argument("--output_dir", type=str, default="./eval_results", 
                       help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=500,
                       help="Number of samples to evaluate (use -1 for all)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config JSON with model parameters")
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.checkpoint1 and args.checkpoint2:
        # Compare two checkpoints
        print(f"\n{'#'*60}")
        print("COMPARING TWO CHECKPOINTS")
        print(f"#"*60)
        
        results1 = evaluate_checkpoint(
            args.checkpoint1, args.teacher_model_path, args.data_path,
            args.output_dir, args.batch_size, args.num_samples, device, config,
        )
        results2 = evaluate_checkpoint(
            args.checkpoint2, args.teacher_model_path, args.data_path,
            args.output_dir, args.batch_size, args.num_samples, device, config,
        )
        
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Checkpoint 1: {args.checkpoint1}")
        print(f"  Final MSE: {results1['final_mse_mean']:.6f}, CosSim: {results1['final_cos_sim_mean']:.4f}")
        print(f"\nCheckpoint 2: {args.checkpoint2}")
        print(f"  Final MSE: {results2['final_mse_mean']:.6f}, CosSim: {results2['final_cos_sim_mean']:.4f}")
        
        mse_diff = results1['final_mse_mean'] - results2['final_mse_mean']
        cos_diff = results2['final_cos_sim_mean'] - results1['final_cos_sim_mean']
        print(f"\nDelta: MSE={mse_diff:+.6f}, CosSim={cos_diff:+.4f}")
        print(f"{'='*60}\n")
        
    else:
        # Single checkpoint evaluation
        results = evaluate_checkpoint(
            args.checkpoint, args.teacher_model_path, args.data_path,
            args.output_dir, args.batch_size, args.num_samples, device, config,
        )
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
