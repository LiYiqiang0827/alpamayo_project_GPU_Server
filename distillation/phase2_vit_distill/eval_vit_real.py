#!/usr/bin/env python
"""
Phase 2 ViT Distillation - Evaluation Script
==============================================
Uses EXACT same architecture as training code for accurate evaluation.
"""

import os, sys, json, glob, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from PIL import Image
from torchvision import transforms

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from vit_projector import build_vision_distillation_module
from vit_distillation import (
    load_teacher_model, load_student_model, 
    ImageInferenceDataset, get_default_transform
)

# ==============================================================================
# Cosmos Vision Encoder (needed by vit_distillation._build_student_vit_model)
# ==============================================================================

class CosmosAttentionBlock(nn.Module):
    """Single attention block for Cosmos student."""
    def __init__(self, sd, idx, hidden_size=1024, inter_size=4096):
        super().__init__()
        p = f"model.visual.blocks.{idx}"
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        
        qkv_w = sd[f"{p}.attn.qkv.weight"].float()
        qkv_b = sd[f"{p}.attn.qkv.bias"].float()
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.qkv.weight.data = qkv_w; self.qkv.bias.data = qkv_b
        
        proj_w = sd[f"{p}.attn.proj.weight"].float()
        proj_b = sd[f"{p}.attn.proj.bias"].float()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.proj.weight.data = proj_w; self.proj.bias.data = proj_b
        
        fc1_w = sd[f"{p}.mlp.linear_fc1.weight"].float()
        fc1_b = sd[f"{p}.mlp.linear_fc1.bias"].float()
        self.mlp_fc1 = nn.Linear(hidden_size, inter_size, bias=True)
        self.mlp_fc1.weight.data = fc1_w; self.mlp_fc1.bias.data = fc1_b
        
        fc2_w = sd[f"{p}.mlp.linear_fc2.weight"].float()
        fc2_b = sd[f"{p}.mlp.linear_fc2.bias"].float()
        self.mlp_fc2 = nn.Linear(inter_size, hidden_size, bias=True)
        self.mlp_fc2.weight.data = fc2_w; self.mlp_fc2.bias.data = fc2_b

    def forward(self, x):
        h = self.norm1(x)
        h = x + self.proj(self._attn(h))
        h = h + self.mlp_fc2(F.gelu(self.mlp_fc1(self.norm2(h))))
        return h
    
    def _attn(self, x):
        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, -1).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = (q.shape[-1] ** -0.5)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        return attn @ v


class CosmosMerger(nn.Module):
    """2x2 spatial merger for Cosmos student."""
    def __init__(self, norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b,
                 hidden_size=1024, out_features=2048, use_postshuffle_norm=True):
        super().__init__()
        self.use_postshuffle_norm = use_postshuffle_norm
        merged_dim = 4 * hidden_size
        
        if use_postshuffle_norm:
            self.norm = nn.LayerNorm(merged_dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm.weight.data = norm_w; self.norm.bias.data = norm_b
        
        self.fc1 = nn.Linear(merged_dim, merged_dim, bias=True)
        self.fc1.weight.data = fc1_w; self.fc1.bias.data = fc1_b
        self.fc2 = nn.Linear(merged_dim, out_features, bias=True)
        self.fc2.weight.data = fc2_w; self.fc2.bias.data = fc2_b
    
    def forward(self, x):
        B, seq, H = x.shape
        grid = int(seq ** 0.5)
        mh = grid // 2
        
        if self.use_postshuffle_norm:
            x = x.reshape(B, mh, 2, mh, 2, H).permute(0,1,3,2,4,5).reshape(B, mh*mh, 4*H)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = x.reshape(B, mh, 2, mh, 2, H).permute(0,1,3,2,4,5).reshape(B, mh*mh, 4*H)
        
        return self.fc2(F.gelu(self.fc1(x)))


class CosmosVisionEncoder(nn.Module):
    """Cosmos-Reason2-2B / Qwen3.5-VL-2B Vision Encoder."""
    def __init__(self, cosmos_sd, deepstack_layers=[5, 11, 17]):
        super().__init__()
        self.deepstack_layers = deepstack_layers
        
        pe_w = cosmos_sd["model.visual.patch_embed.proj.weight"].float()
        pe_b = cosmos_sd["model.visual.patch_embed.proj.bias"].float()
        self.patch_embed = nn.Conv3d(3, 1024, kernel_size=(2, 16, 16), stride=(2, 16, 16), bias=True)
        self.patch_embed.weight.data = pe_w; self.patch_embed.bias.data = pe_b
        
        pos_w = cosmos_sd["model.visual.pos_embed.weight"].float()
        self.pos_embed = nn.Embedding.from_pretrained(pos_w, freeze=True)
        
        self.blocks = nn.ModuleList([
            CosmosAttentionBlock(cosmos_sd, i, hidden_size=1024, inter_size=4096) 
            for i in range(24)
        ])
        
        self.final_merger = CosmosMerger(
            cosmos_sd["model.visual.merger.norm.weight"].float(),
            cosmos_sd["model.visual.merger.norm.bias"].float(),
            cosmos_sd["model.visual.merger.linear_fc1.weight"].float(),
            cosmos_sd["model.visual.merger.linear_fc1.bias"].float(),
            cosmos_sd["model.visual.merger.linear_fc2.weight"].float(),
            cosmos_sd["model.visual.merger.linear_fc2.bias"].float(),
            hidden_size=1024, out_features=2048, use_postshuffle_norm=False,
        )
        
        self.deepstack_mergers = nn.ModuleList([
            CosmosMerger(
                cosmos_sd[f"model.visual.deepstack_merger_list.{i}.norm.weight"].float(),
                cosmos_sd[f"model.visual.deepstack_merger_list.{i}.norm.bias"].float(),
                cosmos_sd[f"model.visual.deepstack_merger_list.{i}.linear_fc1.weight"].float(),
                cosmos_sd[f"model.visual.deepstack_merger_list.{i}.linear_fc1.bias"].float(),
                cosmos_sd[f"model.visual.deepstack_merger_list.{i}.linear_fc2.weight"].float(),
                cosmos_sd[f"model.visual.deepstack_merger_list.{i}.linear_fc2.bias"].float(),
                hidden_size=1024, out_features=2048, use_postshuffle_norm=True,
            ) for i in range(len(deepstack_layers))
        ])
    
    def forward(self, pixel_values):
        B, C, H, W = pixel_values.shape
        x = pixel_values.unsqueeze(2)
        x = F.pad(x, (0,0, 0,0, 0,1))
        x = self.patch_embed(x).squeeze(2)
        B2, C2, H2, W2 = x.shape
        x = x.reshape(B2, C2, H2*W2).permute(0, 2, 1)
        seq_len = x.shape[1]
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(B, -1)
        pos_ids = pos_ids.clamp(max=self.pos_embed.num_embeddings - 1)
        x = x + self.pos_embed(pos_ids)
        
        deepstack_hidden = {}
        for block_idx, block in enumerate(self.blocks):
            x = block(x)
            if block_idx in self.deepstack_layers:
                deepstack_hidden[block_idx] = x
        
        deepstack_features = []
        for layer_idx in self.deepstack_layers:
            h = deepstack_hidden[layer_idx]
            ds_idx = self.deepstack_layers.index(layer_idx)
            merged = self.deepstack_mergers[ds_idx](h)
            deepstack_features.append(merged.mean(dim=1))
        
        final_merged = self.final_merger(x)
        final_out = final_merged.mean(dim=1)
        return final_out, deepstack_features


# ==============================================================================
# Evaluation Logic
# ==============================================================================

def evaluate_checkpoint(teacher, student, distill_module, dataloader, device):
    """Evaluate using EXACT same logic as training."""
    teacher.eval()
    student.eval()
    distill_module.eval()
    
    all_metrics = []
    
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            
            teacher_final, teacher_deepstack = distill_module.get_teacher_features(
                pixel_values, teacher
            )
            student_final, student_deepstack = distill_module.forward_student(pixel_values)
            teacher_final_proj, teacher_deepstack_proj = distill_module.project_teacher_features(
                teacher_final, teacher_deepstack
            )
            
            final_mse = F.mse_loss(teacher_final_proj, student_final).item()
            
            deepstack_mses = []
            for t, s in zip(teacher_deepstack_proj, student_deepstack):
                deepstack_mses.append(F.mse_loss(t, s).item())
            deepstack_mse = np.mean(deepstack_mses)
            
            final_cos = F.cosine_similarity(teacher_final_proj, student_final, dim=-1).mean().item()
            deepstack_coss = []
            for t, s in zip(teacher_deepstack_proj, student_deepstack):
                deepstack_coss.append(F.cosine_similarity(t, s, dim=-1).mean().item())
            
            all_metrics.append({
                "final_mse": final_mse,
                "deepstack_mse": deepstack_mse,
                "final_cos": final_cos,
                "deepstack_cos": np.mean(deepstack_coss),
            })
    
    return {
        "final_mse": np.mean([m["final_mse"] for m in all_metrics]),
        "final_mse_std": np.std([m["final_mse"] for m in all_metrics]),
        "deepstack_mse": np.mean([m["deepstack_mse"] for m in all_metrics]),
        "deepstack_mse_std": np.std([m["deepstack_mse"] for m in all_metrics]),
        "final_cos": np.mean([m["final_cos"] for m in all_metrics]),
        "deepstack_cos": np.mean([m["deepstack_cos"] for m in all_metrics]),
        "num_samples": len(all_metrics) * dataloader.batch_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, 
                       default="/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B/")
    parser.add_argument("--student_path", type=str, default="~/cosmos_reason2_expanded/")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--data_path", type=str, 
                       default="/data01/mikelee/data/data_sample_chunk0/infer/")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    print("Loading Teacher...")
    teacher = load_teacher_model(args.teacher_path, device, 0)
    
    print("Loading Student (base architecture)...")
    student = load_student_model(args.student_path, device, 0)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    student_sd = ckpt.get("student_state_dict", {})
    if student_sd:
        student_sd = {k.replace("module.", ""): v for k, v in student_sd.items()}
        student.load_state_dict(student_sd, strict=False)
        print(f"  Loaded student weights: {len(student_sd)} keys")
    
    print("Building distillation module...")
    distill_module = build_vision_distillation_module()
    distill_module.set_student_vit(student)
    distill_module = distill_module.to(device)
    
    projector_sd = ckpt.get("projector_state_dict", {})
    if projector_sd:
        distill_module.load_state_dict(projector_sd, strict=False)
        print(f"  Loaded projector weights: {len(projector_sd)} keys")
    
    print(f"\nLoading dataset...")
    transform = get_default_transform(args.image_size)
    dataset = ImageInferenceDataset(args.data_path, transform=transform, image_size=args.image_size)
    dataset = torch.utils.data.Subset(dataset, range(min(args.num_samples, len(dataset))))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"\nEvaluating {len(dataset)} samples...")
    metrics = evaluate_checkpoint(teacher, student, distill_module, dataloader, device)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Final MSE:      {metrics['final_mse']:.6f} ± {metrics['final_mse_std']:.6f}")
    print(f"Deepstack MSE:  {metrics['deepstack_mse']:.6f} ± {metrics['deepstack_mse_std']:.6f}")
    print(f"Final CosSim:   {metrics['final_cos']:.4f}")
    print(f"Deepstack Cos:  {metrics['deepstack_cos']:.4f}")
    print(f"Samples:        {metrics['num_samples']}")
    print("="*60)
    
    os.makedirs("eval_results", exist_ok=True)
    with open("eval_results/eval_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved: eval_results/eval_results.json")


if __name__ == "__main__":
    main()
