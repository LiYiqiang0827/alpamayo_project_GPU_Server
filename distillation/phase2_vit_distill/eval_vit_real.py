#!/usr/bin/env python
"""
Phase 2 ViT Distillation - Real Teacher Evaluation
==================================================
Properly implements teacher (Alpamayo-1.5-10B) and student (Cosmos+checkpoint)
models instead of using dummy placeholders.
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


# ==============================================================================
# Real Teacher Vision Encoder (Alpamayo-1.5-10B from safetensors)
# ==============================================================================

class AlpamayoAttentionBlock(nn.Module):
    """Single attention block for Alpamayo."""
    def __init__(self, sd, idx, hidden_size=1152, num_heads=16):
        super().__init__()
        p = f"vlm.model.visual.blocks.{idx}"
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
        inter_size = fc1_w.shape[0]
        self.mlp_fc1 = nn.Linear(hidden_size, inter_size, bias=True)
        self.mlp_fc1.weight.data = fc1_w; self.mlp_fc1.bias.data = fc1_b
        
        fc2_w = sd[f"{p}.mlp.linear_fc2.weight"].float()
        fc2_b = sd[f"{p}.mlp.linear_fc2.bias"].float()
        self.mlp_fc2 = nn.Linear(inter_size, hidden_size, bias=True)
        self.mlp_fc2.weight.data = fc2_w; self.mlp_fc2.bias.data = fc2_b

    def forward(self, x):
        # Pre-norm attention + residual
        h = self.norm1(x)
        h = x + self.proj(self._attn(h))
        # Pre-norm MLP + residual
        h = h + self.mlp_fc2(F.gelu(self.mlp_fc1(self.norm2(h))))
        return h
    
    def _attn(self, x):
        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, -1).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = (q.shape[-1] ** -0.5)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        return attn @ v


class AlpamayoMerger(nn.Module):
    """2x2 spatial merger for Alpamayo."""
    def __init__(self, norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b,
                 hidden_size=1152, out_features=4096, use_postshuffle_norm=True):
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


class AlpamayoVisionEncoder(nn.Module):
    """Full Alpamayo-1.5-10B vision encoder."""
    def __init__(self, state_dict, deepstack_layers=[9, 17, 25], out_features=4096):
        super().__init__()
        self.deepstack_layers = deepstack_layers
        
        # Patch embed (Conv3D)
        pe_w = state_dict["vlm.model.visual.patch_embed.proj.weight"].float()
        pe_b = state_dict["vlm.model.visual.patch_embed.proj.bias"].float()
        self.patch_embed = nn.Conv3d(3, 1152, kernel_size=(2, 16, 16), stride=(2, 16, 16), bias=True)
        self.patch_embed.weight.data = pe_w; self.patch_embed.bias.data = pe_b
        
        # Position embedding
        pos_w = state_dict["vlm.model.visual.pos_embed.weight"].float()
        self.pos_embed = nn.Embedding.from_pretrained(pos_w, freeze=True)
        
        # 27 transformer blocks
        self.blocks = nn.ModuleList([
            AlpamayoAttentionBlock(state_dict, i, hidden_size=1152) for i in range(27)
        ])
        
        # Final merger (pre-norm)
        self.final_merger = AlpamayoMerger(
            state_dict["vlm.model.visual.merger.norm.weight"].float(),
            state_dict["vlm.model.visual.merger.norm.bias"].float(),
            state_dict["vlm.model.visual.merger.linear_fc1.weight"].float(),
            state_dict["vlm.model.visual.merger.linear_fc1.bias"].float(),
            state_dict["vlm.model.visual.merger.linear_fc2.weight"].float(),
            state_dict["vlm.model.visual.merger.linear_fc2.bias"].float(),
            hidden_size=1152, out_features=out_features, use_postshuffle_norm=False,
        )
        
        # Deepstack mergers (post-norm)
        self.deepstack_mergers = nn.ModuleList([
            AlpamayoMerger(
                state_dict[f"vlm.model.visual.deepstack_merger_list.{i}.norm.weight"].float(),
                state_dict[f"vlm.model.visual.deepstack_merger_list.{i}.norm.bias"].float(),
                state_dict[f"vlm.model.visual.deepstack_merger_list.{i}.linear_fc1.weight"].float(),
                state_dict[f"vlm.model.visual.deepstack_merger_list.{i}.linear_fc1.bias"].float(),
                state_dict[f"vlm.model.visual.deepstack_merger_list.{i}.linear_fc2.weight"].float(),
                state_dict[f"vlm.model.visual.deepstack_merger_list.{i}.linear_fc2.bias"].float(),
                hidden_size=1152, out_features=out_features, use_postshuffle_norm=True,
            ) for i in range(len(deepstack_layers))
        ])
    
    def forward(self, pixel_values, output_hidden_states=False):
        B, C, H, W = pixel_values.shape
        
        # Conv3D patch embed (needs temporal >= 2)
        x = pixel_values.unsqueeze(2)  # [B, 3, 1, H, W]
        x = F.pad(x, (0,0, 0,0, 0,1))  # temporal: 1->2
        x = self.patch_embed(x)  # [B, 1152, 1, H//16, W//16]
        x = x.squeeze(2)  # [B, 1152, H//16, W//16]
        
        # Reshape to sequence
        B2, C2, H2, W2 = x.shape
        x = x.reshape(B2, C2, H2*W2).permute(0, 2, 1)  # [B, seq, 1152]
        
        # Positional embedding
        seq_len = x.shape[1]
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(B, -1)
        pos_ids = pos_ids.clamp(max=self.pos_embed.num_embeddings - 1)
        x = x + self.pos_embed(pos_ids)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final output with mean pool
        final_out = self.final_merger(x).mean(dim=1)  # [B, 4096]
        
        # Deepstack: need to recompute through relevant layers
        deepstack_features = self._compute_deepstack(pixel_values)
        
        if output_hidden_states:
            class ModelOutput:
                def __init__(self, lhs, hss):
                    self.last_hidden_state = lhs
                    self.hidden_states = hss
            return ModelOutput(x, (x,))
        
        return final_out, deepstack_features
    
    def _compute_deepstack(self, pixel_values):
        """Recompute deepstack features at specified layers."""
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
        
        deepstack_features = []
        for block_idx, block in enumerate(self.blocks):
            x = block(x)
            if block_idx in self.deepstack_layers:
                ds_idx = self.deepstack_layers.index(block_idx)
                merged = self.deepstack_mergers[ds_idx](x)
                deepstack_features.append(merged.mean(dim=1))
        
        return [deepstack_features[self.deepstack_layers.index(i)] 
                for i in self.deepstack_layers]


class AlpamayoWrapper(nn.Module):
    """Wrapper to make AlpamayoVisionEncoder compatible with distillation interface."""
    def __init__(self, encoder):
        super().__init__()
        self.vision_encoder = encoder
    
    def forward(self, pixel_values, output_hidden_states=False):
        return self.vision_encoder(pixel_values, output_hidden_states=output_hidden_states)


# ==============================================================================
# Real Student Vision Encoder (Cosmos base + trained checkpoint)
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
        
        # MLP: inter_size from Cosmos base
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
    """
    Cosmos-Reason2 expanded vision encoder with trained checkpoint weights.
    
    Key: The checkpoint MLP weights are [1024, 1024] - meaning the student was 
    trained with inter_size=1024 (no MLP expansion), NOT inter_size=4096 from Cosmos.
    We override the MLP layers with the trained [1024, 1024] weights.
    """
    def __init__(self, cosmos_sd, ckpt_sd, deepstack_layers=[5, 11, 17]):
        super().__init__()
        self.deepstack_layers = deepstack_layers
        
        # Patch embed from Cosmos
        pe_w = cosmos_sd["model.visual.patch_embed.proj.weight"].float()
        pe_b = cosmos_sd["model.visual.patch_embed.proj.bias"].float()
        self.patch_embed = nn.Conv3d(3, 1024, kernel_size=(2, 16, 16), stride=(2, 16, 16), bias=True)
        self.patch_embed.weight.data = pe_w; self.patch_embed.bias.data = pe_b
        
        # Position embedding
        pos_w = cosmos_sd["model.visual.pos_embed.weight"].float()
        self.pos_embed = nn.Embedding.from_pretrained(pos_w, freeze=True)
        
        # Load checkpoint to get trained MLP inter_size
        clean = {k.replace("module.", ""): v for k, v in ckpt_sd.items()}
        trained_mlp_w = clean["layers.0.weight"].float()
        trained_inter_size = trained_mlp_w.shape[0]  # Should be 1024
        trained_hidden_size = trained_mlp_w.shape[1]  # Should be 1024
        print(f"  Trained MLP: hidden={trained_hidden_size}, inter={trained_inter_size}")
        
        # 24 transformer blocks, override MLP with trained weights
        self.blocks = nn.ModuleList([
            CosmosAttentionBlock(cosmos_sd, i, hidden_size=1024, inter_size=trained_inter_size) 
            for i in range(24)
        ])
        
        # Override MLP weights with trained checkpoint
        for i in range(24):
            w = clean[f"layers.{i}.weight"].float()   # [1024, 1024]
            b = clean[f"layers.{i}.bias"].float()      # [1024]
            self.blocks[i].mlp_fc1.weight.data = w
            self.blocks[i].mlp_fc1.bias.data = b
            # mlp_fc2 is identity (trained model has no expansion)
            self.blocks[i].mlp_fc2.weight.data = torch.eye(1024)
            self.blocks[i].mlp_fc2.bias.data = torch.zeros(1024)
        
        # Final merger (pre-norm, 1024->4096->2048)
        self.final_merger = CosmosMerger(
            cosmos_sd["model.visual.merger.norm.weight"].float(),
            cosmos_sd["model.visual.merger.norm.bias"].float(),
            cosmos_sd["model.visual.merger.linear_fc1.weight"].float(),
            cosmos_sd["model.visual.merger.linear_fc1.bias"].float(),
            cosmos_sd["model.visual.merger.linear_fc2.weight"].float(),
            cosmos_sd["model.visual.merger.linear_fc2.bias"].float(),
            hidden_size=1024, out_features=2048, use_postshuffle_norm=False,
        )
        
        # Deepstack mergers (post-norm)
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
        
        # Output projection from checkpoint: weight [2048, 1024], bias [2048]
        # This is: output = input @ W^T + b where input is [B, 1024], output is [B, 2048]
        # In PyTorch Linear: weight is [out_features, in_features] = [2048, 1024]
        out_w = clean["output_proj.weight"].float()   # [2048, 1024] (correct for PyTorch Linear)
        out_b = clean["output_proj.bias"].float()    # [2048]
        self.output_proj = nn.Linear(1024, 2048, bias=True)
        self.output_proj.weight.data = out_w
        self.output_proj.bias.data = out_b
    
    def forward(self, pixel_values):
        B, C, H, W = pixel_values.shape
        
        # Conv3D patch embed
        x = pixel_values.unsqueeze(2)
        x = F.pad(x, (0,0, 0,0, 0,1))
        x = self.patch_embed(x).squeeze(2)
        
        # Reshape to sequence [B, seq, 1024]
        B2, C2, H2, W2 = x.shape
        x = x.reshape(B2, C2, H2*W2).permute(0, 2, 1)
        
        # Positional embedding
        seq_len = x.shape[1]
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(B, -1)
        pos_ids = pos_ids.clamp(max=self.pos_embed.num_embeddings - 1)
        x = x + self.pos_embed(pos_ids)
        
        # Transformer blocks + deepstack collection
        deepstack_hidden = {}
        for block_idx, block in enumerate(self.blocks):
            x = block(x)
            if block_idx in self.deepstack_layers:
                deepstack_hidden[block_idx] = x
        
        # Deepstack features (from collected hidden states)
        deepstack_features = []
        for layer_idx in self.deepstack_layers:
            h = deepstack_hidden[layer_idx]
            ds_idx = self.deepstack_layers.index(layer_idx)
            merged = self.deepstack_mergers[ds_idx](h)
            deepstack_features.append(merged.mean(dim=1))
        
        # Final output with mean pool
        # The merger already outputs [B, 2048] - no output_proj needed
        # (output_proj was for the original Cosmos architecture before the merger)
        final_merged = self.final_merger(x)  # [B, seq//4, 2048]
        final_out = final_merged.mean(dim=1)  # [B, 2048]
        
        return final_out, deepstack_features


# ==============================================================================
# Dataset
# ==============================================================================

class ImageInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None, image_size=224):
        import glob as glob_mod, os
        self.data_path = data_path
        self.transform = transform
        self.image_size = image_size
        
        self.image_paths = []
        if os.path.isdir(data_path):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
                self.image_paths.extend(sorted(glob_mod.glob(
                    os.path.join(data_path, '**', ext), recursive=True)))
        
        print(f"Found {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (self.image_size, self.image_size), color='gray')
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img}


# ==============================================================================
# Evaluation
# ==============================================================================

class TeacherProjector(nn.Module):
    """Projects teacher 4096-dim to student 2048-dim."""
    def __init__(self, in_dim=4096, out_dim=2048, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x):
        return self.net(x)


def compute_metrics(t_final, s_final, t_ds, s_ds, projector=None):
    """Compute metrics with optional teacher projection to student dim."""
    metrics = {}
    
    # Project teacher to student dimension if needed
    if projector is not None:
        t_final_p = projector(t_final)
        t_ds_p = [projector(t) for t in t_ds]
    else:
        t_final_p = t_final
        t_ds_p = t_ds
    
    metrics["final_mse"] = F.mse_loss(t_final_p, s_final).item()
    metrics["final_cos_sim"] = F.cosine_similarity(t_final_p, s_final, dim=-1).mean().item()
    
    ds_mses, ds_coss = [], []
    for t_p, s in zip(t_ds_p, s_ds):
        ds_mses.append(F.mse_loss(t_p, s).item())
        ds_coss.append(F.cosine_similarity(t_p, s, dim=-1).mean().item())
    
    metrics["deepstack_mse_avg"] = np.mean(ds_mses)
    metrics["deepstack_cos_sim_avg"] = np.mean(ds_coss)
    metrics["deepstack_mses"] = ds_mses
    metrics["deepstack_coss"] = ds_coss
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str,
                        default="/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B/")
    parser.add_argument("--cosmos_path", type=str,
                        default="/home/user/cosmos_reason2_expanded/")
    parser.add_argument("--student_ckpt", type=str,
                        default="/gpfs-data/mikelee/distillation_output/final_vit_model.pt")
    parser.add_argument("--data_path", type=str,
                        default="/data01/mikelee/data/data_sample_chunk0/infer/")
    parser.add_argument("--output_dir", type=str, default="./eval_results_real_teacher")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load Teacher
    print("\nLoading Teacher (Alpamayo-1.5-10B)...")
    teacher_files = sorted(glob.glob(os.path.join(args.teacher_path, "*.safetensors")))
    teacher_sd = {}
    for f in teacher_files:
        sd = load_file(f, device="cpu")
        teacher_sd.update({k: v.float() for k, v in sd.items()})
        del sd
    print(f"  Loaded {len(teacher_sd)} keys")
    
    teacher_encoder = AlpamayoVisionEncoder(teacher_sd, deepstack_layers=[9, 17, 25], out_features=4096)
    teacher = AlpamayoWrapper(teacher_encoder).to(device).eval()
    print(f"  Teacher: 27 blocks, hidden=1152, output=4096, deepstack=[9,17,25]")
    del teacher_sd
    
    # Load Cosmos + student checkpoint
    print("\nLoading Student (Cosmos + checkpoint)...")
    cosmos_sd = load_file(os.path.join(args.cosmos_path, "model-expanded.safetensors"), device="cpu")
    cosmos_sd = {k: v.float() for k, v in cosmos_sd.items()}
    print(f"  Cosmos: {len(cosmos_sd)} keys")
    
    ckpt = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
    ckpt_sd = ckpt.get("student_state_dict", ckpt)
    print(f"  Checkpoint: {len(ckpt_sd)} keys")
    
    student = CosmosVisionEncoder(cosmos_sd, ckpt_sd, deepstack_layers=[5, 11, 17])
    student = student.to(device).eval()
    print(f"  Student: 24 blocks, hidden=1024, output=2048, deepstack=[5,11,17]")
    del cosmos_sd
    
    # Dataset
    print(f"\nLoading dataset (image_size={args.image_size})...")
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = ImageInferenceDataset(args.data_path, transform=transform, image_size=args.image_size)
    dataset = torch.utils.data.Subset(dataset, range(min(args.num_samples, len(dataset))))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"  Dataset: {len(dataset)} samples, {len(dataloader)} batches")
    
    # Projector: teacher 4096 → student 2048
    projector = TeacherProjector(in_dim=4096, out_dim=2048, hidden_dim=1024).to(device).eval()
    
    # Evaluate
    print("\nRunning evaluation...")
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            pv = batch["pixel_values"].to(device)
            
            # Teacher
            t_out = teacher.vision_encoder(pv, output_hidden_states=False)
            t_final, t_ds = t_out[0], t_out[1]
            
            # Student
            s_final, s_ds = student(pv)
            
            # Metrics (project teacher to student dim)
            metrics = compute_metrics(t_final, s_final, t_ds, s_ds, projector=projector)
            all_metrics.append(metrics)
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: final_mse={metrics['final_mse']:.6f}, "
                      f"final_cos={metrics['final_cos_sim']:.4f}")
    
    # Aggregate
    final_mses = [m["final_mse"] for m in all_metrics]
    final_coss = [m["final_cos_sim"] for m in all_metrics]
    ds_mses = [m["deepstack_mse_avg"] for m in all_metrics]
    ds_coss = [m["deepstack_cos_sim_avg"] for m in all_metrics]
    
    results = {
        "num_samples": len(dataset),
        "num_batches": len(all_metrics),
        "final_mse_mean": float(np.mean(final_mses)),
        "final_mse_std": float(np.std(final_mses)),
        "final_cos_sim_mean": float(np.mean(final_coss)),
        "final_cos_sim_std": float(np.std(final_coss)),
        "deepstack_mse_mean": float(np.mean(ds_mses)),
        "deepstack_mse_std": float(np.std(ds_mses)),
        "deepstack_cos_sim_mean": float(np.mean(ds_coss)),
        "deepstack_cos_sim_std": float(np.std(ds_coss)),
        "teacher": "Alpamayo-1.5-10B (27 blocks, 1152 hidden, 4096 output)",
        "student": "Cosmos + trained checkpoint (24 blocks, 1024 hidden, 2048 output)",
        "deepstack_teacher": [9, 17, 25],
        "deepstack_student": [5, 11, 17],
    }
    
    print("\n" + "="*70)
    print("REAL TEACHER EVALUATION RESULTS")
    print("="*70)
    print(f"Teacher: Alpamayo-1.5-10B (27 blocks, 1152 hidden, 4096 output)")
    print(f"Student: Cosmos + checkpoint (24 blocks, 1024 hidden, 2048 output)")
    print(f"Samples: {results['num_samples']}")
    print(f"\n--- Final Output ---")
    print(f"  MSE:      {results['final_mse_mean']:.6f} ± {results['final_mse_std']:.6f}")
    print(f"  CosSim:   {results['final_cos_sim_mean']:.4f} ± {results['final_cos_sim_std']:.4f}")
    print(f"\n--- Deepstack Features ---")
    print(f"  MSE:      {results['deepstack_mse_mean']:.6f} ± {results['deepstack_mse_std']:.6f}")
    print(f"  CosSim:   {results['deepstack_cos_sim_mean']:.4f} ± {results['deepstack_cos_sim_std']:.4f}")
    
    quality = "✅ Excellent" if results['final_cos_sim_mean'] > 0.95 else \
              "✅ Good" if results['final_cos_sim_mean'] > 0.90 else \
              "⚠️  Moderate" if results['final_cos_sim_mean'] > 0.80 else "❌ Poor"
    print(f"\n--- Quality: {quality} ---")
    
    out_file = os.path.join(args.output_dir, "eval_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
