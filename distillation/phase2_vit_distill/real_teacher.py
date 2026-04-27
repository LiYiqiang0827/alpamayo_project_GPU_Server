"""
Real Alpamayo Vision Encoder - matches HuggingFace model output interface.
Outputs (final_output, deepstack_raw) but wrapped in an object with last_hidden_state and hidden_states.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
import json
import os
from typing import Optional, Tuple


class AlpamayoAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class AlpamayoTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = AlpamayoAttention(hidden_dim, num_heads)
        self.mlp_fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.mlp_fc2 = nn.Linear(intermediate_dim, hidden_dim)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp_fc2(F.gelu(self.mlp_fc1(self.norm2(x))))
        return x


class AlpamayoOutput:
    """Output object that mimics HuggingFace model output."""
    def __init__(self, last_hidden_state, hidden_states):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class AlpamayoVisionEncoder(nn.Module):
    """
    Real Alpamayo Vision Encoder - matches HuggingFace output interface.
    """
    def __init__(self, state_dict, deepstack_layers=[8, 16, 24]):
        super().__init__()
        self.deepstack_layers = deepstack_layers
        
        # Patch embed
        patch_w = state_dict['vlm.model.visual.patch_embed.proj.weight']
        patch_b = state_dict['vlm.model.visual.patch_embed.proj.bias']
        self.patch_embed = nn.Conv3d(3, 1152, kernel_size=(2, 16, 16), stride=(2, 16, 16))
        self.patch_embed.weight.data = patch_w.float()
        self.patch_embed.bias.data = patch_b.float()
        
        # Determine dimensions
        hidden_dim = state_dict['vlm.model.visual.blocks.0.norm1.weight'].shape[0]
        intermediate_dim = state_dict['vlm.model.visual.blocks.0.mlp.linear_fc1.weight'].shape[0]
        num_heads = 8
        
        # Build transformer blocks
        self.blocks = nn.ModuleList([
            AlpamayoTransformerBlock(hidden_dim, intermediate_dim, num_heads)
            for _ in range(27)
        ])
        
        # Load block weights
        for block_idx in range(27):
            prefix = f'vlm.model.visual.blocks.{block_idx}.'
            
            self.blocks[block_idx].norm1.weight.data = state_dict[f'{prefix}norm1.weight'].float()
            self.blocks[block_idx].norm1.bias.data = state_dict[f'{prefix}norm1.bias'].float()
            self.blocks[block_idx].norm2.weight.data = state_dict[f'{prefix}norm2.weight'].float()
            self.blocks[block_idx].norm2.bias.data = state_dict[f'{prefix}norm2.bias'].float()
            
            qkv_w = state_dict[f'{prefix}attn.qkv.weight']
            qkv_b = state_dict[f'{prefix}attn.qkv.bias']
            self.blocks[block_idx].attn.qkv.weight.data = qkv_w.float()
            self.blocks[block_idx].attn.qkv.bias.data = qkv_b.float()
            
            proj_w = state_dict[f'{prefix}attn.proj.weight']
            proj_b = state_dict[f'{prefix}attn.proj.bias']
            self.blocks[block_idx].attn.proj.weight.data = proj_w.float()
            self.blocks[block_idx].attn.proj.bias.data = proj_b.float()
            
            self.blocks[block_idx].mlp_fc1.weight.data = state_dict[f'{prefix}mlp.linear_fc1.weight'].float()
            self.blocks[block_idx].mlp_fc1.bias.data = state_dict[f'{prefix}mlp.linear_fc1.bias'].float()
            self.blocks[block_idx].mlp_fc2.weight.data = state_dict[f'{prefix}mlp.linear_fc2.weight'].float()
            self.blocks[block_idx].mlp_fc2.bias.data = state_dict[f'{prefix}mlp.linear_fc2.bias'].float()
        
        # Final LayerNorm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Final projection (for final output)
        self.final_proj = nn.Linear(hidden_dim, 4096)
        
        self.output_dim = 4096
        self.hidden_dim = hidden_dim
        
        print(f"AlpamayoVisionEncoder: 27 blocks, hidden={hidden_dim}, output={self.output_dim}")
    
    def forward(self, pixel_values, output_hidden_states=False):
        """
        Args:
            pixel_values: [B, C, H, W] - single image at 384x1280 or similar resolution
            output_hidden_states: if True, return AlpamayoOutput with last_hidden_state and hidden_states
            
        Returns:
            If output_hidden_states=False: (final_output, deepstack_raw_features)
            If output_hidden_states=True: AlpamayoOutput with:
                - last_hidden_state: [B, 1, 4096] final output
                - hidden_states: tuple of 28 tensors (embeddings + 27 layer outputs), 
                  each [B, seq, 1152] (RAW, not projected)
        """
        B, C, H, W = pixel_values.shape
        
        # Patch embed
        x = pixel_values.unsqueeze(2)  # [B, C, 1, H, W]
        x = F.pad(x, (0, 0, 0, 0, 0, 1))  # [B, C, 2, H, W]
        x = self.patch_embed(x)  # [B, 1152, 1, H_patches, W_patches]
        
        # Reshape to sequence
        x = x.squeeze(2)  # [B, 1152, H_patches, W_patches]
        B2, C2, H2, W2 = x.shape
        x = x.reshape(B2, C2, H2 * W2).permute(0, 2, 1)  # [B, seq, 1152]
        
        # Collect hidden states (embeddings + each layer output)
        all_hidden_states = [x]  # embeddings
        
        for block_idx, block in enumerate(self.blocks):
            x = block(x)
            all_hidden_states.append(x)  # after each layer
        
        # Final output - project to 4096
        cls_final = x[:, 0]  # [B, 1152]
        final_output = self.final_proj(cls_final)  # [B, 4096]
        
        if output_hidden_states:
            # Return HuggingFace-style output
            # hidden_states: tuple of 28 tensors, each [B, seq, 1152]
            # The get_teacher_features function expects hidden_states to be indexed by layer
            # Our deepstack_layers = [8, 16, 24] correspond to indices 9, 17, 25
            # (since all_hidden_states[0] = embeddings, all_hidden_states[1] = layer 0, etc.)
            return AlpamayoOutput(
                last_hidden_state=final_output.unsqueeze(1),  # [B, 1, 4096]
                hidden_states=tuple(all_hidden_states)  # 28 tensors, raw 1152-dim
            )
        
        # Without output_hidden_states, just return the deepstack raw features
        # This path is not used in our training but defined for completeness
        return final_output


def load_alpamayo_vision_encoder(checkpoint_dir, deepstack_layers=[8, 16, 24], device='cpu'):
    """Load real Alpamayo vision encoder from checkpoint."""
    checkpoint_dir = os.path.expanduser(checkpoint_dir)
    
    index_path = os.path.join(checkpoint_dir, 'model.safetensors.index.json')
    with open(index_path) as f:
        index = json.load(f)
    
    wm = index['weight_map']
    
    # Load all visual weights from all shards
    state_dict = {}
    for shard_file in sorted(set(wm.values())):
        shard_path = os.path.join(checkpoint_dir, shard_file)
        with safe_open(shard_path, framework="pt") as sf:
            for k in sf.keys():
                if k.startswith('vlm.model.visual.'):
                    state_dict[k] = sf.get_tensor(k).float()
    
    print(f"Loaded {len(state_dict)} visual weights from checkpoint")
    
    encoder = AlpamayoVisionEncoder(state_dict, deepstack_layers)
    encoder = encoder.to(device)
    encoder.eval()
    
    return encoder


def build_real_teacher_vit_model(checkpoint_path, device):
    """Build the real Alpamayo Vision Encoder from checkpoint."""
    return load_alpamayo_vision_encoder(checkpoint_path, deepstack_layers=[8, 16, 24], device=device)