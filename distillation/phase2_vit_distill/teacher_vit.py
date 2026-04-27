"""Alpamayo 1.5-10B Vision Encoder for distillation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlpamayoAttentionBlock(nn.Module):
    def __init__(self, state_dict, block_idx, hidden_size=1152):
        super().__init__()
        prefix = "vlm.model.visual.blocks.%d" % block_idx
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        qkv_w = state_dict[prefix + ".attn.qkv.weight"].float()
        qkv_b = state_dict[prefix + ".attn.qkv.bias"].float()
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.qkv.weight.data = qkv_w; self.qkv.bias.data = qkv_b
        proj_w = state_dict[prefix + ".attn.proj.weight"].float()
        proj_b = state_dict[prefix + ".attn.proj.bias"].float()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.proj.weight.data = proj_w; self.proj.bias.data = proj_b
        fc1_w = state_dict[prefix + ".mlp.linear_fc1.weight"].float()
        fc1_b = state_dict[prefix + ".mlp.linear_fc1.bias"].float()
        inter_size = fc1_w.shape[0]
        self.mlp_fc1 = nn.Linear(hidden_size, inter_size, bias=True)
        self.mlp_fc1.weight.data = fc1_w; self.mlp_fc1.bias.data = fc1_b
        fc2_w = state_dict[prefix + ".mlp.linear_fc2.weight"].float()
        fc2_b = state_dict[prefix + ".mlp.linear_fc2.bias"].float()
        self.mlp_fc2 = nn.Linear(inter_size, hidden_size, bias=True)
        self.mlp_fc2.weight.data = fc2_w; self.mlp_fc2.bias.data = fc2_b

    def forward(self, x):
        B, Seq, H = x.shape
        # Pre-norm
        h = self.norm1(x)
        # QKV projection: [B, Seq, 3*H] -> reshape to [B, Seq, 3, H]
        qkv = self.qkv(h).reshape(B, Seq, 3, H)
        q, k, v = qkv[:, :, 0, :], qkv[:, :, 1, :], qkv[:, :, 2, :]  # each [B, Seq, H]
        # Standard self-attention: [B, Seq, H] @ [B, H, Seq] -> [B, Seq, Seq]
        scale = H ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        # [B, Seq, Seq] @ [B, Seq, H] -> [B, Seq, H]
        attn_out = attn @ v
        # Output projection
        attn_out = self.proj(attn_out)
        x = x + attn_out
        # Pre-norm MLP + residual
        x = x + self.mlp_fc2(F.gelu(self.mlp_fc1(self.norm2(x))))
        return x


class AlpamayoMergerBlock(nn.Module):
    def __init__(self, norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b,
                 hidden_size=1152, merged_dim=4608, out_features=4096, use_postshuffle_norm=True):
        super().__init__()
        self.use_postshuffle_norm = use_postshuffle_norm
        if use_postshuffle_norm:
            self.norm = nn.LayerNorm(merged_dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm.weight.data = norm_w; self.norm.bias.data = norm_b
        self.fc1 = nn.Linear(merged_dim, merged_dim, bias=True)
        self.fc1.weight.data = fc1_w; self.fc1.bias.data = fc1_b
        self.fc2 = nn.Linear(merged_dim, out_features, bias=True)
        self.fc2.weight.data = fc2_w; self.fc2.bias.data = fc2_b

    def forward(self, x, mean_pool=True):
        B, seq, H = x.shape
        grid = int(seq ** 0.5)
        mh = grid // 2
        if self.use_postshuffle_norm:
            x = x.reshape(B, mh, 2, mh, 2, H).permute(0,1,3,2,4,5).reshape(B, mh*mh, 4*H)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = x.reshape(B, mh, 2, mh, 2, H).permute(0,1,3,2,4,5).reshape(B, mh*mh, 4*H)
        out = self.fc2(F.gelu(self.fc1(x)))
        if mean_pool:
            return out.mean(dim=1)
        return out


class AlpamayoVisionEncoder(nn.Module):
    def __init__(self, state_dict, deepstack_layers=None, out_features=4096, hidden_size=1152):
        super().__init__()
        if deepstack_layers is None:
            deepstack_layers = [8, 16, 24]
        self.deepstack_layers = deepstack_layers
        self.hidden_size = hidden_size
        self.out_features = out_features
        
        pe_w = state_dict["vlm.model.visual.patch_embed.proj.weight"].float()
        pe_b = state_dict["vlm.model.visual.patch_embed.proj.bias"].float()
        self.patch_embed = nn.Conv3d(3, hidden_size, kernel_size=(2, 16, 16), stride=(2, 16, 16), bias=True)
        self.patch_embed.weight.data = pe_w; self.patch_embed.bias.data = pe_b
        
        pos_w = state_dict["vlm.model.visual.pos_embed.weight"].float()
        self.pos_embed = nn.Embedding.from_pretrained(pos_w, freeze=True)
        
        self.blocks = nn.ModuleList([
            AlpamayoAttentionBlock(state_dict, i, hidden_size=hidden_size) for i in range(27)
        ])
        
        self.final_merger = AlpamayoMergerBlock(
            state_dict["vlm.model.visual.merger.norm.weight"].float(),
            state_dict["vlm.model.visual.merger.norm.bias"].float(),
            state_dict["vlm.model.visual.merger.linear_fc1.weight"].float(),
            state_dict["vlm.model.visual.merger.linear_fc1.bias"].float(),
            state_dict["vlm.model.visual.merger.linear_fc2.weight"].float(),
            state_dict["vlm.model.visual.merger.linear_fc2.bias"].float(),
            hidden_size=hidden_size, merged_dim=4*hidden_size, out_features=out_features,
            use_postshuffle_norm=False,
        )
        
        self.deepstack_mergers = nn.ModuleList([
            AlpamayoMergerBlock(
                state_dict["vlm.model.visual.deepstack_merger_list.%d.norm.weight" % i].float(),
                state_dict["vlm.model.visual.deepstack_merger_list.%d.norm.bias" % i].float(),
                state_dict["vlm.model.visual.deepstack_merger_list.%d.linear_fc1.weight" % i].float(),
                state_dict["vlm.model.visual.deepstack_merger_list.%d.linear_fc1.bias" % i].float(),
                state_dict["vlm.model.visual.deepstack_merger_list.%d.linear_fc2.weight" % i].float(),
                state_dict["vlm.model.visual.deepstack_merger_list.%d.linear_fc2.bias" % i].float(),
                hidden_size=hidden_size, merged_dim=4*hidden_size, out_features=out_features,
                use_postshuffle_norm=True,
            ) for i in range(len(deepstack_layers))
        ])
        
        total_params = sum(p.numel() for p in self.parameters())
        print("[AlpamayoVisionEncoder] Built real vision encoder (27 blocks, hidden=1152, out=4096). Params: %s" % format(total_params, ","))
    
    def forward(self, pixel_values, output_hidden_states=False):
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
        all_hidden = [x]  # index 0: after embedding
        
        deepstack_cached = [None] * len(self.deepstack_layers)
        for block_idx, block in enumerate(self.blocks):
            x = block(x)
            all_hidden.append(x)  # indices 1-27: after each block
            if block_idx in self.deepstack_layers:
                ds_idx = self.deepstack_layers.index(block_idx)
                deepstack_cached[ds_idx] = x
        
        final_out = self.final_merger(x, mean_pool=True)
        
        deepstack_features = []
        for ds_idx, cached_hidden in enumerate(deepstack_cached):
            if cached_hidden is None:
                raise RuntimeError("Deepstack layer %d not cached!" % self.deepstack_layers[ds_idx])
            merged = self.deepstack_mergers[ds_idx](cached_hidden, mean_pool=True)
            deepstack_features.append(merged)
        
        if output_hidden_states:
            class ModelOutput:
                def __init__(self, lhs, hss):
                    self.last_hidden_state = lhs
                    self.hidden_states = hss
            return ModelOutput(final_out, tuple(all_hidden))
        
        return final_out, deepstack_features
