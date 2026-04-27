"""
Vision Encoder Distillation - Projector Layers
===============================================
Projection layers for matching Teacher (4096-dim) to Student (2048-dim).

- FinalOutputProjector: 4096 → 2048 (final output projection)
- DeepstackProjector:   4096 → 2048 (intermediate layer projections, ×3)
- VisionDistillationModule: Full distillation module wrapping projector + student.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class FinalOutputProjector(nn.Module):
    """
    Projects Teacher final output (4096-dim) to Student dimension (2048-dim).
    
    Architecture: LayerNorm → Linear → GELU → Linear → LayerNorm + Residual
    """

    def __init__(self, teacher_dim: int = 4096, student_dim: int = 2048, 
                 hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        
        self.norm1 = nn.LayerNorm(teacher_dim)
        self.proj = nn.Sequential(
            nn.Linear(teacher_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, student_dim),
        )
        # Optional residual projection (if dims match, use identity; otherwise linear)
        if teacher_dim != student_dim:
            self.residual_proj = nn.Linear(teacher_dim, student_dim)
        else:
            self.residual_proj = nn.Identity()
        
        self._init_weights()

    def _init_weights(self):
        for module in self.proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if not isinstance(self.residual_proj, nn.Identity):
            nn.init.xavier_uniform_(self.residual_proj.weight, gain=0.5)
            nn.init.zeros_(self.residual_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Teacher final output, shape [B, teacher_dim]
        Returns:
            Projected tensor, shape [B, student_dim]
        """
        normed = self.norm1(x)
        projected = self.proj(normed)
        residual = self.residual_proj(x)
        return projected + residual


class DeepstackProjector(nn.Module):
    """
    Projects Teacher deepstack intermediate features (4096-dim) to Student dimension (2048-dim).
    
    Three projectors for layers [8, 16, 24] of Teacher → [5, 11, 17] of Student.
    
    Architecture: LayerNorm → Linear → GELU → Linear → LayerNorm + Residual
    """

    def __init__(self, teacher_dim: int = 4096, student_dim: int = 2048,
                 hidden_dim: int = 1024, dropout: float = 0.1, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        
        # Create separate projectors for each deepstack layer
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(teacher_dim),
                nn.Linear(teacher_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, student_dim),
            )
            for _ in range(num_layers)
        ])
        
        # Residual projections
        self.residual_projs = nn.ModuleList([
            nn.Linear(teacher_dim, student_dim) if teacher_dim != student_dim else nn.Identity()
            for _ in range(num_layers)
        ])
        
        self._init_weights()

    def _init_weights(self):
        for proj in self.projectors:
            for module in proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        for res_proj in self.residual_projs:
            if not isinstance(res_proj, nn.Identity):
                nn.init.xavier_uniform_(res_proj.weight, gain=0.5)
                nn.init.zeros_(res_proj.bias)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            features: List of 3 teacher deepstack tensors, each [B, teacher_dim]
        Returns:
            List of 3 projected tensors, each [B, student_dim]
        """
        if len(features) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} deepstack features, got {len(features)}"
            )
        outputs = []
        for i, feat in enumerate(features):
            normed = self.projectors[i][0](feat)  # LayerNorm
            projected = self.projectors[i][1:](normed)  # Linear → GELU → Dropout → Linear
            residual = self.residual_projs[i](feat)
            outputs.append(projected + residual)
        return outputs


class VisionDistillationModule(nn.Module):
    """
    Complete Vision Encoder Distillation Module.
    
    Wraps:
    - Student Vision Encoder (frozen Teacher loaded separately for reference)
    - FinalOutputProjector: 4096 → 2048
    - DeepstackProjector:   4096 → 2048 × 3
    
    Usage:
        module = VisionDistillationModule()
        student_final, student_deepstack = module(student_input)
        # Compare with teacher outputs (projected via module)
    """

    def __init__(self,
                 teacher_dim: int = 4096,
                 student_dim: int = 2048,
                 deepstack_teacher_layers: list[int] = [8, 16, 24],
                 deepstack_student_layers: list[int] = [5, 11, 17],
                 projector_hidden_dim: int = 1024,
                 dropout: float = 0.1,
                 deepstack_teacher_dim: int = None,
                 deepstack_student_dim: int = None):
        super().__init__()
        
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.deepstack_teacher_layers = deepstack_teacher_layers
        self.deepstack_student_layers = deepstack_student_layers
        
        # Use hidden dims for deepstack (raw transformer output), output dims for final
        deepstack_t_dim = deepstack_teacher_dim if deepstack_teacher_dim is not None else teacher_dim
        deepstack_s_dim = deepstack_student_dim if deepstack_student_dim is not None else student_dim
        
        # Projectors
        self.final_projector = FinalOutputProjector(
            teacher_dim=teacher_dim,
            student_dim=student_dim,
            hidden_dim=projector_hidden_dim,
            dropout=dropout,
        )
        
        self.deepstack_projector = DeepstackProjector(
            teacher_dim=deepstack_t_dim,
            student_dim=deepstack_s_dim,
            hidden_dim=projector_hidden_dim,
            dropout=dropout,
            num_layers=len(deepstack_teacher_layers),
        )
        
        # Student ViT placeholder (to be loaded from checkpoint)
        self.student_vit = None

        # Projection from raw transformer hidden dim to teacher_dim for deepstack features
        # DummyTeacherViT uses hidden_dim=1152, teacher projects to teacher_dim=4096
        self.deepstack_hidden_proj = nn.Linear(1152, teacher_dim)

        # Projection for student deepstack features: student_hidden_dim -> student_dim
        # Student deepstack returns raw CLS tokens [B, student_hidden_dim] (1024),
        # needs projection to [B, student_dim] (2048)
        self.deepstack_student_proj = nn.Linear(1024, student_dim)  # project from student_hidden=1024 to student_output=2048

    def set_student_vit(self, student_vit: nn.Module):
        """Set the student Vision Transformer module."""
        self.student_vit = student_vit
        
    def forward_student(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Run student ViT forward pass, returning final output + deepstack features.
        
        Args:
            pixel_values: [B, C, H, W] image tensor
            
        Returns:
            final_output: [B, student_dim]
            deepstack_features: list of [B, student_dim] tensors (3 items)
        """
        if self.student_vit is None:
            raise RuntimeError("Student ViT not set. Call set_student_vit() first.")
        
        # This should be implemented based on actual student ViT architecture
        # Placeholder: assume student_vit returns (final_out, deepstack_list)
        final_out, deepstack_raw = self.student_vit(pixel_values)
        # deepstack_raw: list of [B, student_hidden_dim] (1024) - unprojected CLS tokens
        # final_out: [B, student_dim] (2048) - already projected
        # Project deepstack from student_hidden_dim to student_dim
        deepstack_projected = [self.deepstack_student_proj(feat) for feat in deepstack_raw]
        return final_out, deepstack_projected
    
    def get_teacher_features(self, pixel_values: torch.Tensor, teacher_model: nn.Module) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Extract teacher features (final + deepstack).
        
        Args:
            pixel_values: [B, C, H, W] image tensor
            teacher_model: Full teacher model (Alpamayo1.5-10B)
            
        Returns:
            final_output: [B, teacher_dim]
            deepstack_features: list of [B, teacher_dim] tensors (3 items)
        """
        with torch.no_grad():
            outputs = teacher_model(pixel_values, output_hidden_states=True)
            final_output = outputs.last_hidden_state  # or appropriate final output
            hidden_states = outputs.hidden_states  # tuple of all layer outputs
            
        # Extract deepstack features at specified layers
        # Each hidden_state is [B, 197, hidden_dim], extract CLS token [:, 0] and project to teacher_dim
        deepstack_features = []
        for layer_idx in self.deepstack_teacher_layers:
            # hidden_states are 0-indexed (0 = embeddings, 1 = layer 1, ...)
            if layer_idx < len(hidden_states):
                cls_token = hidden_states[layer_idx][:, 0, :]  # [B, hidden_dim]
                projected = self.deepstack_hidden_proj(cls_token)  # [B, teacher_dim]
                deepstack_features.append(projected)
            else:
                raise ValueError(
                    f"Teacher deepstack layer {layer_idx} out of range "
                    f"(only {len(hidden_states)} total layers)"
                )

        # Also project final_output if it has sequence dim
        if final_output.dim() == 3:
            final_output = final_output[:, 0, :]  # [B, hidden_dim]
            final_output = self.deepstack_hidden_proj(final_output)  # [B, teacher_dim]

        return final_output, deepstack_features

    def project_teacher_features(
        self, 
        teacher_final: torch.Tensor, 
        teacher_deepstack: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Project teacher features to student dimension.
        
        Args:
            teacher_final: [B, teacher_dim]
            teacher_deepstack: list of [B, teacher_dim] tensors
            
        Returns:
            projected_final: [B, student_dim]
            projected_deepstack: list of [B, student_dim] tensors
        """
        projected_final = self.final_projector(teacher_final)
        projected_deepstack = self.deepstack_projector(teacher_deepstack)
        return projected_final, projected_deepstack


def build_vision_distillation_module(
    teacher_dim: int = 4096,
    student_dim: int = 2048,
    deepstack_teacher_layers: list[int] = [8, 16, 24],
    deepstack_student_layers: list[int] = [5, 11, 17],
    projector_hidden_dim: int = 1024,
    deepstack_teacher_dim: int = None,
    deepstack_student_dim: int = None,
) -> VisionDistillationModule:
    """Factory function to build the VisionDistillationModule."""
    return VisionDistillationModule(
        teacher_dim=teacher_dim,
        student_dim=student_dim,
        deepstack_teacher_layers=deepstack_teacher_layers,
        deepstack_student_layers=deepstack_student_layers,
        projector_hidden_dim=projector_hidden_dim,
        deepstack_teacher_dim=deepstack_teacher_dim,
        deepstack_student_dim=deepstack_student_dim,
    )



if __name__ == "__main__":
    # Quick smoke test
    print("=== VisionDistillationModule smoke test ===")
    
    B, C, H, W = 2, 3, 224, 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    module = VisionDistillationModule().to(device)
    
    # Test projectors individually
    teacher_final = torch.randn(B, 4096, device=device)
    teacher_deepstack = [torch.randn(B, 4096, device=device) for _ in range(3)]
    
    proj_final = module.final_projector(teacher_final)
    proj_deepstack = module.deepstack_projector(teacher_deepstack)
    
    print(f"Teacher final: {teacher_final.shape} → Projected: {proj_final.shape}")
    assert proj_final.shape == (B, 2048), f"Expected (B, 2048), got {proj_final.shape}"
    
    for i, (orig, proj) in enumerate(zip(teacher_deepstack, proj_deepstack)):
        print(f"Deepstack[{i}]: {orig.shape} → Projected: {proj.shape}")
        assert proj.shape == (B, 2048)
    
    print("✓ All projector tests passed!")
    
    # Test full module
    try:
        pixel_values = torch.randn(B, C, H, W, device=device)
        # This will fail without actual student_vit set, which is expected
        module.set_student_vit(nn.Identity())  # dummy for test
        student_final, student_deepstack = module.forward_student(pixel_values)
        print(f"Student forward pass OK: final={student_final.shape}")
    except Exception as e:
        print(f"(Expected) Student forward needs real ViT: {e}")
    
    total_params = sum(p.numel() for p in module.parameters())
    print(f"Total module parameters: {total_params:,}")
    print("✓ Smoke test complete!")
