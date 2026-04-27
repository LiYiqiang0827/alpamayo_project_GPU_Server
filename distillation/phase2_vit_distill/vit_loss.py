"""
Vision Encoder Distillation - Loss Functions
==============================================
Distillation loss combining:
- MSE loss on final output
- MSE loss on deepstack intermediate features
- L2 norm regularization

Supports configurable alpha weights via config dict.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ViTDistillationLoss(nn.Module):
    """
    Multi-component distillation loss for Vision Encoder.
    
    Components:
    1. final_mse:   MSE between projected teacher final and student final
    2. deepstack_mse: MSE across all deepstack layer pairs
    3. norm_reg:    L2 norm regularization on student features (prevents drift)
    
    Total loss = alpha_final * final_mse 
               + alpha_deepstack * deepstack_mse 
               + alpha_norm * norm_reg
    """

    def __init__(
        self,
        alpha_final: float = 1.0,
        alpha_deepstack: float = 0.5,
        alpha_norm: float = 0.01,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha_final = alpha_final
        self.alpha_deepstack = alpha_deepstack
        self.alpha_norm = alpha_norm
        self.reduction = reduction
        
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        teacher_final_proj: torch.Tensor,   # [B, student_dim]
        student_final: torch.Tensor,          # [B, student_dim]
        teacher_deepstack_proj: list[torch.Tensor],  # list of [B, student_dim]
        student_deepstack: list[torch.Tensor],        # list of [B, student_dim]
        return_components: bool = False,
    ) -> torch.Tensor | dict:
        """
        Compute distillation loss.
        
        Args:
            teacher_final_proj: Projected teacher final output, [B, D]
            student_final: Student final output, [B, D]
            teacher_deepstack_proj: Projected teacher deepstack features, list of [B, D]
            student_deepstack: Student deepstack features, list of [B, D]
            return_components: If True, return dict of individual loss components
            
        Returns:
            Total loss (scalar tensor), or dict with components if return_components=True
        """
        # 1. Final output MSE
        final_loss = self.mse_loss(teacher_final_proj, student_final)
        
        # 2. Deepstack MSE (average over all layers)
        if len(teacher_deepstack_proj) != len(student_deepstack):
            raise ValueError(
                f"Deepstack length mismatch: teacher={len(teacher_deepstack_proj)}, "
                f"student={len(student_deepstack)}"
            )
        
        deepstack_losses = []
        for i, (t_feat, s_feat) in enumerate(zip(teacher_deepstack_proj, student_deepstack)):
            if t_feat.shape != s_feat.shape:
                raise RuntimeError(
                    f"Shape mismatch at deepstack[{i}]: teacher={t_feat.shape}, student={s_feat.shape}"
                )
            deepstack_losses.append(self.mse_loss(t_feat, s_feat))
        
        deepstack_loss = torch.stack(deepstack_losses).mean()
        
        # 3. Norm regularization (L2 norm of student features, encourages reasonable magnitude)
        norm_reg = student_final.pow(2).mean()
        for s_feat in student_deepstack:
            norm_reg = norm_reg + s_feat.pow(2).mean()
        norm_reg = norm_reg / (1 + len(student_deepstack))  # average
        
        # Weighted sum
        total_loss = (
            self.alpha_final * final_loss
            + self.alpha_deepstack * deepstack_loss
            + self.alpha_norm * norm_reg
        )
        
        if return_components:
            return {
                "total_loss": total_loss,
                "final_loss": final_loss,
                "deepstack_loss": deepstack_loss,
                "norm_reg": norm_reg,
            }
        
        return total_loss


def vit_distillation_loss(
    teacher_final_proj: torch.Tensor,
    student_final: torch.Tensor,
    teacher_deepstack_proj: list[torch.Tensor],
    student_deepstack: list[torch.Tensor],
    alpha_final: float = 1.0,
    alpha_deepstack: float = 0.5,
    alpha_norm: float = 0.01,
    return_components: bool = False,
) -> torch.Tensor | dict:
    """
    Functional API for ViT distillation loss.
    
    Convenience wrapper around ViTDistillationLoss.
    
    Args:
        teacher_final_proj: Projected teacher final output, [B, D]
        student_final: Student final output, [B, D]
        teacher_deepstack_proj: Projected teacher deepstack features, list of [B, D]
        student_deepstack: Student deepstack features, list of [B, D]
        alpha_final: Weight for final output MSE
        alpha_deepstack: Weight for deepstack MSE
        alpha_norm: Weight for norm regularization
        return_components: If True, return dict with all components
        
    Returns:
        Total loss or dict of components
    """
    criterion = ViTDistillationLoss(
        alpha_final=alpha_final,
        alpha_deepstack=alpha_deepstack,
        alpha_norm=alpha_norm,
    )
    return criterion(
        teacher_final_proj,
        student_final,
        teacher_deepstack_proj,
        student_deepstack,
        return_components=return_components,
    )


class DistillationLossWithAlphaSchedule(nn.Module):
    """
    Distillation loss with linear alpha schedule (cosine annealing from init to target).
    
    Useful for warming up: start with higher final MSE weight, gradually shift to deepstack.
    """

    def __init__(
        self,
        alpha_final_init: float = 1.0,
        alpha_final_final: float = 0.7,
        alpha_deepstack_init: float = 0.3,
        alpha_deepstack_final: float = 0.8,
        alpha_norm: float = 0.01,
        total_steps: int = 100000,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.alpha_final_init = alpha_final_init
        self.alpha_final_final = alpha_final_final
        self.alpha_deepstack_init = alpha_deepstack_init
        self.alpha_deepstack_final = alpha_deepstack_final
        self.alpha_norm = alpha_norm
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        
        self.base_criterion = ViTDistillationLoss(
            alpha_final=alpha_final_init,
            alpha_deepstack=alpha_deepstack_init,
            alpha_norm=alpha_norm,
        )
        self._step = 0

    def set_step(self, step: int):
        """Update current step for schedule."""
        self._step = step
        
    def _get_current_alphas(self) -> tuple[float, float]:
        """Compute current alpha values based on linear schedule."""
        if self._step >= self.total_steps:
            return self.alpha_final_final, self.alpha_deepstack_final
        
        progress = self._step / max(self.total_steps, 1)
        alpha_final = self.alpha_final_init + progress * (self.alpha_final_final - self.alpha_final_init)
        alpha_deepstack = self.alpha_deepstack_init + progress * (self.alpha_deepstack_final - self.alpha_deepstack_init)
        return alpha_final, alpha_deepstack

    def forward(self, *args, **kwargs) -> torch.Tensor | dict:
        alpha_final, alpha_deepstack = self._get_current_alphas()
        criterion = ViTDistillationLoss(
            alpha_final=alpha_final,
            alpha_deepstack=alpha_deepstack,
            alpha_norm=self.alpha_norm,
        )
        return criterion(*args, **kwargs)


# ----------------------------------------------------------------------
# Cosine similarity loss (alternative / auxiliary)
# ----------------------------------------------------------------------

class CosineSimilarityLoss(nn.Module):
    """
    Cosine similarity loss (1 - cos) for final and deepstack features.
    Use as auxiliary loss alongside MSE.
    """

    def __init__(self, weight_final: float = 0.1, weight_deepstack: float = 0.05):
        super().__init__()
        self.weight_final = weight_final
        self.weight_deepstack = weight_deepstack

    def forward(
        self,
        teacher_final_proj: torch.Tensor,
        student_final: torch.Tensor,
        teacher_deepstack_proj: list[torch.Tensor],
        student_deepstack: list[torch.Tensor],
    ) -> torch.Tensor:
        # Final
        cos_final = F.cosine_similarity(teacher_final_proj, student_final, dim=-1)
        loss_final = 1.0 - cos_final.mean()
        
        # Deepstack
        losses = [loss_final * self.weight_final]
        for t, s in zip(teacher_deepstack_proj, student_deepstack):
            cos_sim = F.cosine_similarity(t, s, dim=-1)
            losses.append((1.0 - cos_sim).mean() * self.weight_deepstack)
        
        return sum(losses)


if __name__ == "__main__":
    print("=== ViT Distillation Loss smoke test ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B, D = 4, 2048
    torch.manual_seed(42)
    
    # Dummy features
    teacher_final_proj = torch.randn(B, D, device=device)
    student_final = torch.randn(B, D, device=device)
    teacher_deepstack = [torch.randn(B, D, device=device) for _ in range(3)]
    student_deepstack = [torch.randn(B, D, device=device) for _ in range(3)]
    
    # Test basic loss
    loss_fn = ViTDistillationModule().to(device) if False else None
    
    loss = vit_distillation_loss(
        teacher_final_proj, student_final,
        teacher_deepstack, student_deepstack,
        alpha_final=1.0, alpha_deepstack=0.5, alpha_norm=0.01,
    )
    print(f"Total loss: {loss.item():.6f}")
    
    # Test with components
    components = vit_distillation_loss(
        teacher_final_proj, student_final,
        teacher_deepstack, student_deepstack,
        alpha_final=1.0, alpha_deepstack=0.5, alpha_norm=0.01,
        return_components=True,
    )
    print("Components:")
    for k, v in components.items():
        print(f"  {k}: {v.item():.6f}")
    
    # Test cosine similarity loss
    cos_loss_fn = CosineSimilarityLoss().to(device)
    cos_loss = cos_loss_fn(
        teacher_final_proj, student_final,
        teacher_deepstack, student_deepstack,
    )
    print(f"Cosine similarity loss: {cos_loss.item():.6f}")
    
    # Test scheduled loss
    sched_loss_fn = DistillationLossWithAlphaSchedule(total_steps=10000).to(device)
    for test_step in [0, 5000, 10000]:
        sched_loss_fn.set_step(test_step)
        alphas = sched_loss_fn._get_current_alphas()
        print(f"Step {test_step}: alpha_final={alphas[0]:.4f}, alpha_deepstack={alphas[1]:.4f}")
    
    print("✓ Loss tests passed!")
