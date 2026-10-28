"""
Simple 3D registration - 极简版，使用 autograd

这是最干净可靠的实现：直接对 phi 参数求梯度（Adam on phi）。
注意：与 Learn2Reg 的凸优化 ConvexAdam 不是同一个东西，
这里没有凸性保证，也没有可学习参数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from numbers import Integral
import math


class Simple3DRegistration(nn.Module):
    """
    简单可靠的 3D 配准 - 基于 Adam + MSE + TV

    对每对输入直接优化采样位移场；不包含可训练的网络权重。
    """

    def __init__(
        self,
        max_iter: int = 100,
        lr: float = 0.1,
        tv_weight: float = 0.01,
        similarity: str = "ncc",  # "mse" or "ncc"
    ):
        super().__init__()
        self.max_iter = max_iter
        self.lr = lr
        self.tv_weight = tv_weight
        self.similarity = similarity
        self._validate_parameters()

    def _validate_parameters(self):
        # These attributes may also be changed by a CLI after construction.
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, Integral) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not math.isfinite(self.tv_weight) or self.tv_weight < 0:
            raise ValueError("tv_weight must be finite and non-negative")
        if self.similarity not in ("mse", "ncc"):
            raise ValueError("similarity must be 'mse' or 'ncc'")

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            moving: [B, C, D, H, W] (任意 B, C 通常为 1)
            fixed: [B, C, D, H, W]
            mask: [B, 1, D, H, W] or None

        Returns:
            dict: phi, phi_inv, warped, loss, loss_history. ``phi_inv`` is
            None: the inverse of a spatially varying field is not -phi and
            may not exist at all when the deformation folds.
        """
        self._validate_parameters()
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Registration needs internal gradients; use torch.no_grad(), not torch.inference_mode()")
        if moving.ndim != 5 or fixed.ndim != 5:
            raise ValueError("moving and fixed must have shape [B, C, D, H, W]")
        if moving.shape != fixed.shape:
            raise ValueError("moving and fixed must have identical shapes")
        if moving.device != fixed.device:
            raise ValueError("moving and fixed must be on the same device")
        B, C, D, H, W = moving.shape
        if B < 1 or C < 1 or min(D, H, W) < 2:
            raise ValueError("batch/channels must be non-empty and every spatial dimension must be >= 2")
        device = moving.device
        # Inputs are data, not learnable parameters of this per-case optimizer.
        moving = moving.detach().float()
        fixed = fixed.detach().float()
        if not torch.isfinite(moving).all() or not torch.isfinite(fixed).all():
            raise ValueError("moving and fixed must contain only finite values")
        if mask is not None:
            if mask.device != device or mask.ndim != 5:
                raise ValueError("mask must be a 5D tensor on the input device")
            if mask.shape[0] != B or mask.shape[1] not in (1, C) or mask.shape[2:] != (D, H, W):
                raise ValueError("mask must have shape [B, 1 or C, D, H, W]")
            mask = mask.detach().float()
            if not torch.isfinite(mask).all() or (mask < 0).any():
                raise ValueError("mask weights must be finite and non-negative")
            if (mask.sum(dim=(1, 2, 3, 4)) <= 0).any():
                raise ValueError("each sample mask must contain positive weight")
        dtype = moving.dtype

        # The optimizer needs autograd even when its caller uses no_grad().
        with torch.enable_grad():
            phi = torch.zeros(B, 3, D, H, W, device=device, dtype=dtype, requires_grad=True)
            optimizer = torch.optim.Adam([phi], lr=self.lr)
            loss_history = []
            best_phi = phi.detach().clone()
            best_loss = float("inf")

            for it in range(self.max_iter):
                warped = self._warp(moving, phi)
                total_loss = self._objective(warped, fixed, phi, mask)
                loss_val = total_loss.item()
                if not math.isfinite(loss_val):
                    raise FloatingPointError(f"Non-finite registration loss at iteration {it}")
                loss_history.append(loss_val)
                # Capture the exact field used to compute this loss, before the
                # optimizer mutates it. The previous code saved the next field.
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_phi = phi.detach().clone()

                optimizer.zero_grad()
                total_loss.backward()
                if phi.grad is None or not torch.isfinite(phi.grad).all():
                    raise FloatingPointError(f"Non-finite registration gradient at iteration {it}")
                optimizer.step()

                if it > 5 and abs(loss_history[-2] - loss_val) < 1e-6:
                    break

        # The final update has not yet been evaluated in the loop. Include it
        # as a candidate so max_iter=1 can return the first optimized field.
        with torch.no_grad():
            if not torch.isfinite(phi).all():
                raise FloatingPointError("Non-finite final registration field")
            final_warped = self._warp(moving, phi)
            final_loss = self._objective(final_warped, fixed, phi, mask).item()
            if not math.isfinite(final_loss):
                raise FloatingPointError("Non-finite final registration loss")
            if final_loss < best_loss:
                best_loss = final_loss
                best_phi = phi.detach().clone()
            phi_final = best_phi
            warped = self._warp(moving, phi_final)

        return {
            "phi": phi_final,
            "phi_inv": None,
            "warped": warped,
            "loss": best_loss,
            "loss_history": loss_history,
        }

    def _objective(self, warped, fixed, phi, mask):
        if self.similarity == "mse":
            if mask is not None:
                weights = mask.expand_as(warped)
                axes = (1, 2, 3, 4)
                weighted_error = ((warped - fixed).square() * weights).sum(dim=axes)
                similarity_loss = (weighted_error / weights.sum(dim=axes)).mean()
            else:
                similarity_loss = F.mse_loss(warped, fixed)
        else:
            similarity_loss = 1 - self._ncc(warped, fixed, mask)
        return similarity_loss + self.tv_weight * self._tv(phi)

    def _warp(self, image: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """3D 形变"""
        B, C, D, H, W = image.shape
        dtype = image.dtype
        device = image.device

        # 基础网格
        grid_d = torch.linspace(-1, 1, D, device=device, dtype=dtype)
        grid_h = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        grid_w = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        grid_d, grid_h, grid_w = torch.meshgrid(grid_d, grid_h, grid_w, indexing="ij")
        # [D, H, W, 3] 顺序 (W, H, D) for F.grid_sample
        base_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1)
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)

        # 归一化 phi (像素位移 -> [-1, 1] 范围)
        # phi shape: [B, 3, D, H, W] (axis 0 = x=W, 1 = y=H, 2 = z=D)
        phi_norm = phi.clone()
        phi_norm[:, 0] = 2 * phi_norm[:, 0] / max(W - 1, 1)  # x
        phi_norm[:, 1] = 2 * phi_norm[:, 1] / max(H - 1, 1)  # y
        phi_norm[:, 2] = 2 * phi_norm[:, 2] / max(D - 1, 1)  # z

        # 转置为 [B, D, H, W, 3]
        warped_grid = base_grid + phi_norm.permute(0, 2, 3, 4, 1)

        return F.grid_sample(
            image, warped_grid, mode="bilinear", padding_mode="border", align_corners=True
        )

    def _tv(self, phi: torch.Tensor) -> torch.Tensor:
        """全变差正则化"""
        diff_x = phi[:, :, :, :, 1:] - phi[:, :, :, :, :-1]
        diff_y = phi[:, :, :, 1:, :] - phi[:, :, :, :-1, :]
        diff_z = phi[:, :, 1:, :, :] - phi[:, :, :-1, :, :]
        return (diff_x.abs().mean() + diff_y.abs().mean() + diff_z.abs().mean()) / 3

    def _ncc(
        self,
        I: torch.Tensor,
        J: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-sample global NCC; channel and mask weights are counted once."""
        axes = (1, 2, 3, 4)
        if mask is not None:
            weights = mask.expand_as(I)
            weight_sum = weights.sum(dim=axes, keepdim=True)
            mean_I = (I * weights).sum(dim=axes, keepdim=True) / weight_sum
            mean_J = (J * weights).sum(dim=axes, keepdim=True) / weight_sum
            I_c, J_c = I - mean_I, J - mean_J
            num = (I_c * J_c * weights).sum(dim=axes)
            variance_I = (I_c.square() * weights).sum(dim=axes)
            variance_J = (J_c.square() * weights).sum(dim=axes)
        else:
            mean_I = I.mean(dim=axes, keepdim=True)
            mean_J = J.mean(dim=axes, keepdim=True)
            I_c = I - mean_I
            J_c = J - mean_J
            num = (I_c * J_c).sum(dim=axes)
            variance_I = I_c.square().sum(dim=axes)
            variance_J = J_c.square().sum(dim=axes)
        denom = torch.sqrt(variance_I * variance_J + 1e-8)
        return (num / (denom + 1e-8)).mean()


def register_3d_simple(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    max_iter: int = 100,
    lr: float = 0.1,
    tv_weight: float = 0.01,
    similarity: str = "mse",
) -> Dict[str, torch.Tensor]:
    """3D 配准的便捷函数"""
    reg = Simple3DRegistration(max_iter=max_iter, lr=lr, tv_weight=tv_weight, similarity=similarity)
    return reg(moving, fixed)
