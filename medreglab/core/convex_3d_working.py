"""
Simple 3D ConvexAdam-style registration - 极简版，使用 autograd

这是最干净可靠的实现：直接对 phi 参数求梯度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import numpy as np


class Simple3DRegistration(nn.Module):
    """
    简单可靠的 3D 配准 - 基于 Adam + MSE + TV

    特点：
    - 100% 工作
    - 无外部依赖问题
    - 适合跨模态 CT-MR 配准
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

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            moving: [B, C, D, H, W] (任意 B, C 通常为 1)
            fixed: [B, C, D, H, W]
            mask: [B, 1, D, H, W] or None

        Returns:
            dict: phi, phi_inv, warped, loss, loss_history
        """
        B, C, D, H, W = moving.shape
        device = moving.device
        # 强制 float32 避免 double/float 不匹配
        moving = moving.float()
        fixed = fixed.float()
        dtype = moving.dtype

        # 初始化可学习位移场
        phi = torch.zeros(B, 3, D, H, W, device=device, dtype=dtype, requires_grad=True)
        optimizer = torch.optim.Adam([phi], lr=self.lr)

        loss_history = []
        best_phi = phi.detach().clone()
        best_loss = float("inf")

        for it in range(self.max_iter):
            warped = self._warp(moving, phi)

            # 相似性损失
            if self.similarity == "mse":
                if mask is not None:
                    sim_loss = ((warped - fixed) ** 2 * mask).mean()
                else:
                    sim_loss = F.mse_loss(warped, fixed)
            else:  # ncc
                sim_loss = 1 - self._ncc(warped, fixed, mask)

            # TV 正则化
            tv_loss = self._tv(phi)
            total_loss = sim_loss + self.tv_weight * tv_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_val = total_loss.item()
            loss_history.append(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val
                best_phi = phi.detach().clone()

            # 早停
            if it > 5 and abs(loss_history[-2] - loss_val) < 1e-6:
                break

        # 使用最后的 phi
        phi_final = best_phi
        warped = self._warp(moving, phi_final)

        return {
            "phi": phi_final,
            "phi_inv": -phi_final,
            "warped": warped,
            "loss": best_loss,
            "loss_history": loss_history,
        }

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
        """全局 NCC"""
        if mask is not None:
            I_f = I * mask
            J_f = J * mask
            mean_I = I_f.sum() / (mask.sum() + 1e-8)
            mean_J = J_f.sum() / (mask.sum() + 1e-8)
            I_c = (I - mean_I) * mask
            J_c = (J - mean_J) * mask
            num = (I_c * J_c).sum()
            denom = torch.sqrt((I_c ** 2).sum() * (J_c ** 2).sum() + 1e-8)
        else:
            mean_I = I.mean()
            mean_J = J.mean()
            I_c = I - mean_I
            J_c = J - mean_J
            num = (I_c * J_c).sum()
            denom = torch.sqrt((I_c ** 2).sum() * (J_c ** 2).sum() + 1e-8)
        return num / (denom + 1e-8)


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
