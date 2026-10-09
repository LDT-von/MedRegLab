"""
Deformation Field Utilities - 变形场工具

提供变形场的创建、操作、评估等功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import numpy as np


class DeformationField:
    """
    变形场工具类

    提供变形场的创建、组合、求逆等功能
    """

    @staticmethod
    def create_identity(
        size: Tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        创建单位变形场（零位移）

        Args:
            size: 空间尺寸，如 (H, W) 或 (D, H, W)
            device: 设备
            dtype: 数据类型

        Returns:
            torch.Tensor: 单位变形场 [D, H, W] 或 [2, H, W]
        """
        n_dims = len(size)
        shape = (n_dims,) + size if n_dims > 1 else (1, size[0])

        return torch.zeros(shape, device=device, dtype=dtype)

    @staticmethod
    def create_grid(
        size: Tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        创建标准化网格坐标

        Args:
            size: 空间尺寸
            device: 设备
            dtype: 数据类型

        Returns:
            torch.Tensor: 标准化网格坐标 [-1, 1]
        """
        n_dims = len(size)
        grid = []
        for dim_size in size:
            coords = torch.linspace(
                -1, 1, dim_size, device=device, dtype=dtype
            )
            grid.append(coords)

        grid = torch.meshgrid(*grid, indexing="ij")
        return torch.stack(grid, dim=0)

    @staticmethod
    def compose(
        phi1: torch.Tensor, phi2: torch.Tensor, order: int = 1
    ) -> torch.Tensor:
        """
        组合两个变形场

        计算 phi = phi1 ○ phi2，即先应用 phi2，再应用 phi1

        Args:
            phi1: 第一个变形场
            phi2: 第二个变形场
            order: 插值阶数

        Returns:
            torch.Tensor: 组合后的变形场
        """
        if phi1.dim() == 4:  # 2D
            grid = phi2.permute(0, 2, 3, 1)
            grid[..., 0] = 2 * grid[..., 0] / (phi1.shape[3] - 1) - 1
            grid[..., 1] = 2 * grid[..., 1] / (phi1.shape[2] - 1) - 1

            composed = F.grid_sample(
                phi1, grid, mode="bilinear", padding_mode="border", align_corners=True
            )
            composed = composed.permute(0, 2, 3, 1)
        else:
            raise NotImplementedError("3D composition not implemented")

        return composed

    @staticmethod
    def invert(phi: torch.Tensor, num_iter: int = 10) -> torch.Tensor:
        """
        计算逆变形场

        使用固定点迭代

        Args:
            phi: 原始变形场
            num_iter: 迭代次数

        Returns:
            torch.Tensor: 逆变形场
        """
        phi_inv = -phi.clone()

        for _ in range(num_iter):
            # phi_inv = phi_inv - (phi ○ phi_inv - identity)
            composed = DeformationField.compose(phi, phi_inv)
            if composed.dim() == 4:
                identity = DeformationField.create_grid(
                    (phi.shape[2], phi.shape[3]),
                    device=phi.device,
                ).permute(2, 0, 1).unsqueeze(0)
            else:
                identity = DeformationField.create_grid(
                    (phi.shape[2], phi.shape[3], phi.shape[4]),
                    device=phi.device,
                ).permute(3, 0, 1, 2).unsqueeze(0)

            error = composed - identity
            phi_inv = phi_inv - 0.5 * error

        return phi_inv

    @staticmethod
    def scale(phi: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """
        缩放变形场

        Args:
            phi: 变形场
            scale_factor: 缩放因子

        Returns:
            torch.Tensor: 缩放后的变形场
        """
        return phi * scale_factor

    @staticmethod
    def add_bias(phi: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        添加偏置到变形场

        Args:
            phi: 变形场
            bias: 偏置 [D] 或 [B, D]

        Returns:
            torch.Tensor: 添加偏置后的变形场
        """
        return phi + bias

    @staticmethod
    def smooth(phi: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        平滑变形场

        Args:
            phi: 变形场
            sigma: 高斯核 sigma

        Returns:
            torch.Tensor: 平滑后的变形场
        """
        if phi.dim() == 4:  # 2D
            # 创建高斯核
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1

            kernel = self._gaussian_kernel_2d(kernel_size, sigma)
            kernel = kernel.to(phi.device).to(phi.dtype)

            # 对每个通道分别平滑
            smoothed = torch.zeros_like(phi)
            for c in range(phi.shape[1]):
                smoothed[:, c : c + 1] = F.conv2d(
                    phi[:, c : c + 1],
                    kernel,
                    padding=kernel_size // 2,
                )

            return smoothed
        else:
            return phi

    @staticmethod
    def _gaussian_kernel_2d(size: int, sigma: float) -> torch.Tensor:
        """创建 2D 高斯核"""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        x, y = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        return kernel / kernel.sum()


class JacobianField:
    """
    Jacobian 行列式工具

    用于评估变形场的局部体积变化
    """

    @staticmethod
    def compute(phi: torch.Tensor) -> torch.Tensor:
        """
        计算变形场的 Jacobian 行列式

        Args:
            phi: 变形场 [B, D, H, W] 或 [B, D, D, H, W]

        Returns:
            torch.Tensor: Jacobian 行列式
        """
        if phi.dim() == 4:  # 2D
            return JacobianField._compute_2d(phi)
        else:  # 3D
            return JacobianField._compute_3d(phi)

    @staticmethod
    def _compute_2d(phi: torch.Tensor) -> torch.Tensor:
        """计算 2D Jacobian 行列式"""
        # 计算位移场的梯度
        # J[i,j,k] = d(u_i) / d(x_j) at position k

        B, C, H, W = phi.shape
        assert C == 2

        # 使用有限差分计算梯度
        dx_u = (phi[:, 0:1, 1:, :-1] - phi[:, 0:1, :-1, :-1]).clamp(-10, 10)
        dy_u = (phi[:, 0:1, :-1, 1:] - phi[:, 0:1, :-1, :-1]).clamp(-10, 10)
        dx_v = (phi[:, 1:2, 1:, :-1] - phi[:, 1:2, :-1, :-1]).clamp(-10, 10)
        dy_v = (phi[:, 1:2, :-1, 1:] - phi[:, 1:2, :-1, :-1]).clamp(-10, 10)

        # 雅可比矩阵的行列式
        # J = | 1 + du/dx   du/dy  |
        #     | dv/dx       1 + dv/dy |
        det = (1 + dx_u) * (1 + dy_v) - dx_v * dy_u

        return det

    @staticmethod
    def _compute_3d(phi: torch.Tensor) -> torch.Tensor:
        """计算 3D Jacobian 行列式（简化版）"""
        # 简化：使用体积变化的近似
        B, C, D, H, W = phi.shape
        assert C == 3

        dx = (phi[:, 0, 1:, :-1, :-1] - phi[:, 0, :-1, :-1, :-1]).clamp(-5, 5)
        dy = (phi[:, 1, :-1, 1:, :-1] - phi[:, 1, :-1, :-1, :-1]).clamp(-5, 5)
        dz = (phi[:, 2, :-1, :-1, 1:] - phi[:, 2, :-1, :-1, :-1]).clamp(-5, 5)

        # 简化的体积变化
        det = (1 + dx) * (1 + dy) * (1 + dz)

        return det

    @staticmethod
    def check_folding(jacobian: torch.Tensor, threshold: float = 0.0) -> dict:
        """
        检查折叠情况

        Args:
            jacobian: Jacobian 行列式
            threshold: 阈值

        Returns:
            dict: 折叠统计信息
        """
        neg_jac = jacobian < threshold
        num_folds = neg_jac.sum().item()
        total_voxels = jacobian.numel()
        fold_ratio = num_folds / total_voxels

        return {
            "num_folds": num_folds,
            "total_voxels": total_voxels,
            "fold_ratio": fold_ratio,
            "min_jacobian": jacobian.min().item(),
            "max_jacobian": jacobian.max().item(),
            "mean_jacobian": jacobian.mean().item(),
            "has_folding": num_folds > 0,
        }

    @staticmethod
    def compute_smoothness(phi: torch.Tensor) -> torch.Tensor:
        """
        计算变形场的平滑度（通过 Jacobian 的方差）

        Args:
            phi: 变形场

        Returns:
            torch.Tensor: 平滑度损失
        """
        jac = JacobianField.compute(phi)
        return torch.var(jac)
