"""
Deformation Field Utilities - 变形场工具

提供变形场的创建、操作、评估等功能
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from numbers import Integral
import math


class DeformationField:
    """
    变形场工具类

    批量位移场为 [B, 2, H, W] 或 [B, 3, D, H, W]。
    分量顺序始终是 (x=W, y=H[, z=D])，位移单位是体素索引。
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
            torch.Tensor: 未带 batch 维度的零位移场 [2, H, W] 或 [3, D, H, W]
        """
        n_dims = len(size)
        if n_dims not in (2, 3) or any(dim < 1 for dim in size):
            raise ValueError("size must contain 2 or 3 positive spatial dimensions")
        shape = (n_dims,) + tuple(size)

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
            torch.Tensor: [2, H, W] 或 [3, D, H, W]，
                分量顺序 (x, y[, z])，坐标范围 [-1, 1]。
        """
        if len(size) not in (2, 3) or any(dim < 2 for dim in size):
            raise ValueError("size must contain 2 or 3 spatial dimensions >= 2")
        grid = []
        for dim_size in size:
            coords = torch.linspace(
                -1, 1, dim_size, device=device, dtype=dtype
            )
            grid.append(coords)

        grid = torch.meshgrid(*grid, indexing="ij")
        return torch.stack(tuple(reversed(grid)), dim=0)

    @staticmethod
    def _validate_field(phi: torch.Tensor):
        if phi.ndim not in (4, 5) or phi.shape[1] != phi.ndim - 2:
            raise ValueError("field must have shape [B, 2, H, W] or [B, 3, D, H, W]")
        if phi.shape[0] < 1 or min(phi.shape[2:]) < 2:
            raise ValueError("field batch must be non-empty and spatial dimensions >= 2")
        if not torch.is_floating_point(phi):
            raise ValueError("field must use a floating-point dtype")
        if not torch.isfinite(phi).all():
            raise ValueError("field must contain only finite values")

    @staticmethod
    def _sample_at_displacement(source: torch.Tensor, displacement: torch.Tensor, order: int):
        size = tuple(source.shape[2:])
        base = DeformationField.create_grid(size, device=source.device, dtype=source.dtype).unsqueeze(0)
        scale = torch.as_tensor([2.0 / (length - 1) for length in reversed(size)],
                                device=source.device, dtype=source.dtype)
        scale = scale.reshape(1, len(size), *([1] * len(size)))
        grid = (base + displacement * scale).movedim(1, -1)
        return F.grid_sample(source, grid, mode="nearest" if order == 0 else "bilinear",
                             padding_mode="border", align_corners=True)

    @staticmethod
    def compose(
        phi1: torch.Tensor, phi2: torch.Tensor, order: int = 1
    ) -> torch.Tensor:
        """
        组合两个变形场

        计算 phi = phi1 ○ phi2，即先应用 phi2，再应用 phi1：
        u(x) = phi2(x) + phi1(x + phi2(x))。边界以最近边界值延拓。

        Args:
            phi1: 第一个变形场
            phi2: 第二个变形场
            order: 0 为最近邻，1 为线性/三线性插值

        Returns:
            torch.Tensor: 与输入相同形状的体素位移场
        """
        DeformationField._validate_field(phi1)
        DeformationField._validate_field(phi2)
        if phi1.shape != phi2.shape or phi1.device != phi2.device or phi1.dtype != phi2.dtype:
            raise ValueError("fields must have identical shapes, devices and dtypes")
        if order not in (0, 1):
            raise ValueError("interpolation order must be 0 or 1")
        return phi2 + DeformationField._sample_at_displacement(phi1, phi2, order)

    @staticmethod
    def invert(phi: torch.Tensor, num_iter: int = 10, tolerance: float = 1e-3) -> torch.Tensor:
        """
        计算逆变形场

        使用固定点迭代及双向残差验证。不收敛时抛出异常；
        有限网格上的残差检查并不能证明全局可逆。

        Args:
            phi: 原始变形场
            num_iter: 迭代次数
            tolerance: 体素单位的最大双向残差

        Returns:
            torch.Tensor: 逆变形场
        """
        DeformationField._validate_field(phi)
        if isinstance(num_iter, bool) or not isinstance(num_iter, Integral) or num_iter < 1:
            raise ValueError("num_iter must be a positive integer")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        phi_inv = -phi.clone()
        for _ in range(num_iter):
            phi_inv = -DeformationField._sample_at_displacement(phi, phi_inv, 1)
        forward_residual = DeformationField.compose(phi, phi_inv).abs().max().item()
        reverse_residual = DeformationField.compose(phi_inv, phi).abs().max().item()
        if not math.isfinite(forward_residual) or not math.isfinite(reverse_residual):
            raise FloatingPointError("Non-finite inverse composition residual")
        if max(forward_residual, reverse_residual) > tolerance:
            raise RuntimeError("Inverse did not converge on the finite grid: "
                               f"forward={forward_residual:.6g}, reverse={reverse_residual:.6g}")
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
        DeformationField._validate_field(phi)
        if bias.device != phi.device or bias.dtype != phi.dtype:
            raise ValueError("bias must share the field's dtype and device")
        if bias.ndim == 1 and bias.shape[0] == phi.shape[1]:
            bias = bias.reshape(1, phi.shape[1], *([1] * (phi.ndim - 2)))
        elif bias.ndim == 2 and bias.shape == phi.shape[:2]:
            bias = bias.reshape(phi.shape[0], phi.shape[1], *([1] * (phi.ndim - 2)))
        else:
            raise ValueError("bias must have shape [C] or [B, C]")
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
        DeformationField._validate_field(phi)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        radius = max(1, math.ceil(3 * sigma))
        coords = torch.arange(-radius, radius + 1, device=phi.device, dtype=phi.dtype)
        kernel = torch.exp(-0.5 * (coords / sigma).square())
        kernel = kernel / kernel.sum()
        dimensions = phi.ndim - 2
        channels = phi.shape[1]
        smoothed = phi
        for axis in range(dimensions):
            kernel_shape = [1] * dimensions
            kernel_shape[axis] = kernel.numel()
            weight = kernel.reshape(1, 1, *kernel_shape).repeat(channels, 1, *([1] * dimensions))
            padding = [0] * (2 * dimensions)
            reverse_axis = dimensions - axis - 1
            padding[2 * reverse_axis:2 * reverse_axis + 2] = [radius, radius]
            padded = F.pad(smoothed, padding, mode="replicate")
            convolution = F.conv2d if dimensions == 2 else F.conv3d
            smoothed = convolution(padded, weight, groups=channels)
        return smoothed

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
            phi: 体素位移场 [B, 2, H, W] 或 [B, 3, D, H, W]，
                分量顺序为 (x, y[, z])，各空间轴步长均为一个体素。

        Returns:
            torch.Tensor: Jacobian 行列式
        """
        DeformationField._validate_field(phi)
        if phi.dim() == 4:  # 2D
            return JacobianField._compute_2d(phi)
        else:  # 3D
            return JacobianField._compute_3d(phi)

    @staticmethod
    def _compute_2d(phi: torch.Tensor) -> torch.Tensor:
        """计算 2D Jacobian 行列式"""
        base = phi[:, :, :-1, :-1]
        dx = phi[:, :, :-1, 1:] - base
        dy = phi[:, :, 1:, :-1] - base
        return (1 + dx[:, 0:1]) * (1 + dy[:, 1:2]) - dy[:, 0:1] * dx[:, 1:2]

    @staticmethod
    def _compute_3d(phi: torch.Tensor) -> torch.Tensor:
        """完整的 det(I + du/dx)，使用 voxel 单位的前向差分。"""
        base = phi[:, :, :-1, :-1, :-1]
        dx = phi[:, :, :-1, :-1, 1:] - base
        dy = phi[:, :, :-1, 1:, :-1] - base
        dz = phi[:, :, 1:, :-1, :-1] - base
        a, b, c = 1 + dx[:, 0], dy[:, 0], dz[:, 0]
        d, e, f = dx[:, 1], 1 + dy[:, 1], dz[:, 1]
        g, h, i = dx[:, 2], dy[:, 2], 1 + dz[:, 2]
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

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
        if jacobian.numel() == 0 or not torch.isfinite(jacobian).all() or not math.isfinite(threshold):
            raise ValueError("jacobian and threshold must be finite and non-empty")
        neg_jac = jacobian <= threshold
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
        return torch.var(jac, unbiased=False)
