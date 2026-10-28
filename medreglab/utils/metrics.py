"""
Registration Metrics - 配准评估指标

提供常用的医学图像配准评估指标
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Union, Tuple
from scipy.ndimage import label
import SimpleITK as sitk


class RegistrationMetrics:
    """
    配准评估指标计算器

    支持的指标：
    - Dice: 重叠系数
    - ASD: 平均表面距离
    - HD: Hausdorff 距离
    - NCC: 归一化互相关
    - SSIM: 结构相似性指数
    """

    def __init__(self, spacing: Optional[Union[tuple, list]] = None):
        """
        Args:
            spacing: 体素间距
        """
        self.spacing = tuple(spacing) if spacing is not None else (1.0, 1.0, 1.0)
        if not all(np.isfinite(value) and value > 0 for value in self.spacing):
            raise ValueError("Spacing values must be finite and positive")

    def compute_all(
        self,
        moving: Union[torch.Tensor, np.ndarray],
        fixed: Union[torch.Tensor, np.ndarray],
        warped: Union[torch.Tensor, np.ndarray],
        moving_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        fixed_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        warped_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """
        计算所有指标

        Args:
            moving: 浮动图像
            fixed: 参考图像
            warped: 配准后的图像
            moving_mask: 原始浮动图像掩码（不能代替变形后掩码）
            fixed_mask: 参考图像掩码
            warped_mask: 已使用变形场映射到固定图像空间的浮动掩码

        Returns:
            dict: 各指标的值
        """
        # 转换为 numpy
        if isinstance(moving, torch.Tensor):
            moving = moving.detach().cpu().numpy()
        if isinstance(fixed, torch.Tensor):
            fixed = fixed.detach().cpu().numpy()
        if isinstance(warped, torch.Tensor):
            warped = warped.detach().cpu().numpy()
        if isinstance(moving_mask, torch.Tensor):
            moving_mask = moving_mask.detach().cpu().numpy()
        if isinstance(fixed_mask, torch.Tensor):
            fixed_mask = fixed_mask.detach().cpu().numpy()
        if isinstance(warped_mask, torch.Tensor):
            warped_mask = warped_mask.detach().cpu().numpy()

        metrics = {}

        # NCC
        metrics["ncc"] = self.ncc(warped, fixed)

        # MSE
        metrics["mse"] = self.mse(warped, fixed)

        # SSIM (简化版)
        metrics["ssim"] = self.ssim(warped, fixed)

        # Dice (如果有掩码)
        if moving_mask is not None and warped_mask is None:
            raise ValueError("Provide warped_mask; an intensity image is not a warped segmentation")
        if (fixed_mask is None) != (warped_mask is None):
            raise ValueError("fixed_mask and warped_mask must both be provided")
        if fixed_mask is not None:
            metrics["dice"] = self.dice(warped_mask, fixed_mask)

        return metrics

    @staticmethod
    def ncc(
        I: Union[torch.Tensor, np.ndarray],
        J: Union[torch.Tensor, np.ndarray],
        window_size: int = 9,
    ) -> float:
        """
        全局归一化互相关（window_size 参数仅为兼容保留）

        Args:
            I: 图像1
            J: 图像2
            window_size: 兼容参数，不启用局部窗口

        Returns:
            float: NCC 值
        """
        if isinstance(I, torch.Tensor):
            I = I.detach().cpu().numpy()
        if isinstance(J, torch.Tensor):
            J = J.detach().cpu().numpy()

        if np.shape(I) != np.shape(J):
            raise ValueError("Images must have the same shape")

        I = I.flatten().astype(np.float64)
        J = J.flatten().astype(np.float64)

        I_mean = I.mean()
        J_mean = J.mean()

        I_std = I.std()
        J_std = J.std()

        ncc = np.mean((I - I_mean) * (J - J_mean)) / (I_std * J_std + 1e-8)
        return float(ncc)

    @staticmethod
    def mse(
        I: Union[torch.Tensor, np.ndarray],
        J: Union[torch.Tensor, np.ndarray],
    ) -> float:
        """均方误差"""
        if isinstance(I, torch.Tensor):
            I = I.detach().cpu().numpy()
        if isinstance(J, torch.Tensor):
            J = J.detach().cpu().numpy()

        if np.shape(I) != np.shape(J):
            raise ValueError("Images must have the same shape")
        return float(np.mean((np.asarray(I, dtype=np.float64) - np.asarray(J, dtype=np.float64)) ** 2))

    @staticmethod
    def ssim(
        I: Union[torch.Tensor, np.ndarray],
        J: Union[torch.Tensor, np.ndarray],
        window_size: int = 11,
        C1: float = 0.01,
        C2: float = 0.03,
    ) -> float:
        """
        全局结构相似性指数（非滑窗 SSIM）

        Args:
            I: 图像1
            J: 图像2
            window_size: 兼容参数，不启用局部窗口
            C1, C2: 稳定性常数

        Returns:
            float: SSIM 值
        """
        if isinstance(I, torch.Tensor):
            I = I.detach().cpu().numpy()
        if isinstance(J, torch.Tensor):
            J = J.detach().cpu().numpy()

        if np.shape(I) != np.shape(J):
            raise ValueError("Images must have the same shape")

        I = I.astype(np.float64)
        J = J.astype(np.float64)

        # 常数
        K1 = C1
        K2 = C2
        L = I.max() - I.min() if I.max() > I.min() else 1

        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2

        # 计算统计量
        mu_I = I.mean()
        mu_J = J.mean()
        sigma_I = I.var()
        sigma_J = J.var()
        sigma_IJ = np.mean((I - mu_I) * (J - mu_J))

        # SSIM
        numerator = (2 * mu_I * mu_J + C1) * (2 * sigma_IJ + C2)
        denominator = (mu_I**2 + mu_J**2 + C1) * (sigma_I + sigma_J + C2)

        ssim = numerator / denominator
        return float(ssim)

    @staticmethod
    def dice(
        warped_mask: np.ndarray,
        fixed_mask: np.ndarray,
        warped: Optional[np.ndarray] = None,
    ) -> float:
        """
        Dice 系数

        计算变形后掩码与固定掩码的重叠度
        """
        if warped is not None:
            raise ValueError("Pass the warped segmentation and fixed segmentation, not a warped image")
        if isinstance(warped_mask, torch.Tensor):
            warped_mask = warped_mask.detach().cpu().numpy()
        if isinstance(fixed_mask, torch.Tensor):
            fixed_mask = fixed_mask.detach().cpu().numpy()
        if np.shape(warped_mask) != np.shape(fixed_mask):
            raise ValueError("Segmentation masks must have the same shape")
        warped_binary = np.asarray(warped_mask) > 0.5
        fixed_binary = np.asarray(fixed_mask) > 0.5
        denominator = int(warped_binary.sum()) + int(fixed_binary.sum())
        if denominator == 0:
            return 1.0
        return float(2 * np.count_nonzero(warped_binary & fixed_binary) / denominator)

    @staticmethod
    def asd(
        surface1: np.ndarray, surface2: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0)
    ) -> float:
        """
        平均表面距离 (Average Surface Distance)

        Args:
            surface1: 表面1的点坐标
            surface2: 表面2的点坐标
            spacing: 体素间距

        Returns:
            float: ASD 值
        """
        from scipy.spatial import cKDTree

        surface1 = np.asarray(surface1, dtype=np.float64)
        surface2 = np.asarray(surface2, dtype=np.float64)
        spacing = np.asarray(spacing, dtype=np.float64)
        if (surface1.ndim != 2 or surface2.ndim != 2 or not len(surface1) or not len(surface2)
                or surface1.shape[1] != surface2.shape[1] or spacing.shape != (surface1.shape[1],)
                or not np.isfinite(surface1).all() or not np.isfinite(surface2).all()
                or not np.isfinite(spacing).all() or np.any(spacing <= 0)):
            raise ValueError("Surface point arrays and positive spacing must have matching dimensions")
        surface1 = surface1 * spacing
        surface2 = surface2 * spacing
        d1 = cKDTree(surface2).query(surface1)[0]
        d2 = cKDTree(surface1).query(surface2)[0]
        return float((d1.mean() + d2.mean()) / 2)

    @staticmethod
    def hd(
        surface1: np.ndarray, surface2: np.ndarray, percentile: float = 95,
        spacing: Optional[tuple] = None,
    ) -> float:
        """
        Hausdorff 距离

        Args:
            surface1: 表面1的点坐标
            surface2: 表面2的点坐标
            percentile: 百分位数

        Returns:
            float: HD 值
        """
        from scipy.spatial import cKDTree

        surface1 = np.asarray(surface1, dtype=np.float64)
        surface2 = np.asarray(surface2, dtype=np.float64)
        if (surface1.ndim != 2 or surface2.ndim != 2 or not len(surface1) or not len(surface2)
                or surface1.shape[1] != surface2.shape[1] or not np.isfinite(surface1).all()
                or not np.isfinite(surface2).all() or not 0 <= percentile <= 100):
            raise ValueError("Surface point arrays must be finite, nonempty and have matching dimensions")
        if spacing is not None:
            spacing = np.asarray(spacing, dtype=np.float64)
            if spacing.shape != (surface1.shape[1],) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
                raise ValueError("Spacing must contain one positive finite value per coordinate")
            surface1 = surface1 * spacing
            surface2 = surface2 * spacing
        hd1 = np.percentile(cKDTree(surface2).query(surface1)[0], percentile)
        hd2 = np.percentile(cKDTree(surface1).query(surface2)[0], percentile)
        return float(max(hd1, hd2))


def compute_jacobian_determinant(
    deformation_field: Union[torch.Tensor, np.ndarray],
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    spatial_dims: Optional[int] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """
    计算相对物理位移场对应的完整 Jacobian 行列式，返回原网格大小。

    Args:
        deformation_field: [2,H,W]、[3,D,H,W] 或带 batch 维度的相应张量；
            分量为物理坐标 x、y、z 方向的位移，单位与 spacing 一致。
        spacing: 按 (x,y[,z]) 排列的体素间距。2D 的默认 (1,1,1) 视为 (1,1)。
        spatial_dims: 当形状同时匹配 3D 和 batched 2D 时，显式指定 2 或 3。
            此函数假定数组轴与物理坐标轴平行，不处理方向矩阵。

    Returns:
        与输入类型、dtype 和设备相同的标量场；torch 输入保持可微。
    """
    import math

    if not isinstance(deformation_field, (torch.Tensor, np.ndarray)):
        raise TypeError("deformation_field must be a torch tensor or numpy array")
    is_tensor = isinstance(deformation_field, torch.Tensor)
    field = deformation_field if is_tensor else torch.from_numpy(np.ascontiguousarray(deformation_field))
    if field.dtype not in (torch.float32, torch.float64):
        raise ValueError("deformation_field must be float32 or float64")
    candidates = []
    if field.ndim == 3 and field.shape[0] == 2:
        candidates.append((2, False))
    if field.ndim == 4 and field.shape[0] == 3:
        candidates.append((3, False))
    if field.ndim == 4 and field.shape[1] == 2:
        candidates.append((2, True))
    if field.ndim == 5 and field.shape[1] == 3:
        candidates.append((3, True))
    if spatial_dims is not None:
        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3")
        candidates = [item for item in candidates if item[0] == spatial_dims]
    if len(candidates) != 1:
        raise ValueError("invalid or ambiguous field shape; provide spatial_dims for ambiguous shapes")
    dimensions, batched = candidates[0]
    field = field if batched else field.unsqueeze(0)
    if field.shape[0] < 1 or min(field.shape[2:]) < 2:
        raise ValueError("field batch must be non-empty and spatial dimensions >= 2")
    if not torch.isfinite(field).all():
        raise ValueError("deformation_field must contain only finite values")
    steps = tuple(float(value) for value in spacing)
    if dimensions == 2 and steps == (1.0, 1.0, 1.0):
        steps = (1.0, 1.0)
    if len(steps) != dimensions or any(not math.isfinite(value) or value <= 0 for value in steps):
        raise ValueError("spacing must contain one finite positive value per spatial axis in x,y,z order")
    # Tensor axes are [..., y, x] or [..., z, y, x], whereas vectors and
    # spacing use x,y,z order. One-sided differences retain the edge voxels.
    derivatives = [torch.gradient(field, spacing=(steps[i],), dim=(-1 - i,), edge_order=1)[0]
                   for i in range(dimensions)]
    dx, dy = derivatives[:2]
    if dimensions == 2:
        determinant = (1 + dx[:, 0]) * (1 + dy[:, 1]) - dy[:, 0] * dx[:, 1]
    else:
        dz = derivatives[2]
        a, b, c = 1 + dx[:, 0], dy[:, 0], dz[:, 0]
        d, e, f = dx[:, 1], 1 + dy[:, 1], dz[:, 1]
        g, h, i = dx[:, 2], dy[:, 2], 1 + dz[:, 2]
        determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    determinant = determinant if batched else determinant[0]
    return determinant if is_tensor else determinant.numpy()


def count_folding(
    jacobian: Union[torch.Tensor, np.ndarray], threshold: float = 0.0
) -> int:
    """计算折叠体素数量"""
    if isinstance(jacobian, torch.Tensor):
        jacobian = jacobian.cpu().numpy()

    return int(np.sum(jacobian < threshold))
