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
    - TRE: 目标配准误差
    - NCC: 归一化互相关
    - SSIM: 结构相似性指数
    """

    def __init__(self, spacing: Optional[Union[tuple, list]] = None):
        """
        Args:
            spacing: 体素间距
        """
        self.spacing = spacing or (1.0, 1.0, 1.0)

    def compute_all(
        self,
        moving: Union[torch.Tensor, np.ndarray],
        fixed: Union[torch.Tensor, np.ndarray],
        warped: Union[torch.Tensor, np.ndarray],
        moving_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        fixed_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """
        计算所有指标

        Args:
            moving: 浮动图像
            fixed: 参考图像
            warped: 配准后的图像
            moving_mask: 浮动图像掩码
            fixed_mask: 参考图像掩码

        Returns:
            dict: 各指标的值
        """
        # 转换为 numpy
        if isinstance(moving, torch.Tensor):
            moving = moving.cpu().numpy()
        if isinstance(fixed, torch.Tensor):
            fixed = fixed.cpu().numpy()
        if isinstance(warped, torch.Tensor):
            warped = warped.cpu().numpy()
        if isinstance(moving_mask, torch.Tensor):
            moving_mask = moving_mask.cpu().numpy()
        if isinstance(fixed_mask, torch.Tensor):
            fixed_mask = fixed_mask.cpu().numpy()

        metrics = {}

        # NCC
        metrics["ncc"] = self.ncc(warped, fixed)

        # MSE
        metrics["mse"] = self.mse(warped, fixed)

        # SSIM (简化版)
        metrics["ssim"] = self.ssim(warped, fixed)

        # Dice (如果有掩码)
        if moving_mask is not None and fixed_mask is not None:
            metrics["dice"] = self.dice(moving_mask, fixed_mask, warped)

        return metrics

    @staticmethod
    def ncc(
        I: Union[torch.Tensor, np.ndarray],
        J: Union[torch.Tensor, np.ndarray],
        window_size: int = 9,
    ) -> float:
        """
        归一化互相关

        Args:
            I: 图像1
            J: 图像2
            window_size: 局部窗口大小

        Returns:
            float: NCC 值
        """
        if isinstance(I, torch.Tensor):
            I = I.cpu().numpy()
        if isinstance(J, torch.Tensor):
            J = J.cpu().numpy()

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
            I = I.cpu().numpy()
        if isinstance(J, torch.Tensor):
            J = J.cpu().numpy()

        return float(np.mean((I - J) ** 2))

    @staticmethod
    def ssim(
        I: Union[torch.Tensor, np.ndarray],
        J: Union[torch.Tensor, np.ndarray],
        window_size: int = 11,
        C1: float = 0.01,
        C2: float = 0.03,
    ) -> float:
        """
        结构相似性指数

        Args:
            I: 图像1
            J: 图像2
            window_size: 窗口大小
            C1, C2: 稳定性常数

        Returns:
            float: SSIM 值
        """
        if isinstance(I, torch.Tensor):
            I = I.cpu().numpy()
        if isinstance(J, torch.Tensor):
            J = J.cpu().numpy()

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
        moving_mask: np.ndarray,
        fixed_mask: np.ndarray,
        warped: np.ndarray,
    ) -> float:
        """
        Dice 系数

        计算变形后掩码与固定掩码的重叠度
        """
        # 简单实现：使用固定掩码内的相似度
        warped_binary = (warped > warped.mean()).astype(np.float32)
        fixed_binary = (fixed_mask > fixed_mask.mean()).astype(np.float32)

        intersection = np.sum(warped_binary * fixed_binary)
        dice = 2 * intersection / (np.sum(warped_binary) + np.sum(fixed_binary) + 1e-8)

        return float(dice)

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
        from scipy.spatial.distance import cdist

        distances = cdist(surface1, surface2)
        asd = (np.mean(np.min(distances, axis=1)) + np.mean(np.min(distances, axis=0))) / 2

        return float(asd)

    @staticmethod
    def hd(
        surface1: np.ndarray, surface2: np.ndarray, percentile: float = 95
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
        from scipy.spatial.distance import cdist

        distances = cdist(surface1, surface2)
        hd1 = np.percentile(np.min(distances, axis=1), percentile)
        hd2 = np.percentile(np.min(distances, axis=0), percentile)

        return float(max(hd1, hd2))


def compute_jacobian_determinant(
    deformation_field: Union[torch.Tensor, np.ndarray],
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> Union[torch.Tensor, np.ndarray]:
    """
    计算变形场的 Jacobian 行列式

    Args:
        deformation_field: 变形场
        spacing: 体素间距

    Returns:
        Jacobian 行列式
    """
    if isinstance(deformation_field, torch.Tensor):
        is_tensor = True
        field = deformation_field.detach().cpu().numpy()
    else:
        is_tensor = False
        field = deformation_field

    if field.ndim == 4:  # 2D [D, H, W] or [B, D, H, W]
        # 使用 SimpleITK 计算
        field_sitk = sitk.GetImageFromArray(field.transpose(1, 2, 0))
        field_sitk.SetSpacing(spacing[1:])

        jacobian = sitk.DisplacementFieldJacobianDeterminant(field_sitk)
        jacobian_arr = sitk.GetArrayFromImage(jacobian)
    else:  # 3D
        field_sitk = sitk.GetImageFromArray(field.transpose(1, 2, 3, 0))
        field_sitk.SetSpacing(spacing)

        jacobian = sitk.DisplacementFieldJacobianDeterminant(field_sitk)
        jacobian_arr = sitk.GetArrayFromImage(jacobian)

    if is_tensor:
        return torch.from_numpy(jacobian_arr)
    return jacobian_arr


def count_folding(
    jacobian: Union[torch.Tensor, np.ndarray], threshold: float = 0.0
) -> int:
    """计算折叠体素数量"""
    if isinstance(jacobian, torch.Tensor):
        jacobian = jacobian.cpu().numpy()

    return int(np.sum(jacobian < threshold))
