"""
Loss Builder - 损失函数构建器
支持多种损失函数的组合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable


class LossBuilder:
    """
    损失函数构建器

    支持的损失类型：
    - mse: 均方误差
    - ncc: 归一化互相关
    - nmi: 归一化互信息
    - impact: IMPACT 语义损失
    - tv: 全变差正则化
    - jacobian: Jacobian 行列式正则化
    """

    def __init__(
        self,
        losses: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        impact_weight: float = 0.5,
        tv_weight: float = 0.1,
        jacobian_weight: float = 0.01,
        **kwargs,
    ):
        """
        Args:
            losses: 损失函数列表
            weights: 各损失权重字典
            impact_weight: IMPACT 语义损失权重
            tv_weight: 全变差正则化权重
            jacobian_weight: Jacobian 正则化权重
        """
        self.losses = losses or ["mse", "ncc"]
        self.weights = weights or {"mse": 1.0, "ncc": 1.0}
        self.impact_weight = impact_weight
        self.tv_weight = tv_weight
        self.jacobian_weight = jacobian_weight

    def __call__(
        self,
        warped: torch.Tensor,
        fixed: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        features_warped: Optional[torch.Tensor] = None,
        features_fixed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失

        Args:
            warped: 变形后的图像
            fixed: 参考图像
            phi: 变形场
            features_warped: 变形图像的语义特征
            features_fixed: 参考图像的语义特征

        Returns:
            dict: 各损失及总损失
        """
        total_loss = 0.0
        losses = {}

        # 基础损失
        for loss_name in self.losses:
            if loss_name == "mse":
                loss_val = F.mse_loss(warped, fixed)
            elif loss_name == "ncc":
                loss_val = self._ncc_loss(warped, fixed)
            elif loss_name == "nmi":
                loss_val = self._nmi_loss(warped, fixed)
            else:
                continue

            weight = self.weights.get(loss_name, 1.0)
            total_loss = total_loss + weight * loss_val
            losses[loss_name] = loss_val.item()

        # IMPACT 语义损失
        if features_warped is not None and features_fixed is not None:
            impact_loss = self._impact_loss(features_warped, features_fixed)
            total_loss = total_loss + self.impact_weight * impact_loss
            losses["impact"] = impact_loss.item()

        # 全变差正则化
        if phi is not None and self.tv_weight > 0:
            tv_loss = self._tv_regularization(phi)
            total_loss = total_loss + self.tv_weight * tv_loss
            losses["tv"] = tv_loss.item()

        # Jacobian 正则化
        if phi is not None and self.jacobian_weight > 0:
            jac_loss = self._jacobian_regularization(phi)
            total_loss = total_loss + self.jacobian_weight * jac_loss
            losses["jacobian"] = jac_loss.item()

        losses["total"] = total_loss.item()
        losses["total_tensor"] = total_loss

        return losses

    def _ncc_loss(self, I: torch.Tensor, J: torch.Tensor, window_size: int = 9) -> torch.Tensor:
        """
        局部归一化互相关损失

        Args:
            I: 图像1
            J: 图像2
            window_size: 局部窗口大小

        Returns:
            torch.Tensor: NCC 损失 (1 - NCC)
        """
        # 计算局部均值和标准差
        device = I.device
        channels = I.shape[1] if I.dim() == 5 else 1

        # 简化版 NCC
        I_mean = I.mean()
        J_mean = J.mean()
        I_std = I.std()
        J_std = J.std()

        ncc = ((I - I_mean) * (J - J_mean)).mean() / (I_std * J_std + 1e-8)
        return 1 - ncc

    def _nmi_loss(self, I: torch.Tensor, J: torch.Tensor, bins: int = 32) -> torch.Tensor:
        """
        归一化互信息损失

        Args:
            I: 图像1
            J: 图像2
            bins: 直方图 bin 数量

        Returns:
            torch.Tensor: NMI 损失 (1 - NMI)
        """
        # 简化版：使用互相关作为代理
        I_flat = I.flatten()
        J_flat = J.flatten()

        # 计算互信息近似
        mi = (I_flat * J_flat).mean() - I_flat.mean() * J_flat.mean()
        nmi = mi / (I_flat.std() * J_flat.std() + 1e-8)

        return 1 - nmi

    def _impact_loss(
        self, features_warped: torch.Tensor, features_fixed: torch.Tensor
    ) -> torch.Tensor:
        """
        IMPACT 语义损失

        基于 DINOv2 特征的语义感知配准损失

        Args:
            features_warped: 变形图像的语义特征
            features_fixed: 参考图像的语义特征

        Returns:
            torch.Tensor: IMPACT 损失
        """
        # 余弦相似度损失
        cos_sim = F.cosine_similarity(features_warped, features_fixed, dim=-1)
        return 1 - cos_sim.mean()

    def _tv_regularization(self, phi: torch.Tensor) -> torch.Tensor:
        """
        全变差正则化

        Args:
            phi: 变形场 [B, D, H, W] 或 [B, D, D, H, W]

        Returns:
            torch.Tensor: TV 正则化值
        """
        if phi.dim() == 4:  # 2D [B, 2, H, W]
            dx = phi[:, :, 1:, :] - phi[:, :, :-1, :]
            dy = phi[:, :, :, 1:] - phi[:, :, :, :-1]
            tv = torch.mean(torch.sqrt(dx * dx + 1e-8)) + torch.mean(
                torch.sqrt(dy * dy + 1e-8)
            )
        else:  # 3D [B, 3, D, H, W]
            dx = phi[:, :, 1:, :, :] - phi[:, :, :-1, :, :]
            dy = phi[:, :, :, 1:, :] - phi[:, :, :, :-1, :]
            dz = phi[:, :, :, :, 1:] - phi[:, :, :, :, :-1]
            tv = (
                torch.mean(torch.sqrt(dx * dx + 1e-8))
                + torch.mean(torch.sqrt(dy * dy + 1e-8))
                + torch.mean(torch.sqrt(dz * dz + 1e-8))
            )
        return tv

    def _jacobian_regularization(self, phi: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Jacobian 行列式正则化

        惩罚负的 Jacobian 行列式（折叠）

        Args:
            phi: 变形场
            eps: 数值稳定性参数

        Returns:
            torch.Tensor: Jacobian 正则化值
        """
        # 计算变形场的 Jacobian
        if phi.dim() == 4:  # 2D
            # 简化的 2D Jacobian 计算
            dx_dx = self._gradient(phi[:, 0:1, :, :], dim=2)
            dx_dy = self._gradient(phi[:, 0:1, :, :], dim=3)
            dy_dx = self._gradient(phi[:, 1:2, :, :], dim=2)
            dy_dy = self._gradient(phi[:, 1:2, :, :], dim=3)
            det = dx_dx * dy_dy - dx_dy * dy_dx
        else:  # 3D
            # 简化的 3D Jacobian 计算
            phi = phi.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]
            dphi_dx = self._gradient_3d(phi, dim=0)
            dphi_dy = self._gradient_3d(phi, dim=1)
            dphi_dz = self._gradient_3d(phi, dim=2)
            jac = torch.stack([dphi_dx, dphi_dy, dphi_dz], dim=-1)
            det = torch.det(jac + torch.eye(3, device=phi.device) * (1 - eps))

        # 惩罚负值和接近零的值
        det_positive = torch.clamp(det, min=eps)
        reg = torch.mean(1.0 / det_positive) + torch.mean(F.relu(-det + eps))

        return reg

    def _gradient(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """计算一阶差分"""
        if dim == 2:
            return x[:, :, 1:, :] - x[:, :, :-1, :]
        else:
            return x[:, :, :, 1:] - x[:, :, :, :-1]

    def _gradient_3d(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """计算 3D 张量的一阶差分"""
        if dim == 0:
            return x[1:, :, :, :] - x[:-1, :, :, :]
        elif dim == 1:
            return x[:, 1:, :, :] - x[:, :-1, :, :]
        else:
            return x[:, :, 1:, :] - x[:, :, :-1, :]
