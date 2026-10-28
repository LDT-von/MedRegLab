"""
Loss Builder - 损失函数构建器
支持多种损失函数的组合
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
import math


class LossBuilder:
    """
    损失函数构建器

    支持的损失类型：
    - mse: 均方误差
    - ncc: 归一化互相关
    - tv: 全变差正则化
    - jacobian: Jacobian 行列式正则化

    NMI and IMPACT are not implemented and are rejected explicitly.
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
        alias = kwargs.pop("loss_types", None)
        if kwargs:
            raise TypeError(f"Unsupported LossBuilder options: {sorted(kwargs)}")
        if losses is not None and alias is not None:
            raise ValueError("Specify either losses or loss_types, not both")
        selected = losses if losses is not None else alias
        self.losses = list(selected) if selected is not None else ["mse", "ncc"]
        if not self.losses or len(set(self.losses)) != len(self.losses):
            raise ValueError("Loss names must be nonempty and unique")
        unsupported = set(self.losses) & {"nmi", "impact"}
        if unsupported:
            raise NotImplementedError(f"Losses not implemented: {sorted(unsupported)}")
        unknown = set(self.losses) - {"mse", "ncc", "tv", "jacobian"}
        if unknown:
            raise ValueError(f"Unknown loss names: {sorted(unknown)}")
        self.weights = dict(weights) if weights is not None else {"mse": 1.0, "ncc": 1.0}
        if set(self.weights) - {"mse", "ncc", "tv", "jacobian"}:
            raise ValueError("Unknown loss weight name")
        for weight in [*self.weights.values(), impact_weight, tv_weight, jacobian_weight]:
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("Loss weights must be finite and nonnegative")
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
        if warped.shape != fixed.shape or warped.device != fixed.device:
            raise ValueError("Warped and fixed images must have the same shape and device")
        if features_warped is not None or features_fixed is not None:
            raise NotImplementedError("IMPACT semantic loss is not implemented")
        if phi is None and ({"tv", "jacobian"} & set(self.losses)):
            raise ValueError("A displacement field is required for TV/Jacobian loss")
        total_loss = warped.new_zeros(())
        losses = {}

        # 基础损失
        for loss_name in self.losses:
            if loss_name == "mse":
                loss_val = F.mse_loss(warped, fixed)
            elif loss_name == "ncc":
                loss_val = self._ncc_loss(warped, fixed)
            else:  # TV and Jacobian are handled once below.
                continue

            weight = self.weights.get(loss_name, 1.0)
            total_loss = total_loss + weight * loss_val
            losses[loss_name] = loss_val.item()

        # 全变差正则化
        tv_weight = self.weights.get("tv", self.tv_weight) if "tv" in self.losses else self.tv_weight
        jac_weight = self.weights.get("jacobian", self.jacobian_weight) if "jacobian" in self.losses else self.jacobian_weight
        if phi is not None and tv_weight > 0:
            tv_loss = self._tv_regularization(phi)
            total_loss = total_loss + tv_weight * tv_loss
            losses["tv"] = tv_loss.item()

        # Jacobian 正则化
        if phi is not None and jac_weight > 0:
            jac_loss = self._jacobian_regularization(phi)
            total_loss = total_loss + jac_weight * jac_loss
            losses["jacobian"] = jac_loss.item()

        losses["total"] = total_loss.item()
        losses["total_tensor"] = total_loss

        return losses

    def _ncc_loss(self, I: torch.Tensor, J: torch.Tensor, window_size: int = 9) -> torch.Tensor:
        """
        每个批次、通道内的全局归一化互相关损失。

        Args:
            I: 图像1
            J: 图像2
            window_size: 旧接口参数；当前计算全局 NCC，不使用局部窗口

        Returns:
            torch.Tensor: NCC 损失 (1 - NCC)
        """
        if I.dim() < 3 or I.shape != J.shape:
            raise ValueError("NCC inputs must have matching [B, C, spatial...] shapes")
        dims = tuple(range(2, I.dim()))
        centered_i = I - I.mean(dim=dims, keepdim=True)
        centered_j = J - J.mean(dim=dims, keepdim=True)
        covariance = (centered_i * centered_j).mean(dim=dims)
        variance_i = centered_i.square().mean(dim=dims)
        variance_j = centered_j.square().mean(dim=dims)
        ncc = covariance / torch.sqrt(variance_i * variance_j + 1e-12)
        return 1 - ncc.mean()

    def _nmi_loss(self, I: torch.Tensor, J: torch.Tensor, bins: int = 32) -> torch.Tensor:
        """
        未实现的归一化互信息损失；显式抛错以防误用。

        Args:
            I: 图像1
            J: 图像2
            bins: 直方图 bin 数量

        Returns:
            torch.Tensor: NMI 损失 (1 - NMI)
        """
        raise NotImplementedError("Normalized mutual information loss is not implemented")

    def _impact_loss(
        self, features_warped: torch.Tensor, features_fixed: torch.Tensor
    ) -> torch.Tensor:
        """
        未实现的 IMPACT 语义损失；显式抛错以防误用。

        Args:
            features_warped: 变形图像的语义特征
            features_fixed: 参考图像的语义特征

        Returns:
            torch.Tensor: IMPACT 损失
        """
        raise NotImplementedError("IMPACT semantic loss is not implemented")

    def _tv_regularization(self, phi: torch.Tensor) -> torch.Tensor:
        """
        全变差正则化

        Args:
            phi: 变形场 [B, D, H, W] 或 [B, D, D, H, W]

        Returns:
            torch.Tensor: TV 正则化值
        """
        self._validate_displacement(phi)
        return sum(torch.diff(phi, dim=axis).abs().mean() for axis in range(2, phi.dim()))

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
        self._validate_displacement(phi)
        # Fields are voxel displacements in channel order x=W, y=H, z=D.
        # All forward differences share the same lower-corner grid.
        if phi.dim() == 4:
            base = phi[:, :, :-1, :-1]
            dx = phi[:, :, :-1, 1:] - base
            dy = phi[:, :, 1:, :-1] - base
            det = (1 + dx[:, 0]) * (1 + dy[:, 1]) - dy[:, 0] * dx[:, 1]
        else:
            base = phi[:, :, :-1, :-1, :-1]
            dx = phi[:, :, :-1, :-1, 1:] - base
            dy = phi[:, :, :-1, 1:, :-1] - base
            dz = phi[:, :, 1:, :-1, :-1] - base
            a, b, c = 1 + dx[:, 0], dy[:, 0], dz[:, 0]
            d, e, f = dx[:, 1], 1 + dy[:, 1], dz[:, 1]
            g, h, i = dx[:, 2], dy[:, 2], 1 + dz[:, 2]
            det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        return F.relu(eps - det).mean()

    @staticmethod
    def _validate_displacement(phi: torch.Tensor) -> None:
        if phi.dim() not in (4, 5) or phi.shape[1] != phi.dim() - 2:
            raise ValueError("Expected voxel displacement [B, 2, H, W] or [B, 3, D, H, W]")
        if not phi.is_floating_point() or any(size < 2 for size in phi.shape[2:]):
            raise ValueError("Displacement must be floating point with each spatial dimension >= 2")
        if not torch.isfinite(phi).all():
            raise ValueError("Displacement contains NaN or Inf")
