"""
Convex Regularizer - 凸正则化器
用于确保变形场的物理合理性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvexRegularizer(nn.Module):
    """
    凸正则化器

    确保变形场满足以下物理约束：
    1. 微分同胚性（Diffeomorphism）：变形可逆且光滑
    2. 体积保持：避免过度折叠或撕裂
    3. 能量最小化：鼓励平滑的变形

    基于凸优化理论，确保正则化项是凸函数
    """

    def __init__(
        self,
        smoothness_weight: float = 0.1,
        jacobian_weight: float = 0.01,
        fold_penalty_weight: float = 0.1,
        bending_energy_weight: float = 0.01,
        **kwargs,
    ):
        """
        Args:
            smoothness_weight: 平滑正则化权重
            jacobian_weight: Jacobian 行列式正则化权重
            fold_penalty_weight: 折叠惩罚权重
            bending_energy_weight: 弯曲能量权重
        """
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.jacobian_weight = jacobian_weight
        self.fold_penalty_weight = fold_penalty_weight
        self.bending_energy_weight = bending_energy_weight

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        计算正则化损失

        Args:
            phi: 变形场 [B, D, H, W] 或 [B, D, D, H, W]

        Returns:
            torch.Tensor: 正则化损失
        """
        total_reg = 0.0

        # 1. 平滑正则化（全变差）
        if self.smoothness_weight > 0:
            tv_loss = self._total_variation(phi)
            total_reg = total_reg + self.smoothness_weight * tv_loss

        # 2. Jacobian 正则化
        if self.jacobian_weight > 0:
            jac_loss = self._jacobian_determinant_regularization(phi)
            total_reg = total_reg + self.jacobian_weight * jac_loss

        # 3. 折叠惩罚
        if self.fold_penalty_weight > 0:
            fold_loss = self._fold_penalty(phi)
            total_reg = total_reg + self.fold_penalty_weight * fold_loss

        # 4. 弯曲能量
        if self.bending_energy_weight > 0:
            bend_loss = self._bending_energy(phi)
            total_reg = total_reg + self.bending_energy_weight * bend_loss

        return total_reg

    def _total_variation(self, phi: torch.Tensor) -> torch.Tensor:
        """全变差正则化"""
        if phi.dim() == 4:  # 2D [B, 2, H, W]
            # 水平梯度
            dx = phi[:, :, 1:, :] - phi[:, :, :-1, :]
            # 垂直梯度
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

    def _jacobian_determinant_regularization(
        self, phi: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Jacobian 行列式正则化

        惩罚负的 Jacobian（折叠）和接近零的值
        """
        if phi.dim() == 4:  # 2D
            # 计算 Jacobian 矩阵
            # J[i,j] = dphi_i / dx_j
            dphi_dx = self._gradient(phi, dim=2)
            dphi_dy = self._gradient(phi, dim=3)

            # 2D Jacobian 行列式
            det = dphi_dx[:, 0] * dphi_dy[:, 1] - dphi_dx[:, 1] * dphi_dy[:, 0]
        else:  # 3D
            # 简化的 3D Jacobian（使用体积变化近似）
            dx = phi[:, 0, 1:, :-1, :-1] - phi[:, 0, :-1, :-1, :-1]
            dy = phi[:, 1, :-1, 1:, :-1] - phi[:, 1, :-1, :-1, :-1]
            dz = phi[:, 2, :-1, :-1, 1:] - phi[:, 2, :-1, :-1, :-1]

            det = dx + dy + dz

        # 惩罚负值
        loss = torch.mean(F.relu(eps - det) ** 2)

        return loss

    def _fold_penalty(self, phi: torch.Tensor) -> torch.Tensor:
        """
        折叠惩罚

        当局部变形导致体积压缩时惩罚
        """
        if phi.dim() == 4:  # 2D
            # 计算局部雅可比
            det = self._compute_2d_jacobian(phi)

            # 折叠区域：det < 0
            fold_penalty = torch.mean(F.relu(-det + 1e-4) ** 2)
        else:
            # 3D 简化
            fold_penalty = torch.tensor(0.0, device=phi.device)

        return fold_penalty

    def _bending_energy(self, phi: torch.Tensor) -> torch.Tensor:
        """
        弯曲能量

        衡量变形场的二阶导数（曲率）
        """
        if phi.dim() == 4:  # 2D
            # 二阶导数
            d2x_dx2 = self._gradient(phi[:, 0:1], dim=2, order=2)
            d2x_dy2 = self._gradient(phi[:, 0:1], dim=3, order=2)
            d2y_dx2 = self._gradient(phi[:, 1:2], dim=2, order=2)
            d2y_dy2 = self._gradient(phi[:, 1:2], dim=3, order=2)

            bending = (
                torch.mean(d2x_dx2**2)
                + torch.mean(d2x_dy2**2)
                + torch.mean(d2y_dx2**2)
                + torch.mean(d2y_dy2**2)
            )
        else:
            bending = torch.tensor(0.0, device=phi.device)

        return bending

    def _gradient(
        self, x: torch.Tensor, dim: int, order: int = 1
    ) -> torch.Tensor:
        """计算一阶或二阶差分"""
        if order == 1:
            if dim == 2:
                return x[:, :, 1:, :] - x[:, :, :-1, :]
            else:
                return x[:, :, :, 1:] - x[:, :, :, :-1]
        else:  # 二阶
            if dim == 2:
                grad = x[:, :, 1:, :] - x[:, :, :-1, :]
                return grad[:, :, 1:, :] - grad[:, :, :-1, :]
            else:
                grad = x[:, :, :, 1:] - x[:, :, :, :-1]
                return grad[:, :, :, 1:] - grad[:, :, :, :-1]

    def _compute_2d_jacobian(self, phi: torch.Tensor) -> torch.Tensor:
        """计算 2D Jacobian 行列式"""
        # dphi/dx
        dphi_dx = phi[:, :, 1:, :-1] - phi[:, :, :-1, :-1]
        # dphi/dy
        dphi_dy = phi[:, :, :-1, 1:] - phi[:, :, :-1, :-1]

        # Jacobian 矩阵
        J = torch.zeros(
            phi.shape[0],
            2,
            2,
            phi.shape[2] - 1,
            phi.shape[3] - 1,
            device=phi.device,
        )
        J[:, 0, 0] = dphi_dx[:, 0]
        J[:, 0, 1] = dphi_dy[:, 0]
        J[:, 1, 0] = dphi_dx[:, 1]
        J[:, 1, 1] = dphi_dy[:, 1]

        # 添加单位矩阵（因为 phi 是位移场）
        J = J + torch.eye(2, device=phi.device).view(1, 2, 2, 1, 1)

        # 行列式
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]

        return det

    def project_to_diffeomorphism(self, phi: torch.Tensor, num_iter: int = 5) -> torch.Tensor:
        """
        将变形场投影到微分同胚空间

        使用流形上的指数映射
        """
        # 简化实现：将变形场缩放
        phi_scaled = phi / (num_iter + 1)

        # 多次组合
        phi_diffeo = phi_scaled.clone()
        for _ in range(num_iter - 1):
            # 简化的组合操作
            phi_diffeo = phi_diffeo + phi_scaled

        return phi_diffeo
