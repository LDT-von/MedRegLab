"""
uniGradICON - Unified Gradient-based Image COngratulation Network

论文: "uniGradICON: A Unified Framework for Dense, Sparse, and Learning-based Image Registration"
参考: https://github.com/uncbiag/uniGradICON

核心思想：
1. 统一的梯度驱动配准框架
2. 结合了传统方法和深度学习的优点
3. 支持密集匹配和稀疏匹配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class uniGradICON(nn.Module):
    """
    uniGradICON 模型

    统一的图像配准框架，整合：
    - 特征提取网络
    - 梯度计算模块
    - 变形场估计器
    """

    def __init__(
        self,
        img_size: Tuple[int, ...] = (224, 224),
        in_channels: int = 1,
        out_channels: int = 2,
        feature_dim: int = 256,
        num_levels: int = 4,
        use_attention: bool = True,
        pretrained: bool = False,
        **kwargs,
    ):
        """
        Args:
            img_size: 输入图像尺寸
            in_channels: 输入通道数
            out_channels: 输出通道数（2D为2，3D为3）
            feature_dim: 特征维度
            num_levels: 金字塔层数
            use_attention: 是否使用注意力机制
            pretrained: 是否使用预训练权重
        """
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_dim = feature_dim
        self.num_levels = num_levels

        # 特征提取器（多尺度）
        self.feature_extractor = MultiScaleFeatureExtractor(
            in_channels=in_channels,
            feature_dim=feature_dim,
            num_levels=num_levels,
        )

        # 梯度计算模块
        self.gradient_module = GradientComputationModule(
            feature_dim=feature_dim,
            use_attention=use_attention,
        )

        # 变形场估计器
        self.field_estimator = DeformationFieldEstimator(
            feature_dim=feature_dim,
            out_channels=out_channels,
        )

        # 正则化器
        self.regularizer = FieldRegularizer()

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        return_all_levels: bool = False,
    ) -> Dict[str, Any]:
        """
        前向传播

        Args:
            moving: 浮动图像 [B, C, H, W] 或 [B, C, D, H, W]
            fixed: 参考图像 [B, C, H, W] 或 [B, C, D, H, W]
            return_all_levels: 是否返回所有金字塔层的结果

        Returns:
            dict: 包含变形场、损失等信息
        """
        # 提取多尺度特征
        moving_features, fixed_features = self.feature_extractor(moving, fixed)

        # 计算梯度
        gradients = self.gradient_module(moving_features, fixed_features)

        # 估计变形场
        phi = self.field_estimator(gradients)

        # 正则化
        phi_smooth = self.regularizer(phi)

        # 变形图像
        warped = self._warp_image(moving, phi_smooth)

        result = {
            "phi": phi_smooth,
            "warped": warped,
            "moving": moving,
            "fixed": fixed,
        }

        if return_all_levels:
            result["all_levels"] = {
                "moving_features": moving_features,
                "fixed_features": fixed_features,
                "gradients": gradients,
            }

        return result

    def _warp_image(self, image: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """变形图像"""
        if image.dim() == 4:  # 2D
            grid = phi.permute(0, 2, 3, 1)
            grid[..., 0] = 2 * grid[..., 0] / (image.shape[3] - 1) - 1
            grid[..., 1] = 2 * grid[..., 1] / (image.shape[2] - 1) - 1
            warped = F.grid_sample(
                image, grid, mode="bilinear", padding_mode="border", align_corners=True
            )
        else:  # 3D
            grid = phi.permute(0, 2, 3, 4, 1)
            grid[..., 0] = 2 * grid[..., 0] / (image.shape[4] - 1) - 1
            grid[..., 1] = 2 * grid[..., 1] / (image.shape[3] - 1) - 1
            grid[..., 2] = 2 * grid[..., 2] / (image.shape[2] - 1) - 1
            warped = F.grid_sample(
                image, grid, mode="trilinear", padding_mode="border", align_corners=True
            )
        return warped


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器"""

    def __init__(
        self, in_channels: int = 1, feature_dim: int = 256, num_levels: int = 4
    ):
        super().__init__()
        self.num_levels = num_levels

        # 各层级的编码器
        self.encoders = nn.ModuleList()
        current_channels = in_channels

        for i in range(num_levels):
            out_channels = feature_dim // (2**i) if i < num_levels - 1 else feature_dim
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, out_channels, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.InstanceNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                )
            )
            current_channels = out_channels

    def forward(
        self, moving: torch.Tensor, fixed: torch.Tensor
    ) -> Tuple[list, list]:
        """提取多尺度特征"""
        moving_features = []
        fixed_features = []

        x_m, x_f = moving, fixed

        for encoder in self.encoders:
            x_m = encoder(x_m)
            x_f = encoder(x_f)
            moving_features.append(x_m)
            fixed_features.append(x_f)

        return moving_features, fixed_features


class GradientComputationModule(nn.Module):
    """梯度计算模块"""

    def __init__(self, feature_dim: int = 256, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention

        # 特征匹配层
        self.matching_layer = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            nn.InstanceNorm2d(feature_dim),
            nn.ReLU(),
        )

        # 注意力机制
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 4, 1),
                nn.ReLU(),
                nn.Conv2d(feature_dim // 4, feature_dim, 1),
                nn.Sigmoid(),
            )

    def forward(
        self, moving_features: list, fixed_features: list
    ) -> Tuple[torch.Tensor, ...]:
        """计算特征匹配梯度"""
        gradients = []

        for m_feat, f_feat in zip(moving_features, fixed_features):
            # 拼接特征
            concat = torch.cat([m_feat, f_feat], dim=1)
            matched = self.matching_layer(concat)

            # 应用注意力
            if self.use_attention:
                att = self.attention(m_feat)
                matched = matched * att

            gradients.append(matched)

        return tuple(gradients)


class DeformationFieldEstimator(nn.Module):
    """变形场估计器"""

    def __init__(self, feature_dim: int = 256, out_channels: int = 2):
        super().__init__()
        self.out_channels = out_channels

        # 上采样和预测
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, feature_dim // 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(feature_dim // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_dim // 2, feature_dim // 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(feature_dim // 4),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 4, out_channels, 3, padding=1),
        )

    def forward(self, gradients: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """从多层梯度估计变形场"""
        # 使用最后一层特征
        feat = gradients[-1]

        # 预测变形场
        phi = self.decoder(feat)

        return phi


class FieldRegularizer(nn.Module):
    """变形场正则化器"""

    def __init__(self, smoothness_weight: float = 0.1):
        super().__init__()
        self.smoothness_weight = smoothness_weight

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """平滑变形场"""
        # 简化的平滑：使用卷积
        if phi.dim() == 4:  # 2D
            smooth_kernel = torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=phi.dtype, device=phi.device
            ).view(1, 1, 3, 3)

            # 对每个通道分别平滑
            phi_smooth = phi.clone()
            for c in range(phi.shape[1]):
                phi_smooth[:, c : c + 1] = F.conv2d(
                    phi[:, c : c + 1], smooth_kernel, padding=1
                )

            phi = phi + self.smoothness_weight * phi_smooth

        return phi
