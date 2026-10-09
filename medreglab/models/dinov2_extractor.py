"""
DINOv2 Feature Extractor - Learn2Reg 2023 冠军方案核心

使用 Meta 的 DINOv2 视觉 transformer 提取自监督特征
这些特征对跨模态配准特别有效
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import numpy as np


class DINOv2Extractor(nn.Module):
    """
    DINOv2 特征提取器

    DINOv2 是 Meta 提出的自监督视觉 transformer，
    其特征对于医学图像的跨模态配准非常有效。

    优势：
    1. 语义丰富的特征表示
    2. 对图像变换的不变性
    3. 跨模态迁移能力
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitl14",
        feature_dim: int = 1024,
        use_frozen: bool = True,
        patch_size: int = 14,
        upsample_mode: str = "bilinear",
        **kwargs,
    ):
        """
        Args:
            model_name: DINOv2 模型名称
                - "dinov2_vitb14": Base 模型
                - "dinov2_vitl14": Large 模型
                - "dinov2_vitg14": Giant 模型
            feature_dim: 特征维度
            use_frozen: 是否冻结 DINOv2 权重
            patch_size: patch 大小
            upsample_mode: 上采样模式
        """
        super().__init__()
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.upsample_mode = upsample_mode

        # 延迟加载模型
        self._model = None
        self._use_pretrained = True

        if use_frozen:
            self._freeze()

    @property
    def model(self):
        """懒加载模型"""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """加载 DINOv2 模型"""
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for DINOv2. Install with: pip install timm"
            )

        # 使用 timm 加载 DINOv2
        self._model = timm.create_model(
            self.model_name,
            pretrained=self._use_pretrained,
            num_classes=0,  # 移除分类头
        )
        self._model.eval()

    def _freeze(self):
        """冻结模型参数"""
        self._freeze_params = True

    def forward(
        self,
        x: torch.Tensor,
        return_spatial_features: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        提取特征

        Args:
            x: 输入图像 [B, C, H, W] 或 [B, 1, H, W]
            return_spatial_features: 是否返回空间特征图

        Returns:
            torch.Tensor 或 (global_feature, spatial_feature)
        """
        # 确保输入是 3 通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 提取特征
        with torch.no_grad():
            features = self.model(x)

        if return_spatial_features:
            # 将特征图上采样到原始分辨率
            spatial_features = self._upsample_features(features, x.shape[2:])
            return features, spatial_features

        return features

    def _upsample_features(
        self, features: torch.Tensor, target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """将 patch 级别的特征上采样到像素级别"""
        B, N, D = features.shape  # N = H*W / patch_size^2

        # 估算 spatial 维度
        # 对于 ViT-L14, patch_size=14, 假设输入是 224x224
        # 则 N = 16*16 = 256
        h = w = int(np.sqrt(N))

        # 重塑为 2D 特征图
        features_2d = features.reshape(B, h, w, D).permute(0, 3, 1, 2)  # [B, D, h, w]

        # 上采样到目标分辨率
        spatial_features = F.interpolate(
            features_2d,
            size=target_size,
            mode=self.upsample_mode,
            align_corners=True if self.upsample_mode == "bilinear" else None,
        )

        return spatial_features

    def extract_multiscale(
        self, x: torch.Tensor, scales: Optional[list] = None
    ) -> dict:
        """
        提取多尺度特征

        Args:
            x: 输入图像
            scales: 缩放因子列表

        Returns:
            dict: 各尺度的特征
        """
        if scales is None:
            scales = [0.5, 1.0, 2.0]

        multiscale_features = {}

        for scale in scales:
            if scale != 1.0:
                scaled_h = int(x.shape[2] * scale)
                scaled_w = int(x.shape[3] * scale)
                x_scaled = F.interpolate(
                    x, size=(scaled_h, scaled_w), mode="bilinear", align_corners=True
                )
            else:
                x_scaled = x

            features = self.forward(x_scaled, return_spatial_features=False)
            multiscale_features[scale] = features

        return multiscale_features


class FeatureProjection(nn.Module):
    """
    特征投影头

    将 DINOv2 特征投影到统一的特征空间
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class FeatureMatcher(nn.Module):
    """
    特征匹配器

    计算两组特征之间的相似度并生成匹配代价
    """

    def __init__(self, feature_dim: int = 1024):
        super().__init__()
        self.feature_dim = feature_dim

        self.matcher = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算匹配代价

        Args:
            features_a: 特征 A [B, N, D]
            features_b: 特征 B [B, M, D]

        Returns:
            torch.Tensor: 匹配代价矩阵 [B, N, M]
        """
        B, N, D = features_a.shape
        M = features_b.shape[1]

        # 扩展维度
        a_expanded = features_a.unsqueeze(2).expand(B, N, M, D)
        b_expanded = features_b.unsqueeze(1).expand(B, N, M, D)

        # 拼接
        concat = torch.cat([a_expanded, b_expanded], dim=-1)

        # 计算匹配代价
        cost = self.matcher(concat).squeeze(-1)

        return cost
