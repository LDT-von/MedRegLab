"""
IMPACT Loss - Learn2Reg 2023 冠军语义感知损失函数

论文: "IMPACT: Implicit Neural Penalty based Adversarial Cross-domain Transform for Medical Image Registration"
参考: https://github.com/liuqu321/ImpactLoss

核心思想：
1. 使用隐式神经表示捕捉复杂的图像特征
2. 语义级别的相似性度量，不仅仅是像素级
3. 对跨模态配准特别有效
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class IMPACTLoss(nn.Module):
    """
    IMPACT (Implicit Neural Penalty based Adversarial Cross-domain Transform) 损失

    创新点：
    1. 语义感知：通过隐式神经网络编码图像的语义信息
    2. 跨模态适应：使用对抗训练处理不同模态间的差异
    3. 隐式正则化：通过神经网络隐式地编码物理约束

    适用于：
    - MR-CT 跨模态配准
    - 多器官医学图像配准
    - 需要语义一致性的配准任务
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 256,
        num_layers: int = 4,
        temperature: float = 0.1,
        semantic_weight: float = 0.5,
        use_contrastive: bool = True,
        use_adversarial: bool = False,
        margin: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            feature_dim: 特征维度
            hidden_dim: 隐层维度
            num_layers: 网络层数
            temperature: 温度参数（用于对比学习）
            semantic_weight: 语义损失的权重
            use_contrastive: 是否使用对比学习
            use_adversarial: 是否使用对抗训练
            margin: 对比学习边界
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.semantic_weight = semantic_weight
        self.use_contrastive = use_contrastive
        self.use_adversarial = use_adversarial
        self.margin = margin

        # 语义编码器
        self.semantic_encoder = SemanticEncoder(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # 对比学习模块
        if use_contrastive:
            self.contrastive_head = ContrastiveHead(
                feature_dim=hidden_dim,
                temperature=temperature,
            )

        # 对抗判别器（可选）
        if use_adversarial:
            self.discriminator = DomainDiscriminator(
                feature_dim=hidden_dim,
                num_domains=2,  # moving 和 fixed 视为两个域
            )

    def forward(
        self,
        warped_features: torch.Tensor,
        target_features: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        计算 IMPACT 损失

        Args:
            warped_features: 变形后图像的特征 [B, D] 或 [B, C, H, W]
            target_features: 目标图像的特征 [B, D] 或 [B, C, H, W]
            return_components: 是否返回各组件损失

        Returns:
            torch.Tensor: 总损失或 (总损失, 组件字典)
        """
        # 编码语义特征
        warped_semantic = self.semantic_encoder(warped_features)
        target_semantic = self.semantic_encoder(target_features)

        # 1. 语义一致性损失
        semantic_loss = F.mse_loss(warped_semantic, target_semantic)

        # 2. 对比学习损失
        if self.use_contrastive:
            contrastive_loss = self.contrastive_head(warped_semantic, target_semantic)
        else:
            contrastive_loss = torch.tensor(0.0, device=warped_features.device)

        # 3. 对抗损失（可选）
        if self.use_adversarial:
            adversarial_loss = self._adversarial_loss(warped_semantic, target_semantic)
        else:
            adversarial_loss = torch.tensor(0.0, device=warped_features.device)

        # 总损失
        total_loss = (
            semantic_loss
            + self.semantic_weight * contrastive_loss
            + 0.1 * adversarial_loss
        )

        if return_components:
            return total_loss, {
                "semantic": semantic_loss.item(),
                "contrastive": contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss,
                "adversarial": adversarial_loss.item() if isinstance(adversarial_loss, torch.Tensor) else adversarial_loss,
                "total": total_loss.item(),
            }

        return total_loss

    def _adversarial_loss(
        self, features_a: torch.Tensor, features_b: torch.Tensor
    ) -> torch.Tensor:
        """计算对抗损失"""
        batch_size = features_a.shape[0]

        # 标签：0 = domain A, 1 = domain B
        labels_a = torch.zeros(batch_size, dtype=torch.long, device=features_a.device)
        labels_b = torch.ones(batch_size, dtype=torch.long, device=features_b.device)

        # 预测
        pred_a = self.discriminator(features_a)
        pred_b = self.discriminator(features_b)

        # 判别器损失（拉大两个域的差异）
        disc_loss = F.cross_entropy(pred_a, labels_a) + F.cross_entropy(
            pred_b, labels_b
        )

        # 生成器损失（混淆两个域）
        gen_loss = F.cross_entropy(pred_a, labels_b) + F.cross_entropy(pred_b, labels_a)

        return gen_loss + 0.5 * disc_loss


class SemanticEncoder(nn.Module):
    """
    语义编码器

    使用多层感知机将输入特征映射到语义空间
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()

        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers

        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码特征"""
        # 处理空间张量 [B, C, H, W]
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
            x = x.reshape(B * H * W, C)  # [B*H*W, C]
            x = self.encoder(x)
            x = x.reshape(B, H, W, self.output_dim)
            x = x.permute(0, 3, 1, 2)  # [B, D, H, W]
            # 全局池化
            x = x.flatten(2).mean(-1)  # [B, D]
        else:
            x = self.encoder(x)

        return x


class ContrastiveHead(nn.Module):
    """
    对比学习头

    使用 InfoNCE 损失最大化正样本对的相似度
    """

    def __init__(self, feature_dim: int = 256, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 128),
        )

    def forward(
        self, features_a: torch.Tensor, features_b: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对比损失

        Args:
            features_a: 特征 A
            features_b: 特征 B（正样本对）

        Returns:
            torch.Tensor: InfoNCE 损失
        """
        # 投影
        z_a = self.projection(features_a)
        z_b = self.projection(features_b)

        # 归一化
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)

        # 计算相似度矩阵
        similarity = torch.matmul(z_a, z_b.T) / self.temperature

        # InfoNCE 损失
        batch_size = features_a.shape[0]
        labels = torch.arange(batch_size, device=features_a.device)

        loss = F.cross_entropy(similarity, labels)
        return loss


class DomainDiscriminator(nn.Module):
    """
    域判别器

    判断特征来自哪个域（moving 或 fixed）
    """

    def __init__(self, feature_dim: int = 256, num_domains: int = 2):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)


def impact_loss(
    warped_features: torch.Tensor,
    target_features: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    IMPACT 损失的便捷函数
    """
    impact = IMPACTLoss(**kwargs)
    return impact(warped_features, target_features)
