"""
Data Transforms - 医学图像数据增强与预处理

提供3D医学图像的通用增强操作
"""

import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from typing import Optional, Tuple, List, Callable, Dict, Any, Union
import random


class MedicalImageTransform:
    """
    医学图像预处理变换基类
    """

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class RandomFlip3D(MedicalImageTransform):
    """
    随机3D翻转增强

    沿指定轴随机翻转图像
    """

    def __init__(self, axes: List[int] = [0, 1, 2], p: float = 0.5):
        """
        Args:
            axes: 要翻转的轴列表 [0=D, 1=H, 2=W]
            p: 翻转概率
        """
        self.axes = axes
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for axis in self.axes:
            if random.random() < self.p:
                # 翻转图像
                if "fixed" in sample and isinstance(sample["fixed"], torch.Tensor):
                    sample["fixed"] = torch.flip(sample["fixed"], dims=[axis + 2])
                if "moving" in sample and isinstance(sample["moving"], torch.Tensor):
                    sample["moving"] = torch.flip(sample["moving"], dims=[axis + 2])
                # 翻转分割标签
                if "segmentations" in sample:
                    for organ in sample["segmentations"]:
                        seg = sample["segmentations"][organ]
                        sample["segmentations"][organ] = torch.flip(seg, dims=[axis + 1])
                # 记录翻转
                sample[f"flipped_axis_{axis}"] = True
        return sample


class RandomRotate90(MedicalImageTransform):
    """
    随机90度旋转（仅在H-W平面）

    适用于腹部医学图像
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.p:
            k = random.choice([1, 2, 3])  # 90, 180, 270度
            spatial_dims = [2, 3, 4]  # D, H, W

            if "fixed" in sample and isinstance(sample["fixed"], torch.Tensor):
                sample["fixed"] = torch.rot90(sample["fixed"], k, dims=[3, 4])
            if "moving" in sample and isinstance(sample["moving"], torch.Tensor):
                sample["moving"] = torch.rot90(sample["moving"], k, dims=[3, 4])
            if "segmentations" in sample:
                for organ in sample["segmentations"]:
                    seg = sample["segmentations"][organ]
                    sample["segmentations"][organ] = torch.rot90(seg, k, dims=[2, 3])
            sample["rotated_k"] = k
        return sample


class RandomElasticDeformation(MedicalImageTransform):
    """
    随机弹性形变增强

    使用B样条网格生成平滑的随机形变
    """

    def __init__(
        self,
        num_ctrl_points: int = 4,
        deform_std: float = 15.0,
        p: float = 0.3,
    ):
        """
        Args:
            num_ctrl_points: 控制点网格大小
            deform_std: 形变标准差（像素）
            p: 应用概率
        """
        self.num_ctrl_points = num_ctrl_points
        self.deform_std = deform_std
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.p:
            return sample

        # 获取图像尺寸
        if "moving" in sample and isinstance(sample["moving"], torch.Tensor):
            shape = sample["moving"].shape[2:]  # D, H, W
        elif "fixed" in sample and isinstance(sample["fixed"], torch.Tensor):
            shape = sample["fixed"].shape[2:]
        else:
            return sample

        # 生成随机控制点位移
        displacement = torch.randn(3, self.num_ctrl_points, self.num_ctrl_points, self.num_ctrl_points) * self.deform_std

        # 上采样到目标尺寸
        displacement = F.interpolate(
            displacement.unsqueeze(0),
            size=shape,
            mode="trilinear",
            align_corners=True,
        ).squeeze(0)

        # 应用位移场
        if "moving" in sample:
            sample["moving"] = self._apply_displacement(sample["moving"], displacement)
        if "fixed" in sample:
            sample["fixed"] = self._apply_displacement(sample["fixed"], displacement)
        if "segmentations" in sample:
            for organ in sample["segmentations"]:
                seg = sample["segmentations"][organ]
                sample["segmentations"][organ] = self._apply_displacement(
                    seg.unsqueeze(0), displacement
                ).squeeze(0)

        return sample

    def _apply_displacement(
        self, image: torch.Tensor, displacement: torch.Tensor
    ) -> torch.Tensor:
        """应用位移场到图像

        Args:
            image: [B, C, D, H, W]
            displacement: [3, D, H, W]，轴序 (x=W, y=H, z=D)
        """
        B, C, D, H, W = image.shape

        # 基础网格：与 convex_3d_working.py 一致。
        # F.grid_sample 的 3D 约定是 grid 为 [B, D_out, H_out, W_out, 3]，
        # 最后一维为 (x, y, z)。
        grid_d = torch.linspace(-1, 1, D, device=image.device)
        grid_h = torch.linspace(-1, 1, H, device=image.device)
        grid_w = torch.linspace(-1, 1, W, device=image.device)
        grid_d, grid_h, grid_w = torch.meshgrid(
            grid_d, grid_h, grid_w, indexing="ij"
        )
        base_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1)  # [D, H, W, 3]

        # 位移归一化到 [-1, 1]，并按 (x, y, z) 重排到网格最后一维
        disp_norm = displacement.clone()
        disp_norm[0] = 2 * disp_norm[0] / max(W - 1, 1)  # x -> W
        disp_norm[1] = 2 * disp_norm[1] / max(H - 1, 1)  # y -> H
        disp_norm[2] = 2 * disp_norm[2] / max(D - 1, 1)  # z -> D
        # [3, D, H, W] -> [D, H, W, 3]
        disp_norm = disp_norm.permute(1, 2, 3, 0)

        warped_grid = (base_grid + disp_norm).unsqueeze(0)  # [1, D, H, W, 3]

        # grid_sample 要求 grid 与 input dtype 一致
        warped_grid = warped_grid.to(dtype=image.dtype)

        # 采样
        image = F.grid_sample(
            image, warped_grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return image


class RandomIntensityAugmentation(MedicalImageTransform):
    """
    随机强度增强

    包括噪声注入、对比度调整、亮度调整等
    """

    def __init__(
        self,
        noise_std: float = 0.05,
        brightness_range: float = 0.1,
        contrast_range: float = 0.1,
        p: float = 0.5,
    ):
        """
        Args:
            noise_std: 高斯噪声标准差
            brightness_range: 亮度调整范围
            contrast_range: 对比度调整范围
            p: 应用概率
        """
        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for key in ["moving"]:
            if key in sample and isinstance(sample[key], torch.Tensor):
                img = sample[key]

                # 亮度调整
                if random.random() < self.p:
                    brightness = random.uniform(
                        -self.brightness_range, self.brightness_range
                    )
                    img = img + brightness

                # 对比度调整
                if random.random() < self.p:
                    contrast = 1.0 + random.uniform(
                        -self.contrast_range, self.contrast_range
                    )
                    mean = img.mean()
                    img = (img - mean) * contrast + mean

                # 高斯噪声
                if random.random() < self.p and self.noise_std > 0:
                    noise = torch.randn_like(img) * self.noise_std
                    img = img + noise

                sample[key] = img
        return sample


class ComposeTransforms:
    """
    组合多个变换
    """

    def __init__(self, transforms: List[MedicalImageTransform]):
        self.transforms = transforms

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomAugmentation:
    """
    综合随机增强管道

    适用于配准训练的综合数据增强
    """

    def __init__(
        self,
        elastic_deform: bool = True,
        random_flip: bool = True,
        intensity_aug: bool = True,
        elastic_p: float = 0.3,
        flip_p: float = 0.5,
        intensity_p: float = 0.5,
    ):
        """
        Args:
            elastic_deform: 是否启用弹性形变
            random_flip: 是否启用随机翻转
            intensity_aug: 是否启用强度增强
            elastic_p: 弹性形变概率
            flip_p: 翻转概率
            intensity_p: 强度增强概率
        """
        transforms = []
        if random_flip:
            transforms.append(RandomFlip3D(p=flip_p))
        if elastic_deform:
            transforms.append(RandomElasticDeformation(p=elastic_p))
        if intensity_aug:
            transforms.append(RandomIntensityAugmentation(p=intensity_p))

        self.transform = ComposeTransforms(transforms)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return self.transform(sample)


def build_transforms(
    split: str = "train",
    target_size: Optional[Tuple[int, ...]] = None,
    elastic_deform: bool = True,
    random_flip: bool = True,
    intensity_aug: bool = True,
    **kwargs,
) -> Optional[Callable]:
    """
    根据数据集划分构建变换管道

    Args:
        split: 数据集划分 ("train" 或 "val/test")
        target_size: 目标尺寸
        elastic_deform: 是否启用弹性形变
        random_flip: 是否启用随机翻转
        intensity_aug: 是否启用强度增强

    Returns:
        变换函数或 None
    """
    if split == "train":
        return RandomAugmentation(
            elastic_deform=elastic_deform,
            random_flip=random_flip,
            intensity_aug=intensity_aug,
            **kwargs,
        )
    else:
        # 验证/测试集不做增强
        return None
