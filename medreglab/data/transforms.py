"""
Data Transforms - 医学图像数据增强与预处理

提供3D医学图像的通用增强操作
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Callable, Dict, Any
import random
import math


def _spatial_tensors(sample):
    """List aligned images and mask dictionaries; reject ambiguous batches."""
    images = []
    for key in ("fixed", "moving", "image"):
        if key in sample:
            tensor = sample[key]
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 5:
                raise ValueError(f"{key} must be a tensor shaped [B,C,D,H,W]")
            if not tensor.is_floating_point():
                raise ValueError(f"{key} must have a floating-point image dtype")
            images.append((sample, key, tensor))
    masks = []
    if not images:
        if any(sample.get(key) is not None for key in ("seg_fixed", "seg_moving", "seg", "segmentations")):
            raise ValueError("Masks require a matching spatial image")
        return images, masks
    reference = images[0][2]
    for _, key, tensor in images:
        if tensor.shape[0] != reference.shape[0] or tensor.shape[-3:] != reference.shape[-3:]:
            raise ValueError(f"{key} has a different batch or spatial shape")
        if tensor.device != reference.device:
            raise ValueError("All paired images must use the same device")
    seen = set()
    for group, image_key in (("seg_fixed", "fixed"), ("seg_moving", "moving"),
                             ("seg", "image"), ("segmentations", None)):
        entries = sample.get(group)
        if entries is None:
            continue
        if image_key is not None and image_key not in sample:
            raise ValueError(f"{group} requires matching {image_key} image")
        if not isinstance(entries, dict):
            raise ValueError(f"{group} must be an organ-to-mask dictionary")
        for organ, tensor in entries.items():
            if not isinstance(tensor, torch.Tensor) or tensor.ndim not in (3, 4, 5):
                raise ValueError(f"{group}[{organ}] must have 3, 4 or 5 dimensions")
            if tensor.shape[-3:] != reference.shape[-3:]:
                raise ValueError(f"{group}[{organ}] has a different spatial shape")
            if image_key is not None and tensor.device != sample[image_key].device:
                raise ValueError(f"{group}[{organ}] and {image_key} must use the same device")
            if (tensor.ndim == 5 and tensor.shape[0] != reference.shape[0]) or (
                tensor.ndim < 5 and reference.shape[0] != 1
            ):
                raise ValueError(f"{group}[{organ}] has an ambiguous or mismatched batch")
            identity = (id(entries), organ)
            if identity not in seen:
                masks.append((entries, organ, tensor))
                seen.add(identity)
    return images, masks


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

    def __init__(self, axes: List[int] = None, p: float = 0.5):
        """
        Args:
            axes: 要翻转的轴列表 [0=D, 1=H, 2=W]
            p: 翻转概率
        """
        self.axes = [0, 1, 2] if axes is None else list(axes)
        if any(axis not in (0, 1, 2) for axis in self.axes) or len(set(self.axes)) != len(self.axes):
            raise ValueError("Flip axes must be distinct values from 0, 1, 2")
        if not 0 <= p <= 1:
            raise ValueError("Probability must be in [0, 1]")
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for axis in self.axes:
            if random.random() < self.p:
                images, masks = _spatial_tensors(sample)
                for owner, name, tensor in images + masks:
                    owner[name] = torch.flip(tensor, dims=[tensor.ndim - 3 + axis])
                # 记录翻转
                sample[f"flipped_axis_{axis}"] = True
                sample["spatial_augmented"] = True
        return sample


class RandomRotate90(MedicalImageTransform):
    """
    随机90度旋转（仅在H-W平面）

    适用于腹部医学图像
    """

    def __init__(self, p: float = 0.5):
        if not 0 <= p <= 1:
            raise ValueError("Probability must be in [0, 1]")
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.p:
            k = random.choice([1, 2, 3])  # 90, 180, 270度
            images, masks = _spatial_tensors(sample)
            for owner, name, tensor in images + masks:
                owner[name] = torch.rot90(tensor, k, dims=[-2, -1])
            sample["rotated_k"] = k
            sample["spatial_augmented"] = True
        return sample


class RandomElasticDeformation(MedicalImageTransform):
    """
    随机弹性形变增强

    通过控制点网格的三线性插值生成随机形变。
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
        if not isinstance(num_ctrl_points, int) or num_ctrl_points < 2 or not math.isfinite(deform_std) or deform_std < 0 or not 0 <= p <= 1:
            raise ValueError("Expected >=2 control points, nonnegative std and probability in [0, 1]")
        self.num_ctrl_points = num_ctrl_points
        self.deform_std = deform_std
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.p:
            return sample

        images, masks = _spatial_tensors(sample)
        if not images:
            return sample
        reference = images[0][2]
        shape = reference.shape[-3:]

        # 生成随机控制点位移
        dtype = reference.dtype if reference.dtype in (torch.float32, torch.float64) else torch.float32
        displacement = torch.randn(
            3, self.num_ctrl_points, self.num_ctrl_points, self.num_ctrl_points,
            device=reference.device, dtype=dtype,
        ) * self.deform_std

        # 上采样到目标尺寸
        displacement = F.interpolate(
            displacement.unsqueeze(0),
            size=shape,
            mode="trilinear",
            align_corners=True,
        ).squeeze(0)

        # 应用位移场
        for owner, name, tensor in images:
            owner[name] = self._apply_displacement(tensor, displacement, mode="bilinear")
        for owner, name, tensor in masks:
            input_ndim = tensor.ndim
            if input_ndim == 3:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            elif input_ndim == 4:
                tensor = tensor.unsqueeze(0)
            warped = self._apply_displacement(tensor, displacement, mode="nearest")
            owner[name] = warped.squeeze(0).squeeze(0) if input_ndim == 3 else (
                warped.squeeze(0) if input_ndim == 4 else warped
            )
        sample["spatial_augmented"] = True

        return sample

    def _apply_displacement(
        self, image: torch.Tensor, displacement: torch.Tensor, mode: str = "bilinear"
    ) -> torch.Tensor:
        """应用位移场到图像

        Args:
            image: [B, C, D, H, W]
            displacement: [3, D, H, W]，轴序 (x=W, y=H, z=D)
        """
        if image.ndim != 5 or mode not in ("bilinear", "nearest"):
            raise ValueError("Expected a 5D image and bilinear/nearest sampling")
        B, C, D, H, W = image.shape
        if displacement.ndim == 4:
            displacement = displacement.unsqueeze(0)
        if displacement.ndim != 5 or displacement.shape[1:] != (3, D, H, W):
            raise ValueError("Displacement must have shape [1|B,3,D,H,W]")
        if displacement.shape[0] not in (1, B):
            raise ValueError("Displacement batch must be one or match image batch")
        original_dtype = image.dtype
        work = image if image.dtype in (torch.float32, torch.float64) else image.float()
        displacement = displacement.to(device=work.device, dtype=work.dtype).expand(B, -1, -1, -1, -1)

        # 基础网格：与 convex_3d_working.py 一致。
        # F.grid_sample 的 3D 约定是 grid 为 [B, D_out, H_out, W_out, 3]，
        # 最后一维为 (x, y, z)。
        grid_d = torch.linspace(-1, 1, D, device=work.device, dtype=work.dtype)
        grid_h = torch.linspace(-1, 1, H, device=work.device, dtype=work.dtype)
        grid_w = torch.linspace(-1, 1, W, device=work.device, dtype=work.dtype)
        grid_d, grid_h, grid_w = torch.meshgrid(
            grid_d, grid_h, grid_w, indexing="ij"
        )
        base_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1)  # [D, H, W, 3]

        # 位移归一化到 [-1, 1]，并按 (x, y, z) 重排到网格最后一维
        scales = work.new_tensor([2 / max(W - 1, 1), 2 / max(H - 1, 1), 2 / max(D - 1, 1)])
        disp_norm = displacement * scales.view(1, 3, 1, 1, 1)
        warped_grid = base_grid.unsqueeze(0) + disp_norm.permute(0, 2, 3, 4, 1)

        # 采样
        warped = F.grid_sample(work, warped_grid, mode=mode, padding_mode="border", align_corners=True)
        return warped.to(dtype=original_dtype)


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
