"""
IO Utilities - 输入输出工具

提供医学图像的读写、格式转换等功能
"""

import torch
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
import json


def load_image(
    path: Union[str, Path],
    return_sitk: bool = False,
) -> Union[torch.Tensor, sitk.Image, Tuple[torch.Tensor, sitk.Image]]:
    """
    加载医学图像

    支持格式：NIfTI (.nii, .nii.gz), NRRD (.nrrd), MHA (.mha), DICOM 等

    Args:
        path: 图像路径
        return_sitk: 是否返回 SimpleITK 图像

    Returns:
        图像张量，或 (张量, SimpleITK图像) 元组
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # 读取 SimpleITK 图像
    sitk_img = sitk.ReadImage(str(path))

    # 转换为 numpy
    arr = sitk.GetArrayFromImage(sitk_img)  # [D, H, W] 或 [H, W]

    # 转换为 PyTorch 张量
    if arr.ndim == 2:
        tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    else:
        tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)

    if return_sitk:
        return tensor, sitk_img
    else:
        return tensor


def save_image(
    data: Union[torch.Tensor, np.ndarray, sitk.Image],
    path: Union[str, Path],
    reference: Optional[sitk.Image] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    保存医学图像

    Args:
        data: 图像数据
        path: 保存路径
        reference: 参考图像（用于复制元数据）
        metadata: 额外元数据
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 转换为 numpy
    if isinstance(data, torch.Tensor):
        arr = data.squeeze().cpu().numpy()
    elif isinstance(data, sitk.Image):
        arr = sitk.GetArrayFromImage(data)
        # 如果提供参考图像，直接保存
        if reference is None:
            sitk.WriteImage(data, str(path))
            return
    else:
        arr = data

    # 创建 SimpleITK 图像
    if reference is not None:
        sitk_img = sitk.GetImageFromArray(arr)
        sitk_img.CopyInformation(reference)
    else:
        # 默认设置
        sitk_img = sitk.GetImageFromArray(arr)
        sitk_img.SetSpacing((1.0,) * arr.ndim)
        sitk_img.SetOrigin((0.0,) * arr.ndim)
        sitk_img.SetDirection(sitk.GetIdentityDirection(arr.ndim))

    # 保存
    sitk.WriteImage(sitk_img, str(path))

    # 保存元数据
    if metadata is not None:
        meta_path = path.with_suffix(path.suffix + ".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_deformation_field(path: Union[str, Path]) -> torch.Tensor:
    """
    加载变形场

    Args:
        path: 变形场路径

    Returns:
        变形场张量
    """
    tensor, sitk_img = load_image(path, return_sitk=True)

    # 确保维度正确 [D, H, W] 或 [2, H, W]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # [C, H, W]

    return tensor


def save_deformation_field(
    field: Union[torch.Tensor, np.ndarray],
    path: Union[str, Path],
    reference: Optional[sitk.Image] = None,
) -> None:
    """
    保存变形场

    Args:
        field: 变形场张量
        path: 保存路径
        reference: 参考图像
    """
    save_image(field, path, reference=reference)


def load_batch(
    paths: list,
    return_metadata: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
    """
    批量加载图像

    Args:
        paths: 图像路径列表
        return_metadata: 是否返回元数据

    Returns:
        批量张量 [B, 1, D, H, W] 或 (张量, 元数据列表)
    """
    images = []
    metadata = []

    for path in paths:
        tensor, sitk_img = load_image(path, return_sitk=True)
        images.append(tensor)

        if return_metadata:
            meta = {
                "spacing": sitk_img.GetSpacing(),
                "origin": sitk_img.GetOrigin(),
                "direction": sitk_img.GetDirection(),
                "size": sitk_img.GetSize(),
            }
            metadata.append(meta)

    batch = torch.stack(images, dim=0)

    if return_metadata:
        return batch, metadata
    else:
        return batch


def resample_image(
    image: sitk.Image,
    target_spacing: Tuple[float, ...],
    interpolator: int = sitk.sitkLinear,
) -> sitk.Image:
    """
    重采样图像到目标分辨率

    Args:
        image: 输入图像
        target_spacing: 目标体素间距
        interpolator: 插值器

    Returns:
        重采样后的图像
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # 计算新的尺寸
    new_size = [
        int(round(osz * osp / tsp))
        for osz, osp, tsp in zip(original_size, original_spacing, target_spacing)
    ]

    # 重采样
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(image)


def normalize_image(
    image: Union[torch.Tensor, np.ndarray, sitk.Image],
    method: str = "minmax",
    target_range: Tuple[float, float] = (0, 1),
) -> Union[torch.Tensor, np.ndarray, sitk.Image]:
    """
    归一化图像

    Args:
        image: 输入图像
        method: 归一化方法 ("minmax", "zscore", "percentile")
        target_range: 目标范围

    Returns:
        归一化后的图像
    """
    if isinstance(image, sitk.Image):
        arr = sitk.GetArrayFromImage(image)
        normalized = normalize_image(arr, method, target_range)
        result = sitk.GetImageFromArray(normalized)
        result.CopyInformation(image)
        return result

    if isinstance(image, torch.Tensor):
        arr = image.squeeze().cpu().numpy()
        normalized = normalize_image(arr, method, target_range)
        return torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0)

    # numpy 处理
    if method == "minmax":
        min_val = arr.min()
        max_val = arr.max()
        if max_val > min_val:
            normalized = (arr - min_val) / (max_val - min_val)
        else:
            normalized = arr - min_val
    elif method == "zscore":
        mean = arr.mean()
        std = arr.std()
        if std > 0:
            normalized = (arr - mean) / std
        else:
            normalized = arr - mean
    elif method == "percentile":
        p1 = np.percentile(arr, 1)
        p99 = np.percentile(arr, 99)
        normalized = np.clip(arr, p1, p99)
        if p99 > p1:
            normalized = (normalized - p1) / (p99 - p1)
        else:
            normalized = normalized - p1
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # 缩放到目标范围
    if target_range != (0, 1):
        normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]

    return normalized.astype(np.float32)
