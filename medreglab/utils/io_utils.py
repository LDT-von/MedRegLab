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

    if sitk_img.GetDimension() not in (2, 3) or sitk_img.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("load_image requires a scalar 2D/3D image; use load_deformation_field for vector fields")
    arr = sitk.GetArrayFromImage(sitk_img)  # [D, H, W] or [H, W]
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

    # Scalar images only. Remove only known batch/channel axes; indiscriminate
    # squeeze would destroy a legitimate singleton spatial dimension.
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    elif isinstance(data, sitk.Image):
        if data.GetDimension() not in (2, 3) or data.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError("save_image requires a scalar 2D/3D image")
        sitk_img = sitk.Image(data)
        if reference is not None:
            sitk_img.CopyInformation(reference)
        arr = None
    else:
        arr = np.asarray(data)

    if arr is not None:
        if arr.ndim in (4, 5) and arr.shape[:2] == (1, 1):
            arr = arr[0, 0]
        if arr.ndim not in (2, 3):
            raise ValueError("save_image requires [H,W], [D,H,W], [1,1,H,W], or [1,1,D,H,W]")
        sitk_img = sitk.GetImageFromArray(arr, isVector=False)
        if reference is not None:
            sitk_img.CopyInformation(reference)

    # 保存
    sitk.WriteImage(sitk_img, str(path))

    # 保存元数据
    if metadata is not None:
        meta_path = path.with_suffix(path.suffix + ".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_deformation_field(
    path: Union[str, Path], return_sitk: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, sitk.Image]]:
    """
    Load a SimpleITK physical displacement vector field.

    Returns [1, 2, H, W] or [1, 3, D, H, W]. Components are physical
    LPS x/y[/z] displacement in millimetres, not voxel-index units. Spatial
    dimensions follow SimpleITK's NumPy order; with return_sitk=True the
    original image is returned as well, preserving reference geometry.

    Args:
        path: 变形场路径

    Returns:
        变形场张量
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Displacement field not found: {path}")
    sitk_img = sitk.ReadImage(str(path))
    dimension = sitk_img.GetDimension()
    if dimension not in (2, 3) or sitk_img.GetNumberOfComponentsPerPixel() != dimension:
        raise ValueError("Expected a 2D/3D vector image with one component per spatial dimension")
    arr = sitk.GetArrayFromImage(sitk_img)
    if arr.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Displacement vector field must be float32 or float64")
    if not np.isfinite(arr).all():
        raise ValueError("Displacement vector field contains NaN or Inf")
    tensor = torch.from_numpy(np.ascontiguousarray(np.moveaxis(arr, -1, 0))).unsqueeze(0)
    return (tensor, sitk_img) if return_sitk else tensor


def save_deformation_field(
    field: Union[torch.Tensor, np.ndarray],
    path: Union[str, Path],
    reference: Optional[sitk.Image] = None,
) -> None:
    """
    Save a physical LPS x/y[/z] displacement in millimetres as a vector image.

    Accepted shapes are [2,H,W], [3,D,H,W], [1,2,H,W], or [1,3,D,H,W].
    A reference supplies the physical geometry and must match the spatial size.
    This helper does not convert voxel-index displacement into millimetres.

    Args:
        field: 变形场张量
        path: 保存路径
        reference: 参考图像
    """
    arr = field.detach().cpu().numpy() if isinstance(field, torch.Tensor) else np.asarray(field)
    if arr.ndim in (4, 5) and arr.shape[0] == 1 and arr.shape[1] in (2, 3) and arr.ndim == arr.shape[1] + 2:
        arr = arr[0]
    if arr.ndim not in (3, 4) or arr.shape[0] not in (2, 3) or arr.ndim != arr.shape[0] + 1:
        raise ValueError("Expected channel-first [2,H,W], [3,D,H,W], or a single-item batch")
    if arr.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Displacement vector field must be float32 or float64")
    if not np.isfinite(arr).all():
        raise ValueError("Displacement vector field contains NaN or Inf")
    array_vector = np.ascontiguousarray(np.moveaxis(arr, 0, -1))
    image = sitk.GetImageFromArray(array_vector, isVector=True)
    if reference is not None:
        if reference.GetDimension() != arr.shape[0] or tuple(reference.GetSize()) != tuple(arr.shape[1:][::-1]):
            raise ValueError("Displacement grid does not match reference spatial size")
        image.CopyInformation(reference)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


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

    if not images:
        raise ValueError("paths must contain at least one image")
    batch = torch.cat(images, dim=0)

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
    if len(target_spacing) != image.GetDimension() or any(not np.isfinite(x) or x <= 0 for x in target_spacing):
        raise ValueError("target_spacing must contain one finite positive value per spatial dimension")

    # 计算新的尺寸
    new_size = [
        max(1, int(round(osz * osp / tsp)))
        for osz, osp, tsp in zip(original_size, original_spacing, target_spacing)
    ]

    # 重采样
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0.0)
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
        if image.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError("normalize_image requires a scalar image")
        arr = sitk.GetArrayFromImage(image)
        normalized = normalize_image(arr, method, target_range)
        result = sitk.GetImageFromArray(normalized, isVector=False)
        result.CopyInformation(image)
        return result

    if isinstance(image, torch.Tensor):
        value = image.float()
        if value.numel() == 0:
            raise ValueError("Cannot normalize an empty image")
        if method == "minmax":
            lower, upper = value.min(), value.max()
            normalized = (value - lower) / (upper - lower) if upper > lower else value - lower
        elif method == "zscore":
            mean, std = value.mean(), value.std(unbiased=False)
            normalized = (value - mean) / std if std > 0 else value - mean
        elif method == "percentile":
            lower = torch.quantile(value.flatten(), 0.01)
            upper = torch.quantile(value.flatten(), 0.99)
            clipped = value.clamp(lower, upper)
            normalized = (clipped - lower) / (upper - lower) if upper > lower else clipped - lower
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        if target_range != (0, 1):
            normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
        return normalized

    # numpy 处理
    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot normalize an empty image")
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
