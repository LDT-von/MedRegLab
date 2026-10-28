"""
Competition Dataset - 比赛数据集加载器（适配实际数据格式）

数据格式：
- 配对训练 (Datasets/train/)
    - 001-008: TCIA 来源，img{ID}_tcia_CT.nii.gz / img{ID}_tcia_MR.nii.gz
    - 009-024: 主办方格式，{ID}_ct.nii / {ID}_mr.nii
- 未配对辅助 (Datasets/extra/)
    - CT/ct/img*.nii.gz
    - MR/mr/img*.nii.gz
- 测试数据 (按发布)
- 分割标签：0=背景，1=肝，2=脾，3=肾；001-008 另有标签 4 为另一侧肾。
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any, Callable
import warnings
import re


# 器官标签映射（多标签整数 → 器官名）
ORGAN_LABELS = {
    0: "background",
    1: "liver",
    2: "spleen",
    3: "kidney",
    4: "kidney_001_008_second_side",
}

# 用于 DSC 计算的器官（不含背景）
ORGANS_FOR_DSC = ["liver", "spleen", "kidney"]


def competition_masks_from_labels(label_array, case_id):
    """Return all three organ masks for one multi-label image.

    According to the organizer's protocol, cases 001–008 use 3 and 4 for
    the two kidneys, whereas numbered cases 009 and later use 3 for kidney.
    Unknown/non-numeric IDs (e.g. unpaired auxiliary images) retain both
    kidney labels rather than silently discard a side.
    """
    if isinstance(label_array, torch.Tensor):
        label_array = label_array.detach().cpu().numpy()
    labels = np.asarray(label_array)
    if labels.ndim != 3 or any(size == 0 for size in labels.shape) or not np.isfinite(labels).all():
        raise ValueError("Segmentation must be a finite 3D label array")
    if not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Segmentation labels must be integers")
    unknown = set(np.unique(labels).tolist()) - {0, 1, 2, 3, 4}
    if unknown:
        raise ValueError(f"Unexpected segmentation labels: {sorted(unknown)}")
    case_text = str(case_id)
    known_numbered_case = case_text.isdigit()
    both_kidneys = not known_numbered_case or 1 <= int(case_text) <= 8
    if known_numbered_case and not both_kidneys and np.any(labels == 4):
        raise ValueError(f"Case {case_text} has unsupported label 4 under competition rules")
    return {
        "liver": labels == 1,
        "spleen": labels == 2,
        "kidney": (labels == 3) | ((labels == 4) if both_kidneys else False),
    }


class CompetitionDataset(Dataset):
    """
    比赛数据集 - 适配实际数据格式

    支持：
    - 配对训练数据：001-024
    - 未配对辅助数据（CT/MR）
    - 测试数据

    分割标签为单文件多标签整数格式
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transforms: Optional[Callable] = None,
        normalize: bool = True,
        target_size: Optional[Tuple[int, ...]] = None,
        intensity_norm: str = "percentile",
        organs: Optional[List[str]] = None,
        ct_clip: Tuple[float, float] = (-200, 500),  # HU
        mr_clip_percentile: Tuple[float, float] = (0.5, 99.5),
    ):
        """
        Args:
            root: 数据根目录（含 train/, extra/ 等子目录）
            split: "train" | "val" | "test" | "unpaired"
            transforms: 数据增强
            normalize: 是否归一化
            target_size: 目标尺寸 (D, H, W)
            intensity_norm: 归一化方法
            organs: 参与 DSC 计算的器官
            ct_clip: CT 强度裁剪范围 (HU 单位)
            mr_clip_percentile: MR 强度百分位裁剪
        """
        if intensity_norm != "percentile":
            raise ValueError("Only percentile intensity normalization is implemented")
        if (len(ct_clip) != 2 or not np.isfinite(ct_clip).all() or ct_clip[0] >= ct_clip[1]):
            raise ValueError("ct_clip must contain two finite increasing values")
        if (len(mr_clip_percentile) != 2 or not np.isfinite(mr_clip_percentile).all()
                or not 0 <= mr_clip_percentile[0] < mr_clip_percentile[1] <= 100):
            raise ValueError("mr_clip_percentile must contain increasing values in [0, 100]")
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.normalize = normalize
        self.target_size = target_size
        self.intensity_norm = intensity_norm
        self.organs = organs or ORGANS_FOR_DSC
        self.ct_clip = ct_clip
        self.mr_clip_percentile = mr_clip_percentile

        # 扫描数据
        self.samples = self._scan_dataset()
        if len(self.samples) == 0:
            warnings.warn(f"No samples found in {self.root} for split={split}")

    def _scan_dataset(self) -> List[Dict[str, Any]]:
        """扫描数据集目录"""
        samples = []

        if self.split == "train":
            # 配对训练数据
            train_dir = self.root / "train"
            if train_dir.exists():
                samples.extend(self._scan_paired_dir(train_dir))
        elif self.split in ("val", "validation"):
            val_dir = self.root / "val"
            if val_dir.exists():
                samples.extend(self._scan_paired_dir(val_dir))
            else:
                raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        elif self.split == "test":
            test_dir = self.root / "test"
            if test_dir.exists():
                samples.extend(self._scan_paired_dir(test_dir))
            else:
                raise FileNotFoundError(f"Test directory not found: {test_dir}")
        elif self.split == "unpaired_ct":
            extra_ct_dir = self.root / "extra" / "CT" / "ct"
            if extra_ct_dir.exists():
                samples.extend(self._scan_unpaired(extra_ct_dir, "CT"))
        elif self.split == "unpaired_mr":
            extra_mr_dir = self.root / "extra" / "MR" / "mr"
            if extra_mr_dir.exists():
                samples.extend(self._scan_unpaired(extra_mr_dir, "MR"))
        else:
            raise ValueError(f"Unsupported data split: {self.split!r}")

        return samples

    def _scan_paired_dir(self, paired_dir: Path) -> List[Dict[str, Any]]:
        """扫描配对数据目录（001-024）"""
        samples = []

        for case_dir in sorted(paired_dir.iterdir()):
            if not case_dir.is_dir():
                continue

            case_id = case_dir.name
            ct_path, mr_path, seg_ct_path, seg_mr_path = self._find_paired_files(case_dir)

            if ct_path is None or mr_path is None:
                raise ValueError(f"Case {case_id} is missing a CT or MR image in {case_dir}")

            sample = {
                "case_id": case_id,
                "fixed_path": str(ct_path),       # 固定图像 = CT
                "moving_path": str(mr_path),     # 浮动图像 = MR
                "fixed_modality": "CT",
                "moving_modality": "MR",
                "seg_fixed_path": str(seg_ct_path) if seg_ct_path else None,
                "seg_moving_path": str(seg_mr_path) if seg_mr_path else None,
            }
            samples.append(sample)

        return samples

    def _find_paired_files(self, case_dir: Path):
        """查找病例目录中的配对文件（兼容两种命名）"""
        candidates = {"ct_image": [], "mr_image": [], "ct_seg": [], "mr_seg": []}
        for path in sorted(case_dir.iterdir()):
            name = path.name.lower()
            if not path.is_file() or not re.search(r"\.nii(\.gz)?$", name):
                continue
            is_seg = "seg" in name or "label" in name
            ct = re.search(r"(?:^|[_-])ct(?=$|[_.-])", name) is not None
            mr = re.search(r"(?:^|[_-])mr(?=$|[_.-])", name) is not None
            if ct and mr:
                raise ValueError(f"Ambiguous CT/MR file name in {case_dir}: {path.name}")
            if ct or mr:
                candidates[("ct" if ct else "mr") + ("_seg" if is_seg else "_image")].append(path)
        for kind, paths in candidates.items():
            if len(paths) > 1:
                raise ValueError(f"Multiple {kind} candidates in {case_dir}: {[p.name for p in paths]}")
        return tuple(candidates[kind][0] if candidates[kind] else None
                     for kind in ("ct_image", "mr_image", "ct_seg", "mr_seg"))

    def _scan_unpaired(self, unpaired_dir: Path, modality: str) -> List[Dict[str, Any]]:
        """扫描未配对数据目录"""
        samples = []

        for img_path in sorted(unpaired_dir.glob("*.nii*")):
            if "label" in img_path.name.lower() or "seg" in img_path.name.lower():
                continue

            # 查找对应标签
            seg_path = None
            name = img_path.name
            suffix = ".nii.gz" if name.lower().endswith(".nii.gz") else img_path.suffix
            stem = name[:-len(suffix)]
            for candidate in [
                img_path.with_name(stem + "-label" + suffix),
                img_path.with_name(stem + "_label" + suffix),
            ]:
                if candidate.exists():
                    seg_path = candidate
                    break

            sample = {
                "case_id": stem,
                "image_path": str(img_path),
                "modality": modality,
                "seg_path": str(seg_path) if seg_path else None,
            }
            samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_meta = self.samples[idx]

        if "fixed_path" in sample_meta and "moving_path" in sample_meta:
            return self._get_paired_sample(sample_meta)
        else:
            return self._get_unpaired_sample(sample_meta)

    def _get_paired_sample(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """获取配对样本"""
        # 加载固定图像 (CT)
        fixed_tensor, fixed_sitk = self._load_and_preprocess(
            meta["fixed_path"], modality="CT"
        )
        # 加载浮动图像 (MR)
        moving_tensor, moving_sitk = self._load_and_preprocess(
            meta["moving_path"], modality="MR"
        )

        sample = {
            "case_id": meta["case_id"],
            "fixed": fixed_tensor,
            "fixed_sitk": fixed_sitk,
            "moving": moving_tensor,
            "moving_sitk": moving_sitk,
            "has_segmentations": False,
        }

        # 加载分割标签
        seg_fixed = None
        seg_moving = None
        if meta.get("seg_fixed_path") and Path(meta["seg_fixed_path"]).exists():
            seg_fixed = self._load_segmentation(meta["seg_fixed_path"], meta["case_id"], fixed_sitk)
        if meta.get("seg_moving_path") and Path(meta["seg_moving_path"]).exists():
            seg_moving = self._load_segmentation(meta["seg_moving_path"], meta["case_id"], moving_sitk)

        if seg_fixed is not None and seg_moving is not None:
            sample["has_segmentations"] = True
            sample["seg_fixed"] = seg_fixed  # 字典 {organ_name: tensor}
            sample["seg_moving"] = seg_moving
        elif seg_fixed is not None or seg_moving is not None:
            warnings.warn(f"Case {meta['case_id']} has only one segmentation; paired DSC is unavailable")

        # 应用变换
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _get_unpaired_sample(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """获取未配对样本"""
        image_tensor, sitk_img = self._load_and_preprocess(
            meta["image_path"], modality=meta["modality"]
        )

        sample = {
            "case_id": meta["case_id"],
            "image": image_tensor,
            "sitk_img": sitk_img,
            "modality": meta["modality"],
            "has_segmentations": False,
        }

        if meta.get("seg_path") and Path(meta["seg_path"]).exists():
            seg = self._load_segmentation(meta["seg_path"], meta["case_id"], sitk_img)
            if seg is not None:
                sample["seg"] = seg
                sample["has_segmentations"] = True

        return sample

    def _load_and_preprocess(
        self, path: str, modality: str
    ) -> Tuple[torch.Tensor, sitk.Image]:
        """加载并预处理图像"""
        sitk_img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)

        if self.normalize:
            if modality == "CT":
                # CT 使用 HU 单位裁剪 + 归一化
                arr = np.clip(arr, self.ct_clip[0], self.ct_clip[1])
                arr = (arr - self.ct_clip[0]) / (self.ct_clip[1] - self.ct_clip[0])
            else:  # MR
                # MR 使用百分位裁剪 + 归一化
                p_low, p_high = self.mr_clip_percentile
                lo, hi = np.percentile(arr, [p_low, p_high])
                arr = np.clip(arr, lo, hi)
                if hi > lo:
                    arr = (arr - lo) / (hi - lo)
                else:
                    # A constant image has no intensity range to normalize.
                    arr = np.zeros_like(arr)
                # np.percentile 返回 float64，会把整个数组提升成 float64，
                # 导致 CT(float32)/MR(float64) dtype 不一致，
                # 在 grid_sample 等算子里直接报 "expected scalar type Double but found Float"。
                arr = arr.astype(np.float32)

        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

        # 调整尺寸
        if self.target_size is not None:
            tensor = self._resize_tensor(tensor, self.target_size)

        return tensor, sitk_img

    def _load_segmentation(self, path: str, case_id: str, reference: sitk.Image = None) -> Dict[str, torch.Tensor]:
        """Load the three masks, preserving empty organs and spatial alignment."""
        sitk_img = sitk.ReadImage(str(path))
        if reference is not None:
            if sitk_img.GetSize() != reference.GetSize():
                raise ValueError(f"Segmentation size differs from image: {path}")
            for getter in ("GetSpacing", "GetOrigin", "GetDirection"):
                if not np.allclose(getattr(sitk_img, getter)(), getattr(reference, getter)(), rtol=0, atol=1e-5):
                    raise ValueError(f"Segmentation geometry differs from image ({getter}): {path}")
        masks = competition_masks_from_labels(sitk.GetArrayFromImage(sitk_img), case_id)
        tensor = torch.from_numpy(np.stack([masks[organ] for organ in ORGANS_FOR_DSC]).astype(np.float32)).unsqueeze(0)
        if self.target_size is not None and tuple(tensor.shape[2:]) != tuple(self.target_size):
            # Match the image resize coordinates (align_corners=True), then use
            # nearest-neighbor label sampling on the same grid.
            import torch.nn.functional as F
            d, h, w = self.target_size
            grid_d, grid_h, grid_w = torch.meshgrid(
                torch.linspace(-1, 1, d), torch.linspace(-1, 1, h),
                torch.linspace(-1, 1, w), indexing="ij",
            )
            grid = torch.stack([grid_w, grid_h, grid_d], dim=-1).unsqueeze(0)
            tensor = F.grid_sample(tensor, grid, mode="nearest", padding_mode="border", align_corners=True)
        return {organ: tensor[0, index:index + 1] for index, organ in enumerate(ORGANS_FOR_DSC)}

    def _resize_tensor(
        self, tensor: torch.Tensor, target_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """调整张量尺寸"""
        import torch.nn.functional as F

        if tensor.shape[2:] == tuple(target_size):
            return tensor

        mode = "trilinear" if len(target_size) == 3 else "bilinear"
        return F.interpolate(
            tensor, size=target_size, mode=mode, align_corners=True
        )

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """获取样本元信息（不含数据）"""
        return self.samples[idx].copy()


class PairedVolumeDataset(Dataset):
    """简化的配对数据集（用于训练循环）"""

    def __init__(
        self,
        root: Union[str, Path],
        target_size: Optional[Tuple[int, ...]] = None,
        ct_clip: Tuple[float, float] = (-200, 500),
        load_seg: bool = True,
    ):
        self.dataset = CompetitionDataset(
            root=root,
            split="train",
            target_size=target_size,
            ct_clip=ct_clip,
        )
        self.load_seg = load_seg

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def build_competition_dataloader(
    data_root: Union[str, Path],
    split: str = "train",
    batch_size: int = 1,
    num_workers: int = 0,
    target_size: Optional[Tuple[int, ...]] = None,
    transforms: Optional[Callable] = None,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    构建比赛数据集的 DataLoader

    Args:
        data_root: 数据根目录（含 train/、extra/ 等）
        split: 数据集划分
        batch_size: 批次大小
        num_workers: 数据加载线程数
        target_size: 目标尺寸
        transforms: 数据增强
        shuffle: 是否打乱
        drop_last: 是否丢弃最后不完整批次

    Returns:
        DataLoader
    """
    dataset = CompetitionDataset(
        root=data_root,
        split=split,
        transforms=transforms,
        target_size=target_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )
