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
- 分割标签：单文件多标签整数 (0=背景, 1=肝, 2=脾, 3=左肾, 4=右肾)
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
    3: "kidney_left",
    4: "kidney_right",
}

# 用于 DSC 计算的器官（不含背景）
ORGANS_FOR_DSC = ["liver", "spleen", "kidney_left", "kidney_right"]


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
                # 没有独立验证集，从 train 划分
                warnings.warn("No val/ directory; using train/ as val.")
                train_dir = self.root / "train"
                if train_dir.exists():
                    samples.extend(self._scan_paired_dir(train_dir))
        elif self.split == "test":
            test_dir = self.root / "test"
            if test_dir.exists():
                samples.extend(self._scan_paired_dir(test_dir))
            else:
                # 测试集尚未发布时，回退到 train 目录（仅用于生成训练变形场）
                warnings.warn("No test/ directory; using train/ for deformation generation.")
                train_dir = self.root / "train"
                if train_dir.exists():
                    samples.extend(self._scan_paired_dir(train_dir))
        elif self.split == "unpaired_ct":
            extra_ct_dir = self.root / "extra" / "CT" / "ct"
            if extra_ct_dir.exists():
                samples.extend(self._scan_unpaired(extra_ct_dir, "CT"))
        elif self.split == "unpaired_mr":
            extra_mr_dir = self.root / "extra" / "MR" / "mr"
            if extra_mr_dir.exists():
                samples.extend(self._scan_unpaired(extra_mr_dir, "MR"))

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
                warnings.warn(f"Skipping {case_id}: missing CT or MR image")
                continue

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
        files = {f.name: f for f in case_dir.iterdir() if f.is_file()}

        ct_path = None
        mr_path = None
        seg_ct_path = None
        seg_mr_path = None

        # 先分离 seg 和 image
        image_files = {n: p for n, p in files.items()
                       if "seg" not in n.lower() and "label" not in n.lower()}
        seg_files = {n: p for n, p in files.items()
                     if "seg" in n.lower() or "label" in n.lower()}

        # 格式 1: TCIA 风格 (img0002_tcia_CT.nii.gz / seg0002_tcia_CT.nii.gz)
        for name, path in image_files.items():
            if re.search(r"_CT.*\.nii(\.gz)?$", name, re.IGNORECASE):
                ct_path = path
            elif re.search(r"_MR.*\.nii(\.gz)?$", name, re.IGNORECASE):
                mr_path = path

        # 格式 2: {id}_ct.nii / {id}_ct-label.nii
        if ct_path is None:
            for name, path in image_files.items():
                if re.search(r"_ct\.nii(\.gz)?$", name, re.IGNORECASE):
                    ct_path = path
                    break
        if mr_path is None:
            for name, path in image_files.items():
                if re.search(r"_mr\.nii(\.gz)?$", name, re.IGNORECASE):
                    mr_path = path
                    break

        # 格式 3: 裸文件名 ct.nii / mr.nii（官方测试集 025-032 格式）
        if ct_path is None:
            for name, path in image_files.items():
                if re.match(r"^ct\.nii(\.gz)?$", name, re.IGNORECASE):
                    ct_path = path
                    break
        if mr_path is None:
            for name, path in image_files.items():
                if re.match(r"^mr\.nii(\.gz)?$", name, re.IGNORECASE):
                    mr_path = path
                    break

        # 分割标签
        for name, path in seg_files.items():
            lname = name.lower()
            if "ct" in lname and "mr" not in lname:
                seg_ct_path = path
            elif "mr" in lname:
                seg_mr_path = path

        return ct_path, mr_path, seg_ct_path, seg_mr_path

    def _scan_unpaired(self, unpaired_dir: Path, modality: str) -> List[Dict[str, Any]]:
        """扫描未配对数据目录"""
        samples = []

        for img_path in sorted(unpaired_dir.glob("*.nii*")):
            if "label" in img_path.name.lower() or "seg" in img_path.name.lower():
                continue

            # 查找对应标签
            seg_path = None
            for candidate in [
                img_path.with_name(img_path.stem + "-label" + img_path.suffix),
                img_path.with_name(img_path.stem + "_label" + img_path.suffix),
            ]:
                if candidate.exists():
                    seg_path = candidate
                    break

            sample = {
                "case_id": img_path.stem,
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
            seg_fixed = self._load_segmentation(meta["seg_fixed_path"])
        if meta.get("seg_moving_path") and Path(meta["seg_moving_path"]).exists():
            seg_moving = self._load_segmentation(meta["seg_moving_path"])

        if seg_fixed is not None or seg_moving is not None:
            sample["has_segmentations"] = True
            sample["seg_fixed"] = seg_fixed  # 字典 {organ_name: tensor}
            sample["seg_moving"] = seg_moving

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
            seg = self._load_segmentation(meta["seg_path"])
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
                # np.percentile 返回 float64，会把整个数组提升成 float64，
                # 导致 CT(float32)/MR(float64) dtype 不一致，
                # 在 grid_sample 等算子里直接报 "expected scalar type Double but found Float"。
                arr = arr.astype(np.float32)

        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

        # 调整尺寸
        if self.target_size is not None:
            tensor = self._resize_tensor(tensor, self.target_size)

        return tensor, sitk_img

    def _load_segmentation(self, path: str) -> Optional[Dict[str, torch.Tensor]]:
        """加载多标签分割并按器官拆分"""
        try:
            sitk_img = sitk.ReadImage(str(path))
            arr = sitk.GetArrayFromImage(sitk_img).astype(np.int32)
        except Exception as e:
            warnings.warn(f"Failed to load segmentation {path}: {e}")
            return None

        # 按器官拆分为二值 mask
        organ_masks = {}
        for label_id, organ_name in ORGAN_LABELS.items():
            if label_id == 0:
                continue
            binary = (arr == label_id).astype(np.float32)
            if binary.sum() == 0:
                continue
            tensor = torch.from_numpy(binary).unsqueeze(0)  # [1, D, H, W]
            if self.target_size is not None:
                import torch.nn.functional as F

                # nearest-neighbor 上采样
                tensor_5d = tensor.unsqueeze(0)
                tensor_5d = F.interpolate(
                    tensor_5d,
                    size=self.target_size,
                    mode="nearest",
                )
                tensor = tensor_5d.squeeze(0)
            organ_masks[organ_name] = tensor

        return organ_masks if organ_masks else None

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
