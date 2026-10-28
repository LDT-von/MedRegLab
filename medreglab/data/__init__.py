"""
MedRegLab Data Module - 比赛数据集支持
"""

from .dataset import (
    CompetitionDataset,
    PairedVolumeDataset,
    ORGAN_LABELS,
    ORGANS_FOR_DSC,
    competition_masks_from_labels,
)
from .transforms import (
    RandomAugmentation,
    build_transforms,
)
from .sampler import build_dataloader
from .dataset import build_competition_dataloader

__all__ = [
    "CompetitionDataset",
    "PairedVolumeDataset",
    "ORGAN_LABELS",
    "ORGANS_FOR_DSC",
    "competition_masks_from_labels",
    "RandomAugmentation",
    "build_transforms",
    "build_dataloader",
    "build_competition_dataloader",
]
