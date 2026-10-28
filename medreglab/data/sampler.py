"""
Data Sampler - 数据采样器

提供数据集的采样策略
"""

import torch
from torch.utils.data import DataLoader, Sampler
from typing import List, Optional, Iterable
import numpy as np


class InfiniteSampler(Sampler):
    """
    无限采样器

    循环遍历数据集，用于在线训练
    """

    def __init__(self, data_source, num_samples: Optional[int] = None):
        self.data_source = data_source
        self.num_samples = num_samples or len(data_source)

    def __iter__(self):
        n = len(self.data_source)
        while True:
            indices = torch.randperm(n).tolist()
            yield from indices

    def __len__(self):
        return self.num_samples


class WeightedOrganSampler(Sampler):
    """
    加权器官采样器

    根据器官难度或重要性加权采样
    """

    def __init__(
        self,
        data_source,
        organ_weights: Optional[dict] = None,
    ):
        """
        Args:
            data_source: 数据源
            organ_weights: 器官权重 {"liver": 1.0, "spleen": 1.5, "kidney": 2.0}
        """
        self.data_source = data_source
        self.organ_weights = organ_weights or {
            "liver": 1.0,
            "spleen": 1.5,
            "kidney": 2.0,
        }

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        weights = [1.0] * len(self.data_source)

        # 根据样本属性调整权重
        for i, sample in enumerate(self.data_source):
            if hasattr(sample, "difficulty"):
                weights[i] = sample.difficulty

        weights = np.array(weights)
        weights = weights / weights.sum()
        indices = np.random.choice(
            indices, size=len(indices), replace=True, p=weights
        )
        yield from indices.tolist()

    def __len__(self):
        return len(self.data_source)


def build_dataloader(
    dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    sampler: Optional[Sampler] = None,
) -> DataLoader:
    """
    构建 DataLoader 的便捷函数

    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 数据加载线程数
        pin_memory: 是否固定内存
        drop_last: 是否丢弃最后不完整批次
        sampler: 自定义采样器

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        sampler=sampler,
    )
