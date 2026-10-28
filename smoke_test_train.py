"""小规模训练 smoke test - 验证管道可工作"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from experiments.train_competition import train_competition, _build_train_dataset, _compute_losses
from medreglab.core import LossBuilder
from medreglab import load_model

DATA_ROOT = str(Path(__file__).parent / "Datasets")
OUTPUT_DIR = str(Path(__file__).parent / "outputs" / "train_smoke")

print("=" * 60)
print("Smoke Test: 1 epoch, batch_size=1, 128^3")
print("=" * 60)

# 用 1 epoch 快速验证整个 pipeline
config = {
    "model": "convex_3d",
    "batch_size": 1,
    "num_epochs": 1,        # 仅 1 epoch
    "learning_rate": 1e-4,
    "target_size": [128, 128, 128],
    "loss_types": ["mse", "ncc"],
    "loss_weights": {"mse": 1.0, "ncc": 1.0},
    "use_impact_loss": False,  # 关闭避免 DINOv2 加载耗时
    "impact_weight": 0.0,
    "tv_weight": 0.05,
    "jacobian_weight": 0.01,
    "elastic_deform": False,
    "random_flip": False,
    "intensity_aug": False,
    "save_interval": 100,
    "max_grad_norm": 5.0,
    "use_scheduler": False,
    "organ_labels": ["liver", "spleen", "kidney_left", "kidney_right"],
    "seed": 42,
}

train_competition(
    config=config,
    train_data_dir=DATA_ROOT,
    val_data_dir=None,
    output_dir=OUTPUT_DIR,
)

print("\nSmoke test 完成!")
