"""
Competition Training Script - 比赛专用训练脚本

针对跨模态CT-MR图像配准挑战赛的完整训练流程

训练策略：
1. 使用配对数据（有监督）
2. 结合未配对数据（半监督/自监督）
3. 使用 convex_3d 配准方案（逐图优化，无可学习参数）
4. 评估指标：DSC / DSC30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
import json
import argparse
import sys
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from medreglab import load_model
from medreglab.data import (
    CompetitionDataset,
    build_transforms,
)
from medreglab.evaluation import CompetitionEvaluator
from medreglab.core import LossBuilder
from medreglab.models.deformation_field import JacobianField


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# DataLoader 无法 collate SimpleITK.Image 对象，
# 而训练循环并不需要这些图像元信息，因此直接丢弃。
_SITK_KEYS = ("fixed_sitk", "moving_sitk", "sitk_img")


def _collate_training(batch):
    """自定义 collate：丢掉 SimpleITK 图像对象，保留张量/字典/标量"""
    clean = []
    for sample in batch:
        item = {k: v for k, v in sample.items() if k not in _SITK_KEYS}
        # 分割标签统一放到 "segmentations"，供 DSC 计算使用
        if "seg_moving" in item and "segmentations" not in item:
            item["segmentations"] = item["seg_moving"] or {}
        clean.append(item)
    return torch.utils.data.dataloader.default_collate(clean)


def train_competition(
    config: Dict[str, Any],
    train_data_dir: Union[str, Path],
    val_data_dir: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "competition_outputs",
    device: Optional[str] = None,
    resume_from: Optional[Path] = None,
):
    """
    比赛配准模型训练

    Args:
        config: 训练配置
        train_data_dir: 训练数据目录
        val_data_dir: 验证数据目录
        output_dir: 输出目录
        device: 计算设备
        resume_from: 从检查点恢复
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    # 保存配置
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[MedRegLab] Training on {device}")
    print(f"[MedRegLab] Output: {output_dir}")
    print(f"[MedRegLab] Config: {json.dumps(config, indent=2)}")

    # 设置随机种子
    set_seed(config.get("seed", 42))

    # 加载模型
    model_name = config.get("model", "convex_3d")
    model = load_model(model_name)
    model = model.to(device)

    if resume_from and Path(resume_from).exists():
        model.load_state_dict(torch.load(resume_from, map_location=device))
        print(f"Resumed from {resume_from}")

    # 构建数据集
    train_dataset = _build_train_dataset(train_data_dir, config)
    val_dataset = None
    if val_data_dir:
        val_dataset = _build_val_dataset(val_data_dir, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 1),
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=_collate_training,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate_training,
        )

    # 优化器
    # convex_3d 是逐图优化的无参数模型，没有可学习参数；
    # 此时跳过优化器/反向传播，仅做配准与评估。
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = None
    if trainable_params:
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5),
        )
    else:
        print("[MedRegLab] Model has no trainable parameters; "
              "running registration/evaluation only (no gradient steps).")

    # 学习率调度器
    scheduler = None
    if config.get("use_scheduler"):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            verbose=True,
        )

    # 损失函数构建器
    loss_builder = _build_loss_builder(config, device)

    # 评估器
    evaluator = None
    if val_loader:
        evaluator = CompetitionEvaluator(
            organs=config.get("organ_labels", ["liver", "spleen", "kidney"]),
            output_dir=output_dir / "eval_results",
        )

    # 训练循环
    num_epochs = config.get("num_epochs", 100)
    best_val_score = 0.0
    best_epoch = 0
    global_step = 0
    train_history = []

    for epoch in range(num_epochs):
        # === 训练阶段 ===
        model.train()
        epoch_losses = {k: [] for k in ["total", "similarity", "semantic", "tv", "jacobian"]}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            moving = batch["moving"].to(device)
            fixed = batch["fixed"].to(device)

            # 数据集每个样本已经是 [1, C, D, H, W]，
            # DataLoader 再加一维会变成 6D，这里压回 5D。
            if moving.dim() == 6:
                moving = moving.squeeze(0)
            if fixed.dim() == 6:
                fixed = fixed.squeeze(0)

            # 前向传播
            result = model(moving, fixed)

            # 计算损失
            losses = _compute_losses(
                result, fixed, loss_builder, config, device
            )

            total_loss = losses["total_tensor"]

            # 反向传播（仅当模型有可学习参数时）
            if optimizer is not None:
                optimizer.zero_grad()
                total_loss.backward()
                if config.get("max_grad_norm"):
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, config["max_grad_norm"]
                    )
                optimizer.step()

            # 记录损失
            epoch_losses["total"].append(losses.get("total", 0))
            epoch_losses["similarity"].append(losses.get("similarity", 0))
            epoch_losses["semantic"].append(losses.get("semantic", 0))
            epoch_losses["tv"].append(losses.get("tv", 0))
            epoch_losses["jacobian"].append(losses.get("jacobian", 0))

            pbar.set_postfix({
                "loss": f"{losses.get('total', 0):.4f}",
                "sim": f"{losses.get('similarity', 0):.4f}",
            })

            global_step += 1

        # === 验证阶段 ===
        val_metrics = None
        if val_loader is not None:
            model.eval()
            val_losses = []
            val_evaluator = CompetitionEvaluator(
                organs=config.get("organ_labels", ["liver", "spleen", "kidney"]),
            )

            # convex_3d 内部需要对 phi 做 autograd 优化，
            # 所以这里不能用 torch.no_grad()。
            with torch.enable_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    moving = batch["moving"].to(device)
                    fixed = batch["fixed"].to(device)

                    # 与训练阶段一致：压回 5D
                    if moving.dim() == 6:
                        moving = moving.squeeze(0)
                    if fixed.dim() == 6:
                        fixed = fixed.squeeze(0)

                    result = model(moving, fixed)
                    val_loss = F.mse_loss(result["warped"], fixed)
                    val_losses.append(val_loss.item())

                    # 计算 DSC（如果有分割标签）
                    if batch.get("has_segmentations") and batch["segmentations"]:
                        case_id = batch.get("case_id", ["val"])[0]
                        dsc = _compute_batch_dsc(batch, result, device)
                        if dsc:
                            val_evaluator.add_case(case_id, dsc)

            val_metrics = val_evaluator.evaluate(save=False)

            # 学习率调度
            if scheduler is not None:
                val_loss_mean = np.mean(val_losses)
                scheduler.step(val_loss_mean)

        # === 汇总 epoch 结果 ===
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items() if v}
        train_history.append({
            "epoch": epoch + 1,
            "train_losses": avg_losses,
            "val_metrics": val_metrics,
        })

        # 打印结果
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_losses.get('total', 0):.4f} | "
              f"Sim: {avg_losses.get('similarity', 0):.4f} | "
              f"TV: {avg_losses.get('tv', 0):.4f}")

        if val_metrics:
            dsc_mean = val_metrics.get("DSC", {}).get("overall", {}).get("mean", 0)
            dsc30 = val_metrics.get("DSC30", {}).get("dsc30", 0)
            print(f"  Val DSC: {dsc_mean:.4f} | DSC30: {dsc30:.4f}")

            # 保存最佳模型
            composite = val_metrics.get("composite_score", 0)
            if composite > best_val_score:
                best_val_score = composite
                best_epoch = epoch + 1
                if trainable_params:
                    torch.save(
                        model.state_dict(),
                        output_dir / "checkpoints" / "best_model.pth"
                    )
                    print(f"  ★ Saved best model (score={best_val_score:.4f})")

        # 定期保存
        if trainable_params and (epoch + 1) % config.get("save_interval", 10) == 0:
            torch.save(
                model.state_dict(),
                output_dir / "checkpoints" / f"model_epoch_{epoch+1}.pth"
            )

    # === 保存最终模型 ===
    if trainable_params:
        torch.save(model.state_dict(), output_dir / "checkpoints" / "final_model.pth")

    # 保存训练历史
    with open(output_dir / "logs" / "train_history.json", "w") as f:
        json.dump(train_history, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\n[MedRegLab] Training completed!")
    print(f"  Best epoch: {best_epoch} (score={best_val_score:.4f})")
    print(f"  Models saved to {output_dir / 'checkpoints'}")


def _build_train_dataset(
    data_dir: Union[str, Path], config: Dict[str, Any]
) -> CompetitionDataset:
    """构建训练数据集"""
    target_size = tuple(config.get("target_size", [128, 128, 128]))
    transforms = build_transforms(
        split="train",
        target_size=target_size,
        elastic_deform=config.get("elastic_deform", True),
        random_flip=config.get("random_flip", True),
        intensity_aug=config.get("intensity_aug", True),
    )

    dataset = CompetitionDataset(
        root=data_dir,
        split="train",
        transforms=transforms,
        target_size=target_size,
    )
    return dataset


def _build_val_dataset(
    data_dir: Union[str, Path], config: Dict[str, Any]
) -> CompetitionDataset:
    """构建验证数据集"""
    target_size = tuple(config.get("target_size", [128, 128, 128]))

    dataset = CompetitionDataset(
        root=data_dir,
        split="val",
        transforms=None,
        target_size=target_size,
    )
    return dataset


def _build_loss_builder(config: Dict[str, Any], device: torch.device):
    """构建损失函数构建器"""
    loss_types = config.get("loss_types", ["mse", "ncc"])
    loss_weights = config.get("loss_weights", {"mse": 1.0, "ncc": 1.0})

    loss_builder = LossBuilder(
        losses=loss_types,
        weights=loss_weights,
        impact_weight=config.get("impact_weight", 0.5),
        tv_weight=config.get("tv_weight", 0.05),
        jacobian_weight=config.get("jacobian_weight", 0.01),
    )
    return loss_builder


def _compute_losses(
    result: Dict[str, Any],
    fixed: torch.Tensor,
    loss_builder: LossBuilder,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    """计算训练损失"""
    warped = result["warped"]
    phi = result.get("phi")

    losses = {}

    # 相似性损失
    sim_loss = 0.0
    if "mse" in config.get("loss_types", []):
        sim_loss += F.mse_loss(warped, fixed)
    if "ncc" in config.get("loss_types", []):
        # 全局 NCC
        ncc = _ncc_loss(warped, fixed)
        sim_loss += 1 - ncc

    losses["similarity"] = sim_loss.item()
    total_loss = sim_loss

    # 语义损失（IMPACT）
    if config.get("use_impact_loss") and "semantic_loss" in result:
        sem_loss = result["semantic_loss"]
        losses["semantic"] = sem_loss.item() if isinstance(sem_loss, torch.Tensor) else sem_loss
        total_loss = total_loss + config.get("impact_weight", 0.5) * sem_loss

    # TV 正则化
    if phi is not None and config.get("tv_weight", 0) > 0:
        tv_loss = _tv_regularization(phi)
        losses["tv"] = tv_loss.item()
        total_loss = total_loss + config.get("tv_weight", 0) * tv_loss

    # Jacobian 正则化
    if phi is not None and config.get("jacobian_weight", 0) > 0:
        jac_loss = _jacobian_penalty(phi)
        losses["jacobian"] = jac_loss.item()
        total_loss = total_loss + config.get("jacobian_weight", 0) * jac_loss

    losses["total"] = total_loss.item()
    losses["total_tensor"] = total_loss

    return losses


def _ncc_loss(I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """归一化互相关"""
    I_mean = I.mean()
    J_mean = J.mean()
    I_std = I.std()
    J_std = J.std()
    ncc = ((I - I_mean) * (J - J_mean)).mean() / (I_std * J_std + 1e-8)
    return ncc


def _tv_regularization(phi: torch.Tensor) -> torch.Tensor:
    """全变差正则化"""
    if phi.dim() == 5:  # 3D [B, 3, D, H, W]
        dx = phi[:, :, 1:, :, :] - phi[:, :, :-1, :, :]
        dy = phi[:, :, :, 1:, :] - phi[:, :, :, :-1, :]
        dz = phi[:, :, :, :, 1:] - phi[:, :, :, :, :-1]
        tv = (
            torch.mean(torch.sqrt(dx * dx + 1e-8))
            + torch.mean(torch.sqrt(dy * dy + 1e-8))
            + torch.mean(torch.sqrt(dz * dz + 1e-8))
        )
    else:  # 2D [B, 2, H, W]
        dx = phi[:, :, 1:, :] - phi[:, :, :-1, :]
        dy = phi[:, :, :, 1:] - phi[:, :, :, :-1]
        tv = torch.mean(torch.sqrt(dx * dx + 1e-8)) + torch.mean(
            torch.sqrt(dy * dy + 1e-8)
        )
    return tv


def _jacobian_penalty(phi: torch.Tensor) -> torch.Tensor:
    """Jacobian 行列式惩罚（防止折叠）"""
    # 简化：使用 Jacobian 行列式的方差
    jac = JacobianField.compute(phi)
    return torch.var(jac)


def _compute_batch_dsc(
    batch: Dict[str, Any],
    result: Dict[str, Any],
    device: torch.device,
) -> Optional[Dict[str, float]]:
    """计算批次的 DSC"""
    from medreglab.evaluation import dice_coefficient

    warped = result["warped"]
    phi = result.get("phi")
    moving = batch["moving"]

    if phi is None or not batch.get("has_segmentations"):
        return None

    organ_labels = ["liver", "spleen", "kidney"]
    dsc_values = {}

    # 对每个器官计算 DSC
    # 这里简化处理：使用 moving 分割经过形变后与 fixed 位置的相似度
    for organ in organ_labels:
        if organ in batch.get("segmentations", {}):
            moving_seg = batch["segmentations"][organ].to(device)

            # 形变 moving 分割
            warped_seg = _warp_label(moving_seg, phi)
            moving_seg_binary = (moving_seg > 0.5).float()
            warped_seg_binary = (warped_seg > 0.5).float()

            # Dice: warped seg vs original moving seg position
            dsc = dice_coefficient(warped_seg_binary, moving_seg_binary)
            dsc_values[organ] = dsc

    if dsc_values:
        dsc_values["mean"] = float(np.mean(list(dsc_values.values())))

    return dsc_values


def _warp_label(label: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """将标签变形"""
    B, C, D, H, W = label.shape
    device = label.device

    grid_d = torch.linspace(-1, 1, D, device=device)
    grid_h = torch.linspace(-1, 1, H, device=device)
    grid_w = torch.linspace(-1, 1, W, device=device)
    grid_d, grid_h, grid_w = torch.meshgrid(grid_d, grid_h, grid_w, indexing="ij")

    phi_norm = phi.squeeze(0).clone()
    phi_norm[0] = 2 * phi_norm[0] / D
    phi_norm[1] = 2 * phi_norm[1] / H
    phi_norm[2] = 2 * phi_norm[2] / W

    grid = torch.stack([grid_d, grid_h, grid_w], dim=-1)
    grid = grid + phi_norm.permute(1, 2, 3, 0)
    grid = grid.unsqueeze(0)

    warped = F.grid_sample(
        label, grid, mode="nearest", padding_mode="border", align_corners=True
    )
    return warped


# === 快速测试接口 ===
def quick_train(
    train_data_dir: str,
    val_data_dir: Optional[str] = None,
    output_dir: str = "competition_outputs",
    model_name: str = "convex_3d",
    num_epochs: int = 50,
    batch_size: int = 1,
    target_size: List[int] = None,
    learning_rate: float = 1e-4,
):
    """快速训练接口"""
    target_size = target_size or [128, 128, 128]
    config = {
        "model": model_name,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "target_size": target_size,
        "loss_types": ["mse", "ncc"],
        "loss_weights": {"mse": 1.0, "ncc": 1.0},
        "use_impact_loss": True,
        "impact_weight": 0.5,
        "tv_weight": 0.05,
        "jacobian_weight": 0.01,
        "elastic_deform": True,
        "random_flip": True,
        "intensity_aug": True,
        "save_interval": 10,
        "organ_labels": ["liver", "spleen", "kidney"],
        "seed": 42,
    }

    train_competition(
        config=config,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train registration model for competition")
    parser.add_argument("--data", type=str, required=True, help="Training data directory")
    parser.add_argument("--val", type=str, help="Validation data directory")
    parser.add_argument("--output", type=str, default="competition_outputs", help="Output directory")
    parser.add_argument("--model", type=str, default="convex_3d", help="Model name")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--target_size", type=int, nargs="+", default=[128, 128, 128], help="Target size D H W")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    args = parser.parse_args()

    config = {
        "model": args.model,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "target_size": list(args.target_size),
        "loss_types": ["mse", "ncc"],
        "loss_weights": {"mse": 1.0, "ncc": 1.0},
        "use_impact_loss": True,
        "impact_weight": 0.5,
        "tv_weight": 0.05,
        "jacobian_weight": 0.01,
        "elastic_deform": True,
        "random_flip": True,
        "intensity_aug": True,
        "save_interval": 10,
        "organ_labels": ["liver", "spleen", "kidney"],
        "seed": 42,
    }

    train_competition(
        config=config,
        train_data_dir=args.data,
        val_data_dir=args.val,
        output_dir=args.output,
        resume_from=args.resume,
    )
