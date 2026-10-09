"""
Training Script - 配准训练脚本
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict
import json
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from medreglab import load_model
from medreglab.data import RegistrationDataset, MedicalAugmentation
from medreglab.utils import RegistrationMetrics, RegistrationVisualizer
from medreglab.core import LossBuilder


def train_registration(
    config: Dict,
    train_data_dir: Path,
    val_data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    device: Optional[str] = None,
    **kwargs,
):
    """
    训练配准模型

    Args:
        config: 训练配置
        train_data_dir: 训练数据目录
        val_data_dir: 验证数据目录（可选）
        output_dir: 输出目录
        device: 计算设备
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir) if output_dir else Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MedRegLab] Training on {device}")
    print(f"[MedRegLab] Config: {json.dumps(config, indent=2)}")

    # 加载模型
    model_name = config.get("model", "dino_reg_mrct")
    model = load_model(model_name)
    model = model.to(device)

    # 创建数据集
    train_dataset = RegistrationDataset(
        root=train_data_dir,
        mode=config.get("data_mode", "file_list"),
        normalize=True,
        target_size=config.get("target_size"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 1),
        shuffle=True,
        num_workers=config.get("num_workers", 0),
    )

    # 验证数据集
    val_loader = None
    if val_data_dir:
        val_dataset = RegistrationDataset(
            root=val_data_dir,
            mode=config.get("data_mode", "file_list"),
            normalize=True,
            target_size=config.get("target_size"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
    )

    # 损失函数
    loss_builder = LossBuilder(
        losses=config.get("losses", ["mse", "ncc"]),
        weights=config.get("loss_weights", {"mse": 1.0, "ncc": 1.0}),
    )

    # 训练循环
    num_epochs = config.get("num_epochs", 100)
    metrics_calculator = RegistrationMetrics()
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            moving = batch["moving"].to(device)
            fixed = batch["fixed"].to(device)

            # 前向传播
            result = model(moving, fixed)

            # 计算损失
            losses = loss_builder(
                result["warped"],
                fixed,
                phi=result.get("phi"),
            )
            loss = losses["total_tensor"]

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

        # 验证
        if val_loader:
            model.eval()
            val_losses = []

            with torch.no_grad():
                for batch in val_loader:
                    moving = batch["moving"].to(device)
                    fixed = batch["fixed"].to(device)

                    result = model(moving, fixed)
                    losses = loss_builder(
                        result["warped"],
                        fixed,
                        phi=result.get("phi"),
                    )

                    val_losses.append(losses["total_tensor"].item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}")

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), output_dir / "best_model.pth")
                print(f"Saved best model (val_loss={avg_val_loss:.4f})")

        # 定期保存
        if (epoch + 1) % config.get("save_interval", 10) == 0:
            torch.save(
                model.state_dict(),
                output_dir / f"model_epoch_{epoch+1}.pth"
            )

    # 保存最终模型
    torch.save(model.state_dict(), output_dir / "final_model.pth")
    print(f"[MedRegLab] Training completed. Models saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train registration model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--data", type=str, required=True, help="Training data directory")
    parser.add_argument("--val", type=str, help="Validation data directory")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()

    # 加载配置
    with open(args.config) as f:
        config = json.load(f)

    train_registration(
        config=config,
        train_data_dir=args.data,
        val_data_dir=args.val,
        output_dir=args.output,
    )
