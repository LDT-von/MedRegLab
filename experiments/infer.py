"""
Inference Script - 推理脚本

用于对新图像进行配准推理
"""

import torch
from pathlib import Path
from typing import Union, Optional, Dict, Tuple
import json
import argparse
import sys

sys.path.append(str(Path(__file__).parent.parent))

from medreglab import load_model
from medreglab.utils import load_image, save_image, RegistrationVisualizer


def run_inference(
    moving_path: Union[str, Path],
    fixed_path: Union[str, Path],
    model_name: str = "dino_reg_mrct",
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    device: Optional[str] = None,
    save_visualization: bool = True,
    **kwargs,
) -> Dict:
    """
    对一对图像进行配准推理

    Args:
        moving_path: 移动图像路径
        fixed_path: 固定图像路径
        model_name: 模型名称
        model_path: 模型权重路径（可选）
        output_dir: 输出目录
        device: 计算设备
        save_visualization: 是否保存可视化

    Returns:
        dict: 配准结果
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir) if output_dir else Path("inference_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MedRegLab] Running inference on {device}")
    print(f"Moving: {moving_path}")
    print(f"Fixed: {fixed_path}")

    # 加载模型
    model = load_model(model_name)

    if model_path and Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")

    model = model.to(device)
    model.eval()

    # 加载图像
    moving = load_image(moving_path)
    fixed = load_image(fixed_path)

    # 推理
    with torch.no_grad():
        result = model(moving.to(device), fixed.to(device))

    # 保存结果
    warped = result["warped"].cpu()
    save_image(warped, output_dir / "warped.nii.gz")

    if "phi" in result:
        save_image(result["phi"].cpu(), output_dir / "deformation_field.nii.gz")
    if "phi_inv" in result:
        save_image(result["phi_inv"].cpu(), output_dir / "deformation_field_inv.nii.gz")

    print(f"Results saved to {output_dir}")

    # 可视化
    if save_visualization:
        visualizer = RegistrationVisualizer()
        visualizer.plot_comparison(
            moving.squeeze(),
            fixed.squeeze(),
            warped.squeeze(),
            save_path=output_dir / "comparison.png",
            show=False,
        )

        if "phi" in result:
            visualizer.plot_deformation_field(
                result["phi"].squeeze().cpu(),
                save_path=output_dir / "deformation_field.png",
                show=False,
            )

        if "loss_history" in result:
            visualizer.plot_loss_curve(
                result["loss_history"],
                save_path=output_dir / "loss_curve.png",
                show=False,
            )

    return result


def run_batch_inference(
    data_pairs: list,
    model_name: str = "dino_reg_mrct",
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    device: Optional[str] = None,
) -> list:
    """
    批量推理

    Args:
        data_pairs: [(moving_path, fixed_path), ...] 列表
        model_name: 模型名称
        model_path: 模型权重路径
        output_dir: 输出目录
        device: 计算设备

    Returns:
        list: 所有配准结果
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir) if output_dir else Path("batch_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, (moving_path, fixed_path) in enumerate(data_pairs):
        pair_output = output_dir / f"pair_{idx:04d}"
        pair_output.mkdir(exist_ok=True)

        result = run_inference(
            moving_path=moving_path,
            fixed_path=fixed_path,
            model_name=model_name,
            model_path=model_path,
            output_dir=pair_output,
            device=device,
            save_visualization=True,
        )

        results.append(result)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run registration inference")
    parser.add_argument("--moving", type=str, required=True, help="Moving image path")
    parser.add_argument("--fixed", type=str, required=True, help="Fixed image path")
    parser.add_argument("--model", type=str, default="dino_reg_mrct", help="Model name")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--output", type=str, default="inference_results", help="Output directory")

    args = parser.parse_args()

    run_inference(
        moving_path=args.moving,
        fixed_path=args.fixed,
        model_name=args.model,
        model_path=args.checkpoint,
        output_dir=args.output,
    )
