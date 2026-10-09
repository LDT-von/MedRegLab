"""
Evaluation Script - 配准评估脚本
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional
import json
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from medreglab import load_model
from medreglab.data import RegistrationDataset
from medreglab.utils import RegistrationMetrics, RegistrationVisualizer


def evaluate_registration(
    config: Dict,
    data_dir: Path,
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    device: Optional[str] = None,
    **kwargs,
):
    """
    评估配准模型

    Args:
        config: 评估配置
        data_dir: 数据目录
        model_path: 模型权重路径（可选）
        output_dir: 输出目录
        device: 计算设备
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MedRegLab] Evaluating on {device}")

    # 加载模型
    model_name = config.get("model", "dino_reg_mrct")
    model = load_model(model_name)

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")

    model = model.to(device)
    model.eval()

    # 创建数据集
    dataset = RegistrationDataset(
        root=data_dir,
        mode=config.get("data_mode", "file_list"),
        normalize=True,
    )

    # 评估指标
    metrics_calculator = RegistrationMetrics()
    visualizer = RegistrationVisualizer()

    all_metrics = []
    results_summary = {
        "ncc": [],
        "mse": [],
        "ssim": [],
        "folding_ratio": [],
    }

    # 评估每个样本
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            sample = dataset[idx]
            moving = sample["moving"].unsqueeze(0).to(device)
            fixed = sample["fixed"].unsqueeze(0).to(device)

            # 配准
            result = model(moving, fixed)

            # 计算指标
            metrics = metrics_calculator.compute_all(
                moving=moving,
                fixed=fixed,
                warped=result["warped"],
            )

            # Jacobian 折叠检查
            if "phi" in result:
                from medreglab.models.deformation_field import JacobianField
                jac = JacobianField.compute(result["phi"].squeeze())
                jac_stats = JacobianField.check_folding(jac)
                metrics["folding_ratio"] = jac_stats["fold_ratio"]
            else:
                metrics["folding_ratio"] = 0.0

            all_metrics.append(metrics)
            results_summary["ncc"].append(metrics["ncc"])
            results_summary["mse"].append(metrics["mse"])
            results_summary["ssim"].append(metrics["ssim"])
            results_summary["folding_ratio"].append(metrics["folding_ratio"])

            # 保存可视化
            if config.get("save_vis", False) and idx < config.get("max_vis", 10):
                vis_dir = output_dir / f"vis_{idx:04d}"
                vis_dir.mkdir(exist_ok=True)

                visualizer.plot_comparison(
                    moving.squeeze(),
                    fixed.squeeze(),
                    result["warped"].squeeze(),
                    save_path=vis_dir / "comparison.png",
                    show=False,
                )

                if "phi" in result:
                    visualizer.plot_deformation_field(
                        result["phi"].squeeze(),
                        save_path=vis_dir / "deformation_field.png",
                        show=False,
                    )

    # 汇总结果
    summary = {
        "ncc": {
            "mean": float(np.mean(results_summary["ncc"])),
            "std": float(np.std(results_summary["ncc"])),
        },
        "mse": {
            "mean": float(np.mean(results_summary["mse"])),
            "std": float(np.std(results_summary["mse"])),
        },
        "ssim": {
            "mean": float(np.mean(results_summary["ssim"])),
            "std": float(np.std(results_summary["ssim"])),
        },
        "folding_ratio": {
            "mean": float(np.mean(results_summary["folding_ratio"])),
            "std": float(np.std(results_summary["folding_ratio"])),
        },
        "num_samples": len(dataset),
    }

    # 保存结果
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"NCC:      {summary['ncc']['mean']:.4f} ± {summary['ncc']['std']:.4f}")
    print(f"MSE:      {summary['mse']['mean']:.4f} ± {summary['mse']['std']:.4f}")
    print(f"SSIM:     {summary['ssim']['mean']:.4f} ± {summary['ssim']['std']:.4f}")
    print(f"Folding:  {summary['folding_ratio']['mean']:.4f} ± {summary['folding_ratio']['std']:.4f}")
    print("=" * 50)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate registration model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--data", type=str, required=True, help="Data directory")
    parser.add_argument("--model", type=str, help="Model checkpoint path")
    parser.add_argument("--output", type=str, default="evaluation_results", help="Output directory")

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    evaluate_registration(
        config=config,
        data_dir=args.data,
        model_path=args.model,
        output_dir=args.output,
    )
