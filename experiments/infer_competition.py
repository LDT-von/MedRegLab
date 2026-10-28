"""
Competition Inference Script - 比赛推理与提交生成脚本

功能：
1. 对测试集图像进行配准推理
2. 生成变形场 (deformation field) 提交文件
3. 计算验证集 DSC / DSC30 分数
"""

import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
import json
import argparse
import sys
from tqdm import tqdm
import warnings

sys.path.append(str(Path(__file__).parent.parent))

from medreglab import load_model
from medreglab.data import CompetitionDataset
from medreglab.evaluation import CompetitionEvaluator, dice_coefficient
from medreglab.utils import load_image, save_image
from medreglab.models.deformation_field import JacobianField


def run_competition_inference(
    data_root: Union[str, Path],
    output_dir: Union[str, Path],
    model_name: str = "convex_3d",
    model_path: Optional[Path] = None,
    split: str = "test",
    device: Optional[str] = None,
    target_size: Optional[Tuple[int, ...]] = None,
    max_iter: int = 200,
    save_warped: bool = False,
    save_vis: bool = False,
    organ_labels: List[str] = None,
) -> Dict[str, Any]:
    """
    运行比赛推理

    Args:
        data_root: 数据根目录
        output_dir: 输出目录
        model_name: 模型名称
        model_path: 模型权重路径
        split: 数据集划分 ("test" 或 "val")
        device: 计算设备
        target_size: 目标尺寸
        max_iter: 配准最大迭代次数
        save_warped: 是否保存配准后的图像
        save_vis: 是否保存可视化
        organ_labels: 器官标签

    Returns:
        dict: 推理结果统计
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from medreglab.data import ORGANS_FOR_DSC
    organ_labels = organ_labels or ORGANS_FOR_DSC
    # 将 left/right kidney 合并为 "kidney" 用于报告
    report_organs = []
    for o in organ_labels:
        if o in ("kidney_left", "kidney_right"):
            if "kidney" not in report_organs:
                report_organs.append("kidney")
        else:
            report_organs.append(o)
    report_organs = list(dict.fromkeys(report_organs))  # 去重保持顺序

    print(f"[MedRegLab] Competition Inference on {device}")
    print(f"[MedRegLab] Data root: {data_root}")
    print(f"[MedRegLab] Output: {output_dir}")
    print(f"[MedRegLab] Split: {split}")
    print(f"[MedRegLab] Model: {model_name}")

    # 加载模型
    model = load_model(model_name)
    if model_path and Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    model = model.to(device)
    model.eval()

    # 创建数据集
    dataset = CompetitionDataset(
        root=data_root,
        split=split,
        target_size=target_size,
    )
    print(f"[MedRegLab] Loaded {len(dataset)} samples")

    # 创建评估器（仅 split=="val" 时计算 DSC）
    evaluator = None
    if split in ("val", "validation"):
        evaluator = CompetitionEvaluator(
            organs=organ_labels,
            output_dir=output_dir / "eval_results",
        )

    # 变形场输出目录
    deform_dir = output_dir / "deformation_fields"
    deform_dir.mkdir(exist_ok=True)

    if save_warped:
        warped_dir = output_dir / "warped_images"
        warped_dir.mkdir(exist_ok=True)

    results = []
    errors = []

    with torch.enable_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Registering ({split})"):
            sample = dataset[idx]
            case_id = sample.get("case_id", f"case_{idx:04d}")

            try:
                # 准备输入 (sample 已是 5D [1, C, D, H, W])
                fixed = sample["fixed"].to(device)
                moving = sample["moving"].to(device)
                if fixed.dim() == 6:
                    fixed = fixed.squeeze(0)
                if moving.dim() == 6:
                    moving = moving.squeeze(0)

                # 推理
                result = model(moving, fixed)

                # 获取变形场
                phi = result["phi"].cpu()
                warped = result["warped"].cpu()

                # === 生成提交文件 ===
                # 变形场格式: [D, H, W] displacement vectors
                phi_np = phi.squeeze().numpy()  # [3, D, H, W] 或 [2, H, W]

                # 转换形状: displacement field 应该是 [D, H, W, 3]
                if phi_np.ndim == 4:  # 3D [3, D, H, W]
                    phi_submit = np.transpose(phi_np, (1, 2, 3, 0))  # [D, H, W, 3]
                else:  # 2D [2, H, W]
                    phi_submit = np.transpose(phi_np, (1, 2, 0))  # [H, W, 2]

                # 创建 SimpleITK 位移场
                deform_img = sitk.GetImageFromArray(phi_submit, isVector=True)
                deform_img.SetSpacing(sample["fixed_sitk"].GetSpacing())
                deform_img.SetOrigin(sample["fixed_sitk"].GetOrigin())
                deform_img.SetDirection(sample["fixed_sitk"].GetDirection())

                # 保存变形场
                deform_path = deform_dir / f"{case_id}_deformation_field.nii.gz"
                sitk.WriteImage(deform_img, str(deform_path))

                # 保存配准后的图像
                if save_warped:
                    warped_np = warped.squeeze().numpy()
                    warped_img = sitk.GetImageFromArray(warped_np)
                    # 不要用 CopyInformation（会校验尺寸），手动设置原图元信息
                    warped_img.SetSpacing(sample["moving_sitk"].GetSpacing())
                    warped_img.SetOrigin(sample["moving_sitk"].GetOrigin())
                    try:
                        warped_img.SetDirection(sample["moving_sitk"].GetDirection())
                    except Exception:
                        pass
                    sitk.WriteImage(warped_img, str(warped_dir / f"{case_id}_warped.nii.gz"))

                # === 计算 DSC（仅验证集有标签）===
                if evaluator is not None and sample.get("has_segmentations"):
                    dsc_result = _compute_dsc_for_sample(
                        sample, phi, organ_labels
                    )
                    if dsc_result:
                        evaluator.add_case(case_id, dsc_result)
                        results.append({
                            "case_id": case_id,
                            "mean_dsc": dsc_result.get("mean", 0),
                            "dsc": dsc_result,
                        })

                # Jacobian 折叠检查
                jac_result = JacobianField.check_folding(phi.squeeze())
                results.append({
                    "case_id": case_id,
                    "fold_ratio": jac_result["fold_ratio"],
                    "min_jacobian": jac_result["min_jacobian"],
                })

            except Exception as e:
                errors.append({"case_id": case_id, "error": str(e)})
                warnings.warn(f"Error processing {case_id}: {e}")

    # === 保存结果 ===
    summary = {
        "model": model_name,
        "split": split,
        "num_processed": len(dataset) - len(errors),
        "num_errors": len(errors),
        "deform_dir": str(deform_dir),
    }

    if evaluator is not None:
        eval_report = evaluator.evaluate(save=True)
        evaluator.print_report(eval_report)
        summary["evaluation"] = eval_report

    # 保存汇总
    with open(output_dir / "inference_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    if errors:
        with open(output_dir / "errors.json", "w") as f:
            json.dump(errors, f, indent=2)

    print(f"\n[MedRegLab] Inference complete. Results in {output_dir}")
    print(f"  Processed: {summary['num_processed']}, Errors: {summary['num_errors']}")

    return summary


def _compute_dsc_for_sample(
    sample: Dict[str, Any],
    phi: torch.Tensor,
    organ_labels: List[str],
) -> Optional[Dict[str, float]]:
    """
    计算单个样本的 DSC（按比赛官方定义）

    比赛公式：DSC = 2 * |A ∩ B| / (|A| + |B|)
    - A = 固定图像 (CT) 中某一器官的分割标签体素集合
    - B = 配准后浮动图像 (MR) 分割标签经形变场映射至 CT 空间后的体素集合

    即：把 moving 分割用 phi 形变到 fixed 空间，再与 fixed 分割对比。

    Args:
        sample: 数据集样本
        phi: 变形场 (B, 3, D, H, W)
        organ_labels: 器官标签列表

    Returns:
        各器官 DSC 值字典
    """
    seg_moving = sample.get("seg_moving")
    seg_fixed = sample.get("seg_fixed")
    if seg_moving is None or seg_fixed is None:
        return None

    dsc_values = {}
    for organ in organ_labels:
        # 把 "kidney" 合并 left/right
        if organ == "kidney":
            sub_organs = ["kidney_left", "kidney_right"]
        else:
            sub_organs = [organ]

        # 形变 moving 的器官 mask（合并子器官）
        warped_combined = None
        fixed_combined = None

        for sub in sub_organs:
            if sub not in seg_moving or sub not in seg_fixed:
                continue

            moving_mask = seg_moving[sub].unsqueeze(0).to(phi.device).float()  # [1, 1, D, H, W]
            D, H, W = moving_mask.shape[2:]
            if phi.shape[2:] != (D, H, W):
                import torch.nn.functional as F
                phi_resized = F.interpolate(phi, size=(D, H, W), mode="trilinear", align_corners=True)
            else:
                phi_resized = phi

            warped = _apply_deformation_to_label(moving_mask, phi_resized)
            warped = (warped > 0.5).float()

            if warped_combined is None:
                warped_combined = warped
                fixed_combined = (seg_fixed[sub].unsqueeze(0).to(phi.device).float() > 0.5).float()
            else:
                warped_combined = warped_combined + warped
                fixed_combined = fixed_combined + (seg_fixed[sub].unsqueeze(0).to(phi.device).float() > 0.5).float()

        if warped_combined is not None:
            warped_binary = (warped_combined > 0.5).float()
            fixed_binary = (fixed_combined > 0.5).float()

            intersection = (warped_binary * fixed_binary).sum().item()
            sum_a = fixed_binary.sum().item()
            sum_b = warped_binary.sum().item()
            if sum_a + sum_b < 1e-6:
                dsc_values[organ] = 0.0
            else:
                dsc_values[organ] = 2.0 * intersection / (sum_a + sum_b)

    if dsc_values:
        dsc_values["mean"] = float(np.mean(list(dsc_values.values())))

    return dsc_values


def _apply_deformation_to_label(
    label: torch.Tensor,
    phi: torch.Tensor,
    mode: str = "nearest",
) -> torch.Tensor:
    """
    将分割标签变形到目标空间

    Args:
        label: 分割标签张量 [1, 1, D, H, W]
        phi: 变形场 [1, 3, D, H, W] displacement
        mode: 插值模式

    Returns:
        变形后的标签
    """
    # phi: displacement field [1, 3, D, H, W]
    # 需要转换为 grid format for grid_sample

    B, C, D, H, W = phi.shape
    device = phi.device

    # 创建标准化网格
    grid_d = torch.linspace(-1, 1, D, device=device)
    grid_h = torch.linspace(-1, 1, H, device=device)
    grid_w = torch.linspace(-1, 1, W, device=device)
    grid_d, grid_h, grid_w = torch.meshgrid(grid_d, grid_h, grid_w, indexing="ij")

    # 添加位移
    phi_norm = phi.squeeze(0).clone()  # [3, D, H, W]
    phi_norm[0] = 2 * phi_norm[0] / D  # 归一化到 [-1, 1]
    phi_norm[1] = 2 * phi_norm[1] / H
    phi_norm[2] = 2 * phi_norm[2] / W

    # grid: [D, H, W, 3], 每像素的目标坐标
    grid = torch.stack([grid_d, grid_h, grid_w], dim=-1)  # [D, H, W, 3]
    grid = grid + phi_norm.permute(1, 2, 3, 0)  # [D, H, W, 3]
    grid = grid.unsqueeze(0)  # [1, D, H, W, 3]

    # 重塑 label
    label_flat = label.view(1, 1, D, H, W)

    # 采样
    warped = F.grid_sample(
        label_flat,
        grid,
        mode=mode,
        padding_mode="border",
        align_corners=True,
    )

    return warped


def generate_submission(
    output_dir: Union[str, Path],
    submission_dir: Union[str, Path],
    organ_labels: List[str] = None,
):
    """
    生成提交文件

    Args:
        output_dir: 推理结果目录
        submission_dir: 提交文件目录
        organ_labels: 器官标签
    """
    output_dir = Path(output_dir)
    submission_dir = Path(submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)

    deform_dir = output_dir / "deformation_fields"
    if not deform_dir.exists():
        print(f"[ERROR] Deformation fields not found: {deform_dir}")
        return

    organ_labels = organ_labels or ["liver", "spleen", "kidney"]

    # 收集所有变形场文件
    deform_files = sorted(deform_dir.glob("*_deformation_field.nii.gz"))
    print(f"[MedRegLab] Found {len(deform_files)} deformation fields")

    # 生成提交信息
    submission_info = {
        "submission_format": "deformation_field",
        "version": "1.0",
        "organ_labels": organ_labels,
        "num_cases": len(deform_files),
        "files": [],
    }

    for deform_file in deform_files:
        case_id = deform_file.stem.replace("_deformation_field", "")
        dst = submission_dir / deform_file.name
        import shutil

        shutil.copy2(deform_file, dst)
        submission_info["files"].append({
            "case_id": case_id,
            "filename": deform_file.name,
        })

    # 保存提交信息
    with open(submission_dir / "submission_info.json", "w", encoding="utf-8") as f:
        json.dump(submission_info, f, indent=2, ensure_ascii=False)

    print(f"[MedRegLab] Submission generated: {submission_dir}")
    print(f"  Total cases: {len(deform_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run competition inference")
    parser.add_argument(
        "--data", type=str, required=True, help="Data root directory"
    )
    parser.add_argument(
        "--output", type=str, default="competition_output", help="Output directory"
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "val", "validation"],
        help="Data split (test for submission, val for evaluation)"
    )
    parser.add_argument(
        "--model", type=str, default="convex_3d",
        help="Model name (convex_3d)"
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Model checkpoint path"
    )
    parser.add_argument(
        "--target_size", type=int, nargs="+", default=[128, 128, 128],
        help="Target size (D H W)"
    )
    parser.add_argument(
        "--max_iter", type=int, default=200, help="Registration max iterations"
    )
    parser.add_argument(
        "--save_warped", action="store_true", help="Save warped images"
    )

    args = parser.parse_args()

    target_size = tuple(args.target_size) if args.target_size else None

    summary = run_competition_inference(
        data_root=args.data,
        output_dir=args.output,
        model_name=args.model,
        model_path=args.checkpoint,
        split=args.split,
        target_size=target_size,
        max_iter=args.max_iter,
        save_warped=args.save_warped,
    )
