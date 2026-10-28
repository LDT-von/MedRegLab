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
import hashlib
import argparse
import sys
from tqdm import tqdm
import warnings

sys.path.append(str(Path(__file__).parent.parent))

from medreglab import load_model
from medreglab.data import CompetitionDataset
from medreglab.evaluation import CompetitionEvaluator
from medreglab.utils.submission import (
    displacement_to_fixed_grid, warp_original_moving, grid_folding_statistics,
)


def _sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if split not in ("test", "val", "validation"):
        raise ValueError(f"Unsupported split: {split!r}")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if target_size is not None and (len(target_size) != 3 or any(n < 2 for n in target_size)):
        raise ValueError("target_size must contain three dimensions of at least two voxels")
    if model_path is not None:
        raise ValueError("convex_3d has no learned checkpoint; omit --checkpoint")
    if save_vis:
        raise NotImplementedError("save_vis is not implemented")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Refusing to overwrite a nonempty output directory: {output_dir}")

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
    model.max_iter = max_iter
    model = model.to(device)
    model.eval()

    # A submission must never silently fall back to training data.
    if split == "test" and not (Path(data_root) / "test").is_dir():
        raise FileNotFoundError(f"Missing test directory: {Path(data_root) / 'test'}")
    # 创建数据集
    dataset = CompetitionDataset(
        root=data_root,
        split=split,
        target_size=target_size,
    )
    print(f"[MedRegLab] Loaded {len(dataset)} samples")
    if not len(dataset):
        raise ValueError("No cases found; refusing to create an empty submission")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建评估器（仅 split=="val" 时计算 DSC）
    evaluator = None
    if split in ("val", "validation"):
        evaluator = CompetitionEvaluator(
            organs=report_organs,
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
            sample_info = dataset.get_sample_info(idx)
            case_id = sample_info.get("case_id", f"case_{idx:04d}")
            try:
                input_hashes = {
                    "ct_input_sha256": _sha256_file(sample_info["fixed_path"]),
                    "mr_input_sha256": _sha256_file(sample_info["moving_path"]),
                }
                sample = dataset[idx]
                if sample.get("spatial_augmented"):
                    raise ValueError("Spatially augmented tensors cannot be exported using original image geometry")
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
                phi = result["phi"].detach().cpu()

                # === 生成提交文件 ===
                deform_img = displacement_to_fixed_grid(
                    phi, sample["fixed_sitk"], sample["moving_sitk"]
                )

                # 保存变形场
                deform_path = deform_dir / f"{case_id}_deformation_field.nii.gz"
                sitk.WriteImage(deform_img, str(deform_path))

                # 保存配准后的图像
                if save_warped:
                    warped_img = warp_original_moving(
                        sample["moving_sitk"], sample["fixed_sitk"], deform_img
                    )
                    sitk.WriteImage(warped_img, str(warped_dir / f"{case_id}_warped.nii.gz"))

                if input_hashes != {
                    "ct_input_sha256": _sha256_file(sample_info["fixed_path"]),
                    "mr_input_sha256": _sha256_file(sample_info["moving_path"]),
                }:
                    raise RuntimeError("CT or MR input changed during registration")
                case_result = {
                    "case_id": case_id,
                    "loss": float(result["loss"]),
                    "iterations": len(result["loss_history"]),
                    "fixed_size_xyz": list(sample["fixed_sitk"].GetSize()),
                    "moving_size_xyz": list(sample["moving_sitk"].GetSize()),
                    **input_hashes,
                    "field_sha256": _sha256_file(deform_path),
                }
                # === 计算 DSC（仅验证集有标签）===
                if evaluator is not None:
                    dsc_result = _compute_dsc_for_sample(
                        sample, deform_img, sample_info.get("seg_fixed_path"),
                        sample_info.get("seg_moving_path"), report_organs,
                    )
                    evaluator.add_case(case_id, dsc_result)
                    case_result.update({"mean_dsc": dsc_result["mean"], "dsc": dsc_result})

                # Jacobian 折叠检查
                jac_result = grid_folding_statistics(phi)
                case_result.update(jac_result)
                results.append(case_result)
                print(f"  {case_id}: saved native-grid field, loss={result['loss']:.6f}", flush=True)

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
        "max_iter": max_iter,
        "target_size_dhw": list(target_size) if target_size else None,
        "field_grid": "original fixed CT",
        "field_units": "mm",
        "field_components": "physical LPS x,y,z",
        "transform": "pull: fixed CT physical point -> moving MR physical point",
        "cases": results,
    }

    if evaluator is not None and not errors:
        eval_report = evaluator.evaluate(save=True)
        evaluator.print_report(eval_report)
        summary["evaluation"] = eval_report
    elif evaluator is not None:
        summary["evaluation"] = None
        summary["evaluation_status"] = "incomplete: one or more validation cases failed"

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
    deformation_field: sitk.Image,
    fixed_seg_path: Optional[str],
    moving_seg_path: Optional[str],
    organ_labels: List[str],
) -> Dict[str, float]:
    """
    计算单个样本的 DSC（按比赛官方定义）

    比赛公式：DSC = 2 * |A ∩ B| / (|A| + |B|)
    - A = 固定图像 (CT) 中某一器官的分割标签体素集合
    - B = 配准后浮动图像 (MR) 分割标签经形变场映射至 CT 空间后的体素集合

    使用与正式提交完全相同的物理位移场，直接把原始 MR 标签最近邻重采样到原始 CT 网格。

    Args:
        sample: 数据集样本
        deformation_field: CT 网格上的 LPS 毫米位移场
        organ_labels: 器官标签列表

    Returns:
        各器官 DSC 值字典
    """
    from medreglab.data import competition_masks_from_labels
    from medreglab.evaluation import competition_case_dice

    if set(organ_labels) != {"liver", "spleen", "kidney"}:
        raise ValueError("Competition DSC requires liver, spleen, and combined kidney")
    if not fixed_seg_path or not moving_seg_path:
        raise ValueError("Validation requires both original CT and MR segmentation files")
    fixed_seg = sitk.ReadImage(str(fixed_seg_path))
    moving_seg = sitk.ReadImage(str(moving_seg_path))
    for label, image, modality in (
        (fixed_seg, sample["fixed_sitk"], "CT"),
        (moving_seg, sample["moving_sitk"], "MR"),
    ):
        if label.GetSize() != image.GetSize() or any(
            not np.allclose(getattr(label, name)(), getattr(image, name)(), rtol=0, atol=1e-5)
            for name in ("GetSpacing", "GetOrigin", "GetDirection")
        ):
            raise ValueError(f"{modality} segmentation geometry does not match its image")
    transform = sitk.DisplacementFieldTransform(sitk.Cast(deformation_field, sitk.sitkVectorFloat64))
    warped_seg = sitk.Resample(
        moving_seg, sample["fixed_sitk"], transform,
        sitk.sitkNearestNeighbor, 0, moving_seg.GetPixelID(),
    )
    case_id = sample["case_id"]
    fixed_masks = competition_masks_from_labels(sitk.GetArrayViewFromImage(fixed_seg), case_id)
    warped_masks = competition_masks_from_labels(sitk.GetArrayViewFromImage(warped_seg), case_id)
    return competition_case_dice(fixed_masks, warped_masks)


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
    if phi.ndim != 5 or phi.shape[1] != 3 or label.ndim != 5:
        raise ValueError("Expected label [B, C, D, H, W] and phi [B, 3, D, H, W]")
    if phi.shape[0] != label.shape[0] or phi.shape[2:] != label.shape[2:]:
        raise ValueError("Label and displacement must share batch and spatial dimensions")
    _, _, depth, height, width = phi.shape
    if min(depth, height, width) < 2:
        raise ValueError("All spatial dimensions must be at least two")
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, depth, device=phi.device, dtype=phi.dtype),
        torch.linspace(-1, 1, height, device=phi.device, dtype=phi.dtype),
        torch.linspace(-1, 1, width, device=phi.device, dtype=phi.dtype),
        indexing="ij",
    )
    grid = torch.stack((x, y, z), dim=-1).unsqueeze(0)
    scale = phi.new_tensor((2 / (width - 1), 2 / (height - 1), 2 / (depth - 1)))
    grid = grid + phi.permute(0, 2, 3, 4, 1) * scale
    return F.grid_sample(label, grid, mode=mode, padding_mode="zeros", align_corners=True)


def generate_submission(
    output_dir: Union[str, Path],
    submission_dir: Union[str, Path],
    organ_labels: List[str] = None,
    *,
    data_root: Optional[Union[str, Path]] = None,
):
    """
    生成提交文件

    Args:
        output_dir: 推理结果目录
        submission_dir: 提交文件目录
        organ_labels: 器官标签
    """
    if data_root is None:
        raise ValueError("data_root is required to verify the original CT geometry")
    if organ_labels is not None:
        raise ValueError("organ_labels do not apply to competition ZIP packaging")
    from experiments.package_submission import create_submission

    return create_submission(output_dir, data_root, submission_dir)


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
    if summary["num_errors"]:
        raise SystemExit(1)
