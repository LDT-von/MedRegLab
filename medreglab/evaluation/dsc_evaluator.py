"""
DSC & DSC30 Evaluator - 比赛官方评估指标实现

DSC  = Dice Similarity Coefficient
DSC30 = 最低 30% 病例的 DSC 平均值（鲁棒性评分）

计算公式：
    DSC = 2 * |A ∩ B| / (|A| + |B|)
    DSC30 = 所有测试病例中 DSC 值最低的 30% 病例的 DSC 平均值
"""

import torch
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Union, List, Dict, Optional, Any
import json


def dice_coefficient(
    seg_a: Union[np.ndarray, torch.Tensor],
    seg_b: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-5,
) -> float:
    """
    计算两组分割标签之间的 Dice 系数

    Args:
        seg_a: 固定 CT 的二值器官 mask
        seg_b: 已变形到 CT 空间的 MR 器官 mask
        smooth: 仅为兼容旧接口保留；比赛公式不加入平滑项

    Returns:
        float: Dice 系数 [0, 1]
    """
    if isinstance(seg_a, torch.Tensor):
        seg_a = seg_a.detach().cpu().numpy()
    if isinstance(seg_b, torch.Tensor):
        seg_b = seg_b.detach().cpu().numpy()

    seg_a = np.asarray(seg_a)
    seg_b = np.asarray(seg_b)
    if seg_a.shape != seg_b.shape:
        raise ValueError(f"Dice mask shapes differ: {seg_a.shape} vs {seg_b.shape}")
    if not np.isfinite(seg_a).all() or not np.isfinite(seg_b).all():
        raise ValueError("Dice masks contain NaN or Inf")
    fixed = seg_a > 0.5
    warped = seg_b > 0.5
    fixed_count = int(np.count_nonzero(fixed))
    # Competition rule: if the fixed CT organ is absent, the score is 1.0,
    # including when the warped MR contains a false-positive organ.
    if fixed_count == 0:
        return 1.0
    warped_count = int(np.count_nonzero(warped))
    return 2.0 * int(np.count_nonzero(fixed & warped)) / (fixed_count + warped_count)


def competition_case_dice(fixed_masks, warped_masks):
    """Calculate the exact competition per-organ and case mean Dice scores."""
    organs = ("liver", "spleen", "kidney")
    for organ in organs:
        if organ not in fixed_masks or organ not in warped_masks:
            raise ValueError(f"Both CT and warped MR masks are required for {organ}")
    scores = {organ: dice_coefficient(fixed_masks[organ], warped_masks[organ]) for organ in organs}
    scores["mean"] = sum(scores.values()) / len(organs)
    return scores


def dice_per_organ(
    fixed_seg: Dict[str, np.ndarray],
    warped_seg: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    分别计算每个器官的 Dice 系数

    Args:
        fixed_seg: 固定图像各器官分割标签 {"liver": array, "spleen": array, "kidney": array}
        warped_seg: 浮动图像分割标签经形变场映射后的各器官标签

    Returns:
        dict: 各器官的 Dice 值 {"liver": 0.85, "spleen": 0.78, "kidney": 0.72}
    """
    organ_dice = {}
    for organ in fixed_seg:
        if organ in warped_seg:
            organ_dice[organ] = dice_coefficient(fixed_seg[organ], warped_seg[organ])
        else:
            raise ValueError(f"Warped segmentation missing organ {organ!r}")
    return organ_dice


def compute_dsc_for_pair(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    deformation_field: sitk.Image,
    fixed_segmentations: Dict[str, str],
    organs: List[str] = None,
    moving_segmentations: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    对一对图像计算 DSC

    Args:
        fixed_image: 固定图像 (CT)
        moving_image: 浮动图像 (MR)
        deformation_field: 原始 CT 网格上的物理位移场（CT 点到 MR 点）
        fixed_segmentations: CT 各器官二值分割文件路径或 SimpleITK 图像
        moving_segmentations: MR 各器官二值分割文件路径或 SimpleITK 图像
        organs: 器官列表

    Returns:
        dict: {"liver": DSC, "spleen": DSC, "kidney": DSC, "mean": 平均DSC}
    """
    organs = organs or ["liver", "spleen", "kidney"]
    if set(organs) != {"liver", "spleen", "kidney"}:
        raise ValueError("Competition scoring requires liver, spleen and kidney")
    if moving_segmentations is None:
        raise ValueError("Both CT and MR segmentations are required for DSC")
    if deformation_field.GetNumberOfComponentsPerPixel() != 3:
        raise ValueError("Expected a three-component displacement field")

    def same_geometry(left, right):
        return left.GetSize() == right.GetSize() and all(
            np.allclose(getattr(left, name)(), getattr(right, name)(), rtol=0, atol=1e-5)
            for name in ("GetSpacing", "GetOrigin", "GetDirection")
        )

    if not same_geometry(deformation_field, fixed_image):
        raise ValueError("Displacement field must use the original CT geometry")
    transform = sitk.DisplacementFieldTransform(sitk.Cast(deformation_field, sitk.sitkVectorFloat64))
    fixed_masks, warped_masks = {}, {}
    for organ in organs:
        if organ not in fixed_segmentations or organ not in moving_segmentations:
            raise ValueError(f"Both CT and MR segmentations are required for {organ}")
        fixed_input, moving_input = fixed_segmentations[organ], moving_segmentations[organ]
        fixed_seg = fixed_input if isinstance(fixed_input, sitk.Image) else sitk.ReadImage(str(fixed_input))
        moving_seg = moving_input if isinstance(moving_input, sitk.Image) else sitk.ReadImage(str(moving_input))
        if not same_geometry(fixed_seg, fixed_image) or not same_geometry(moving_seg, moving_image):
            raise ValueError(f"Segmentation geometry mismatch for {organ}")
        warped_seg = sitk.Resample(
            moving_seg, fixed_image, transform, sitk.sitkNearestNeighbor, 0, moving_seg.GetPixelID()
        )
        fixed_masks[organ] = sitk.GetArrayFromImage(fixed_seg)
        warped_masks[organ] = sitk.GetArrayFromImage(warped_seg)
    return competition_case_dice(fixed_masks, warped_masks)


class DSC_Scorer:
    """
    DSC 评分器

    计算配准结果与固定图像分割标签的重叠度
    """

    def __init__(
        self,
        organs: List[str] = None,
        smooth: float = 1e-5,
    ):
        """
        Args:
            organs: 器官列表
            smooth: 平滑项
        """
        self.organs = organs or ["liver", "spleen", "kidney"]
        self.smooth = smooth
        self.reset()

    def reset(self):
        """重置评分器状态"""
        self.results = []

    def add_case(
        self,
        case_id: str,
        organ_dice: Dict[str, float],
    ):
        """
        添加一个病例的评分结果

        Args:
            case_id: 病例编号
            organ_dice: 各器官 DSC 值
        """
        if any(r["case_id"] == case_id for r in self.results):
            raise ValueError(f"Duplicate case ID: {case_id}")
        missing = set(self.organs) - set(organ_dice)
        if missing:
            raise ValueError(f"Missing organ scores: {sorted(missing)}")
        scores = {organ: float(organ_dice[organ]) for organ in self.organs}
        if not all(np.isfinite(score) and 0 <= score <= 1 for score in scores.values()):
            raise ValueError("Organ DSC values must be finite and between 0 and 1")
        mean_dsc = float(np.mean(list(scores.values())))
        if "mean" in organ_dice and not np.isclose(organ_dice["mean"], mean_dsc, rtol=0, atol=1e-8):
            raise ValueError("Case mean must equal the mean of the organ scores")
        record = {"case_id": case_id, **scores, "mean": mean_dsc}
        self.results.append(record)

    def compute(self) -> Dict[str, float]:
        """
        计算汇总 DSC 分数

        Returns:
            dict: {"liver": mean±std, "spleen": mean±std, "kidney": mean±std, "mean": mean±std}
        """
        if not self.results:
            return {}

        summary = {}
        for organ in self.organs:
            values = [r.get(organ, 0.0) for r in self.results]
            if values:
                summary[organ] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

        # 计算总体平均
        all_dsc = [r.get("mean", 0.0) for r in self.results if "mean" in r]
        if all_dsc:
            summary["overall"] = {
                "mean": float(np.mean(all_dsc)),
                "std": float(np.std(all_dsc)),
            }

        return summary

    def to_json(self, path: Optional[Path] = None) -> str:
        """导出评分结果为 JSON"""
        data = {
            "individual_results": self.results,
            "summary": self.compute(),
        }
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        if path:
            Path(path).write_text(json_str, encoding="utf-8")
        return json_str


class DSC30_Scorer:
    """
    DSC30 鲁棒性评分器

    DSC30 = 所有测试病例中 DSC 值最低的 30% 病例的 DSC 平均值

    该指标惩罚极端劣质的配准结果，衡量模型在最差情况下的鲁棒性
    """

    def __init__(self, percent: float = 0.3):
        """
        Args:
            percent: 取最低 DSC 病例的比例（默认 30%）
        """
        if not 0 < percent <= 1:
            raise ValueError("DSC30 percentage must be in (0, 1]")
        self.percent = percent
        self.case_scores = []

    def reset(self):
        self.case_scores = []

    def add_case(self, case_id: str, mean_dsc: float, organ_dice: Dict[str, float] = None):
        """
        添加一个病例的 DSC 分数

        Args:
            case_id: 病例编号
            mean_dsc: 该病例的平均 DSC（所有器官平均）
            organ_dice: 各器官详细 DSC
        """
        if any(item["case_id"] == case_id for item in self.case_scores):
            raise ValueError(f"Duplicate case ID: {case_id}")
        if not np.isfinite(mean_dsc) or not 0 <= mean_dsc <= 1:
            raise ValueError("Mean DSC must be finite and between 0 and 1")
        self.case_scores.append({
            "case_id": case_id,
            "mean_dsc": mean_dsc,
            "organ_dice": organ_dice or {},
        })

    def compute(self) -> Dict[str, Any]:
        """
        计算 DSC30

        计算步骤：
        1. 将所有测试病例按 DSC 值从低到高排序
        2. 取排名前 30%（向上取整）的病例
        3. 计算这些病例 DSC 值的算术平均数

        Returns:
            dict: {"dsc30": float, "worst_cases": [case_ids], "num_worst": int, "total": int}
        """
        if not self.case_scores:
            return {"dsc30": None, "worst_cases": [], "num_worst": 0, "total": 0, "worst_dsc_values": []}

        # 按 mean_dsc 从低到高排序
        sorted_scores = sorted(self.case_scores, key=lambda x: x["mean_dsc"])

        # 向上取整取 30%
        total = len(sorted_scores)
        num_worst = max(1, int(np.ceil(total * self.percent)))

        # 取最低的 30%
        worst_cases = sorted_scores[:num_worst]
        dsc_values = [c["mean_dsc"] for c in worst_cases]

        dsc30 = float(np.mean(dsc_values))

        return {
            "dsc30": dsc30,
            "worst_cases": [c["case_id"] for c in worst_cases],
            "worst_dsc_values": dsc_values,
            "num_worst": num_worst,
            "total": total,
        }

    def compute_per_organ(
        self, organ: str
    ) -> Dict[str, Any]:
        """
        计算某个器官的 DSC30

        Args:
            organ: 器官名称

        Returns:
            dict: 该器官的 DSC30 相关信息
        """
        organ_scores = []
        for c in self.case_scores:
            if organ in c.get("organ_dice", {}):
                organ_scores.append({
                    "case_id": c["case_id"],
                    "dsc": c["organ_dice"][organ],
                })

        if not organ_scores:
            return {f"{organ}_dsc30": None, f"{organ}_worst_cases": [],
                    f"{organ}_num_worst": 0, f"{organ}_total": 0}

        sorted_scores = sorted(organ_scores, key=lambda x: x["dsc"])
        total = len(sorted_scores)
        num_worst = int(np.ceil(total * self.percent))
        worst = sorted_scores[:num_worst]

        return {
            f"{organ}_dsc30": float(np.mean([c["dsc"] for c in worst])),
            f"{organ}_worst_cases": [c["case_id"] for c in worst],
            f"{organ}_num_worst": num_worst,
            f"{organ}_total": total,
        }


class CompetitionEvaluator:
    """
    比赛综合评估器

    整合 DSC 和 DSC30 计算，输出完整的评估报告
    """

    def __init__(
        self,
        organs: List[str] = None,
        dsc30_percent: float = 0.3,
        output_dir: Optional[Path] = None,
    ):
        """
        Args:
            organs: 器官列表
            dsc30_percent: DSC30 最低比例
            output_dir: 输出目录
        """
        self.organs = organs or ["liver", "spleen", "kidney"]
        self.dsc30_percent = dsc30_percent
        self.output_dir = Path(output_dir) if output_dir else None

        self.dsc_scorer = DSC_Scorer(organs=self.organs)
        self.dsc30_scorer = DSC30_Scorer(percent=self.dsc30_percent)
        self.detailed_results = []

    def reset(self):
        """重置评估器"""
        self.dsc_scorer.reset()
        self.dsc30_scorer.reset()
        self.detailed_results = []

    def add_case(
        self,
        case_id: str,
        organ_dice: Dict[str, float],
    ):
        """
        添加一个病例的评估结果

        Args:
            case_id: 病例编号
            organ_dice: 各器官 DSC 值
        """
        self.dsc_scorer.add_case(case_id, organ_dice)
        record = self.dsc_scorer.results[-1]
        mean_dsc = record["mean"]
        validated_dice = {organ: record[organ] for organ in self.organs}
        validated_dice["mean"] = mean_dsc
        self.dsc30_scorer.add_case(case_id, mean_dsc, validated_dice)

        self.detailed_results.append({
            "case_id": case_id,
            "dsc": validated_dice,
            "mean_dsc": mean_dsc,
        })

    def evaluate(
        self,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        执行完整评估

        Args:
            save: 是否保存结果

        Returns:
            dict: 完整评估报告
        """
        if not self.detailed_results:
            raise ValueError("No labeled cases available for DSC evaluation")
        dsc_summary = self.dsc_scorer.compute()
        dsc30_result = self.dsc30_scorer.compute()

        report = {
            "DSC": dsc_summary,
            "DSC30": dsc30_result,
            "per_case_results": self.detailed_results,
        }

        # Official ranking is lexicographic, not an invented weighted score.
        report["ranking_rule"] = {"primary": "DSC", "secondary": "DSC30"}

        if save and self.output_dir:
            self._save_report(report)

        return report

    def print_report(self, report: Dict[str, Any] = None):
        """打印评估报告"""
        if report is None:
            report = self.evaluate(save=False)

        print("\n" + "=" * 60)
        print("     跨模态医学图像配准挑战赛 - 评估报告")
        print("=" * 60)

        # DSC 结果
        print("\n【DSC（Dice Similarity Coefficient）】")
        if "DSC" in report:
            dsc = report["DSC"]
            for organ in self.organs:
                if organ in dsc:
                    vals = dsc[organ]
                    print(
                        f"  {organ:>10s}: {vals['mean']:.4f} ± {vals['std']:.4f}  "
                        f"(min={vals['min']:.4f}, max={vals['max']:.4f})"
                    )
            if "overall" in dsc:
                print(f"  {'OVERALL':>10s}: {dsc['overall']['mean']:.4f} ± {dsc['overall']['std']:.4f}")

        # DSC30 结果
        print("\n【DSC30（鲁棒性评分 - 最低30%病例平均）】")
        if "DSC30" in report:
            dsc30 = report["DSC30"]
            print(f"  DSC30 = {dsc30['dsc30']:.4f}")
            print(f"  最差病例数: {dsc30['num_worst']}/{dsc30['total']}")
            print(f"  最差病例: {dsc30['worst_cases']}")

        print("\n【排名规则】主指标 DSC，次指标 DSC30")

        print("=" * 60 + "\n")

    def _save_report(self, report: Dict[str, Any]):
        """保存评估报告"""
        if not self.output_dir:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 保存 JSON
        report_path = self.output_dir / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=lambda x: float(x) if isinstance(x, np.floating) else x)

        # 保存 CSV 格式的 per-case 结果
        try:
            import csv

            csv_path = self.output_dir / "per_case_results.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["case_id", "mean_dsc"] + [f"{o}_dsc" for o in self.organs]
                writer.writerow(header)
                for r in report.get("per_case_results", []):
                    row = [r["case_id"], r["mean_dsc"]] + [
                        r["dsc"].get(o, 0.0) for o in self.organs
                    ]
                    writer.writerow(row)
        except ImportError:
            pass

        print(f"[CompetitionEvaluator] Report saved to {self.output_dir}")


def submit_deformation_fields(
    deformation_fields: List[Union[str, Path, sitk.Image]],
    case_ids: List[str],
    output_dir: Union[str, Path],
    organ_labels: List[str] = None,
) -> Dict[str, Any]:
    """
    Deprecated: this legacy interface cannot verify CT geometry and its old
    flat output layout did not satisfy the competition ZIP specification.

    Args:
        deformation_fields: 变形场列表（与 case_ids 对应）
        case_ids: 病例编号列表
        output_dir: 输出目录
        organ_labels: 器官标签

    Returns:
        dict: 提交文件信息
    """
    raise RuntimeError(
        "The legacy flat submission format is unsupported. Use "
        "experiments/package_submission.py with the original CT data root "
        "to generate and verify the official competition ZIP."
    )
