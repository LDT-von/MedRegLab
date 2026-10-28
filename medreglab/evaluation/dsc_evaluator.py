"""
DSC & DSC30 Evaluator - 比赛官方评估指标实现

DSC  = Dice Similarity Coefficient
DSC30 = 最低 30% 病例的 DSC 平均值（鲁棒性评分）

计算公式：
    DSC = 2 * |A ∩ B| / (|A| + |B|)
    DSC30 = 所有测试病例中 DSC 值最低的 30% 病例的 DSC 平均值
"""

import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional, Any
import json
import warnings


def dice_coefficient(
    seg_a: Union[np.ndarray, torch.Tensor],
    seg_b: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-5,
) -> float:
    """
    计算两组分割标签之间的 Dice 系数

    Args:
        seg_a: 分割标签 A（体素集合）
        seg_b: 分割标签 B（体素集合）
        smooth: 平滑项防止除零

    Returns:
        float: Dice 系数 [0, 1]
    """
    if isinstance(seg_a, torch.Tensor):
        seg_a = seg_a.detach().cpu().numpy()
    if isinstance(seg_b, torch.Tensor):
        seg_b = seg_b.detach().cpu().numpy()

    # 转为二值
    seg_a = (seg_a > 0.5).astype(np.float32).flatten()
    seg_b = (seg_b > 0.5).astype(np.float32).flatten()

    intersection = np.sum(seg_a * seg_b)
    union_sum = seg_a.sum() + seg_b.sum()

    if union_sum < smooth:
        return 0.0

    dice = (2.0 * intersection + smooth) / (union_sum + smooth)
    return float(dice)


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
            organ_dice[organ] = 0.0
    return organ_dice


def compute_dsc_for_pair(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    deformation_field: sitk.Image,
    fixed_segmentations: Dict[str, str],
    organs: List[str] = None,
) -> Dict[str, float]:
    """
    对一对图像计算 DSC

    Args:
        fixed_image: 固定图像 (CT)
        moving_image: 浮动图像 (MR)
        deformation_field: 变形场 (displacement field，与 moving_image 相同空间)
        fixed_segmentations: 固定图像各器官分割文件路径
        organs: 器官列表

    Returns:
        dict: {"liver": DSC, "spleen": DSC, "kidney": DSC, "mean": 平均DSC}
    """
    organs = organs or ["liver", "spleen", "kidney"]

    # 读取固定图像分割标签
    fixed_segs = {}
    for organ in organs:
        if organ in fixed_segmentations:
            seg_path = fixed_segmentations[organ]
            if seg_path and Path(seg_path).exists():
                seg_img = sitk.ReadImage(seg_path)
                fixed_segs[organ] = sitk.GetArrayFromImage(seg_img)

    if not fixed_segs:
        warnings.warn("No fixed segmentations found for DSC computation.")
        return {}

    # 将变形场应用到 moving 图像的分割标签
    # 注意：这里需要 moving 图像的分割标签
    # 简化：直接使用 fixed 分割标签经过逆变形场映射
    warped_segs = {}

    # 使用 SimpleITK 的重采样器将固定图像分割标签映射到变形后的空间
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(moving_image)
    resampler.SetTransform(sitk.GetDisplacementFieldTransformer(deformation_field))
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    for organ, seg_img_path in fixed_segmentations.items():
        if seg_img_path and Path(seg_img_path).exists():
            # 读取分割标签
            seg_img = sitk.ReadImage(seg_img_path)
            # 逆变换：将 fixed 空间映射回 moving 空间
            inv_transform = sitk.DisplacementFieldTransform(deformation_field)
            inv_transform.InvertTransform()
            resampler.SetTransform(inv_transform)
            warped_seg = resampler.Execute(seg_img)
            warped_segs[organ] = sitk.GetArrayFromImage(warped_seg)

    # 计算各器官 DSC
    results = {}
    for organ in organs:
        if organ in fixed_segs and organ in warped_segs:
            results[organ] = dice_coefficient(fixed_segs[organ], warped_segs[organ])
        elif organ in fixed_segs:
            # 无对应 warped 分割时，预测为全0
            results[organ] = dice_coefficient(fixed_segs[organ], np.zeros_like(fixed_segs[organ]))
        else:
            results[organ] = 0.0

    if results:
        results["mean"] = np.mean(list(results.values()))

    return results


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
        record = {"case_id": case_id}
        record.update(organ_dice)
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
            return {"dsc30": 0.0, "worst_cases": [], "num_worst": 0, "total": 0, "worst_dsc_values": []}

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
            return {"dsc30": 0.0}

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
        mean_dsc = organ_dice.get("mean", np.mean([organ_dice.get(o, 0) for o in self.organs]))
        self.dsc30_scorer.add_case(case_id, mean_dsc, organ_dice)

        self.detailed_results.append({
            "case_id": case_id,
            "dsc": organ_dice,
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
        dsc_summary = self.dsc_scorer.compute()
        dsc30_result = self.dsc30_scorer.compute()

        report = {
            "DSC": dsc_summary,
            "DSC30": dsc30_result,
            "per_case_results": self.detailed_results,
        }

        # 计算综合排名分数
        dsc_mean = dsc_summary.get("overall", {}).get("mean", 0.0)
        dsc30_value = dsc30_result.get("dsc30", 0.0)

        # 综合分数 = 0.5 * DSC_mean + 0.5 * DSC30
        report["composite_score"] = 0.5 * dsc_mean + 0.5 * dsc30_value
        report["composite_score_description"] = "0.5 * DSC_mean + 0.5 * DSC30"

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

        # 综合分数
        print("\n【综合评分】")
        print(f"  综合分数 = 0.5 × DSC_mean + 0.5 × DSC30")
        if "composite_score" in report:
            print(f"  综合分数 = {report['composite_score']:.4f}")

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
    生成变形场提交文件

    Args:
        deformation_fields: 变形场列表（与 case_ids 对应）
        case_ids: 病例编号列表
        output_dir: 输出目录
        organ_labels: 器官标签

    Returns:
        dict: 提交文件信息
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_info = {
        "submission_format_version": "1.0",
        "organ_labels": organ_labels or ["liver", "spleen", "kidney"],
        "files": [],
    }

    for case_id, field in zip(case_ids, deformation_fields):
        if isinstance(field, (str, Path)):
            # 复制文件到提交目录
            src = Path(field)
            dst = output_dir / f"{case_id}_deformation_field.nii.gz"
            import shutil

            shutil.copy2(src, dst)
            submission_info["files"].append({
                "case_id": case_id,
                "filename": dst.name,
                "path": str(dst),
            })
        elif isinstance(field, sitk.Image):
            # 直接保存 SimpleITK 图像
            dst = output_dir / f"{case_id}_deformation_field.nii.gz"
            sitk.WriteImage(field, str(dst))
            submission_info["files"].append({
                "case_id": case_id,
                "filename": dst.name,
                "path": str(dst),
            })
        else:
            warnings.warn(f"Unknown deformation field type for case {case_id}: {type(field)}")

    # 保存提交信息
    info_path = output_dir / "submission_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(submission_info, f, indent=2, ensure_ascii=False)

    return submission_info
