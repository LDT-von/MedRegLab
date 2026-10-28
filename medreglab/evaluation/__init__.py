"""
MedRegLab Evaluation Module - 比赛评分与评估工具
"""

from .dsc_evaluator import (
    DSC_Scorer,
    DSC30_Scorer,
    CompetitionEvaluator,
    submit_deformation_fields,
    dice_coefficient,
    dice_per_organ,
    competition_case_dice,
)

__all__ = [
    "DSC_Scorer",
    "DSC30_Scorer",
    "CompetitionEvaluator",
    "submit_deformation_fields",
    "dice_coefficient",
    "dice_per_organ",
    "competition_case_dice",
]
