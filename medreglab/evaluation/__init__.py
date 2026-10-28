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
from .dev_eval import (
    ALL_CONFIGS,
    DEFAULT_CONFIGS,
    PREALIGN_CONFIGS,
    ConfigResult,
    EvalConfig,
    EvaluationError,
    build_comparison,
    collect_development_cases,
    config_by_name,
    preflight,
    run_configurations,
    score_case,
    zero_model_identity_field,
    zero_physical_field,
)

__all__ = [
    "DSC_Scorer",
    "DSC30_Scorer",
    "CompetitionEvaluator",
    "submit_deformation_fields",
    "dice_coefficient",
    "dice_per_organ",
    "competition_case_dice",
    "ALL_CONFIGS",
    "DEFAULT_CONFIGS",
    "PREALIGN_CONFIGS",
    "ConfigResult",
    "EvalConfig",
    "EvaluationError",
    "build_comparison",
    "collect_development_cases",
    "config_by_name",
    "preflight",
    "run_configurations",
    "score_case",
    "zero_model_identity_field",
    "zero_physical_field",
]
