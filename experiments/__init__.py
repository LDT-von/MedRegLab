"""
Experiments Module - 实验脚本入口

主入口：
- train_competition.py : 训练
- infer_competition.py : 推理 / 生成提交文件
"""

from .train_competition import (
    train_competition,
    quick_train,
    set_seed,
)
from .infer_competition import (
    run_competition_inference,
    generate_submission,
)

__all__ = [
    "train_competition",
    "quick_train",
    "set_seed",
    "run_competition_inference",
    "generate_submission",
]
