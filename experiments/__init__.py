# 实验脚本
from .train import train_registration
from .eval import evaluate_registration
from .infer import run_inference

__all__ = ["train_registration", "evaluate_registration", "run_inference"]
