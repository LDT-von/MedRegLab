"""
MedRegLab - Medical Image Registration Laboratory
融合 Learn2Reg 挑战赛三大冠军方案的跨模态医学图像配准工具箱
"""

__version__ = "0.1.0"

from .core import ConvexAdam, RegistrationEngine, LossBuilder
from .modules import IMPACTLoss, uniGradICON, ConvexRegularizer
from .models import DINORegModel, DeformationField

__all__ = [
    # Core
    "ConvexAdam",
    "RegistrationEngine",
    "LossBuilder",
    # Modules
    "IMPACTLoss",
    "uniGradICON",
    "ConvexRegularizer",
    # Models
    "DINORegModel",
    "DeformationField",
]


def load_model(name: str = "dino_reg_mrct"):
    """
    Load a pre-configured registration model.

    Args:
        name: Model name. Options: "dino_reg_mrct", "impact_mrct", "convex_mrct"

    Returns:
        DINORegModel: Pre-configured registration model
    """
    from .models import DINORegModel

    configs = {
        "dino_reg_mrct": {
            "feature_extractor": "dinov2",
            "feature_dim": 1024,
            "optimizer": "convex_adam",
            "max_iter": 200,
        },
        "impact_mrct": {
            "feature_extractor": "dinov2",
            "feature_dim": 1024,
            "optimizer": "convex_adam",
            "loss_type": "impact",
            "semantic_weight": 0.5,
            "max_iter": 200,
        },
        "convex_mrct": {
            "optimizer": "convex_adam",
            "max_iter": 200,
            "use_physical_priors": True,
        },
    }

    if name not in configs:
        raise ValueError(f"Unknown model: {name}. Available: {list(configs.keys())}")

    return DINORegModel(**configs[name])
