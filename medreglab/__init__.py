"""
MedRegLab - Medical Image Registration Laboratory
跨模态医学图像配准工具箱（MR-CT）
"""

__version__ = "0.1.0"

from .core import LossBuilder
from .core.convex_3d_working import Simple3DRegistration, register_3d_simple
from .models import DeformationField, JacobianField

__all__ = [
    # Core
    "LossBuilder",
    "Simple3DRegistration",
    "register_3d_simple",
    # Models
    "DeformationField",
    "JacobianField",
]


def load_model(name: str = "convex_3d"):
    """
    Load a pre-configured registration model.

    Args:
        name: Model name. Only "convex_3d" is available:
            - "convex_3d": 3D registration via Adam + MSE/NCC + TV
              (per-image optimization, no learnable parameters)

    Returns:
        nn.Module: 模型
    """
    if name == "convex_3d":
        from .core.convex_3d_working import Simple3DRegistration
        return Simple3DRegistration(max_iter=100, lr=0.1, tv_weight=0.01, similarity="mse")

    raise ValueError(
        f"Unknown model: {name!r}. Available: ['convex_3d']. "
        "DINOv2 / IMPACT variants were removed (their ConvexAdam backend was broken)."
    )
