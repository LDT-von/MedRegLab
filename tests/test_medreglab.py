"""
MedRegLab Test Suite
"""

import pytest
import torch
import numpy as np


# Test imports
def test_imports():
    """Test that all modules can be imported"""
    from medreglab import load_model, Simple3DRegistration
    from medreglab.core import LossBuilder
    from medreglab.models import DeformationField, JacobianField
    from medreglab.utils import RegistrationMetrics


def test_load_model():
    """Test the only registered model"""
    from medreglab import load_model

    model = load_model("convex_3d")
    assert model is not None

    with pytest.raises(ValueError):
        load_model("dino_reg_mrct")


def test_3d_registration():
    """Test 3D registration produces a valid deformation field"""
    from medreglab import load_model

    model = load_model("convex_3d")
    moving = torch.randn(1, 1, 16, 16, 16)
    fixed = torch.randn(1, 1, 16, 16, 16)

    result = model(moving, fixed)
    assert result["phi"].shape == (1, 3, 16, 16, 16)
    assert result["warped"].shape == (1, 1, 16, 16, 16)
    # Simple3DRegistration 返回的 loss 是 Python float
    assert float(result["loss"]) >= 0


def test_loss_builder():
    """Test LossBuilder"""
    from medreglab.core import LossBuilder

    loss_builder = LossBuilder(loss_types=["mse", "ncc"])
    assert loss_builder is not None


def test_metrics():
    """Test registration metrics"""
    from medreglab.utils import RegistrationMetrics

    metrics = RegistrationMetrics()

    I = np.random.rand(100, 100)
    J = np.random.rand(100, 100)

    ncc = metrics.ncc(I, J)
    mse = metrics.mse(I, J)

    assert -1 <= ncc <= 1
    assert mse >= 0


def test_deformation_field():
    """Test deformation field utilities"""
    from medreglab.models.deformation_field import DeformationField, JacobianField

    phi = torch.randn(1, 2, 64, 64)
    jac = JacobianField.compute(phi)
    # _compute_2d 保留 batch 与通道维: [B, 1, H-1, W-1]
    assert jac.shape == (1, 1, 63, 63)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
