"""
MedRegLab Test Suite
"""

import pytest
import torch
import numpy as np

# Test imports
def test_imports():
    """Test that all modules can be imported"""
    from medreglab import load_model
    from medreglab.core import ConvexAdam, RegistrationEngine
    from medreglab.modules import IMPACTLoss, uniGradICON
    from medreglab.models import DINORegModel
    from medreglab.utils import RegistrationMetrics


def test_convex_adam():
    """Test ConvexAdam optimizer"""
    from medreglab.core import ConvexAdam

    optimizer = ConvexAdam(max_iter=10)
    assert optimizer.max_iter == 10


def test_impact_loss():
    """Test IMPACT loss"""
    from medreglab.modules import IMPACTLoss

    loss_fn = IMPACTLoss(feature_dim=128)
    x = torch.randn(2, 128)
    y = torch.randn(2, 128)

    loss = loss_fn(x, y)
    assert loss.item() >= 0


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
    assert jac.shape == (1, 63, 63)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
