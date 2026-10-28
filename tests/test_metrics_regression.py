"""Meaningful checks for registration metric inputs and physical units."""

import numpy as np
import pytest
import torch

from medreglab.utils.metrics import RegistrationMetrics


def test_dice_requires_warped_segmentation_not_intensity_image():
    metrics = RegistrationMetrics()
    fixed = np.array([0, 1, 1, 0], dtype=np.uint8)
    moved_mask = np.array([1, 1, 0, 0], dtype=np.uint8)
    warped_mask = np.array([0, 1, 0, 0], dtype=np.uint8)
    image = np.array([10, 20, 30, 40], dtype=np.float32)
    assert metrics.dice(warped_mask, fixed) == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="warped_mask"):
        metrics.compute_all(image, image, image, moving_mask=moved_mask, fixed_mask=fixed)
    assert metrics.compute_all(image, image, image, fixed_mask=fixed, warped_mask=warped_mask)["dice"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="segmentation"):
        metrics.dice(moved_mask, fixed, image)


def test_image_metrics_do_not_overflow_and_accept_grad_tensors():
    assert RegistrationMetrics.mse(np.array([0], np.uint8), np.array([255], np.uint8)) == 65025
    x = torch.tensor([0.0, 2.0], requires_grad=True)
    assert RegistrationMetrics.mse(x, x) == 0
    assert RegistrationMetrics.ncc(x, x) == pytest.approx(1)
    with pytest.raises(ValueError, match="same shape"):
        RegistrationMetrics.mse(np.ones((2, 1)), np.ones((2,)))


def test_surface_distances_respect_spacing():
    a = np.array([[0, 0, 0]], dtype=float)
    b = np.array([[1, 0, 0]], dtype=float)
    assert RegistrationMetrics.asd(a, b, spacing=(2, 3, 4)) == pytest.approx(2)
    assert RegistrationMetrics.hd(a, b, spacing=(2, 3, 4)) == pytest.approx(2)
    with pytest.raises(ValueError, match="Surface"):
        RegistrationMetrics.asd(np.empty((0, 3)), b)
