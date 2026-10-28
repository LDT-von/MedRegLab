"""Regression tests for the per-image optimizer; all volumes are synthetic."""

import numpy as np
import pytest
import torch

from medreglab.core.convex_3d_working import Simple3DRegistration


def ramp():
    return torch.linspace(0, 1, 7).view(1, 1, 1, 1, 7).expand(1, 1, 5, 6, 7).clone()


def test_returned_best_field_has_reported_loss_after_one_update():
    moving = ramp()
    fixed = moving + 0.05
    model = Simple3DRegistration(max_iter=1, lr=0.1, tv_weight=0, similarity="mse")
    result = model(moving, fixed)
    actual_loss = torch.mean((model._warp(moving, result["phi"]) - fixed) ** 2).item()
    assert result["loss"] == pytest.approx(actual_loss, abs=1e-7)
    assert result["loss"] < result["loss_history"][0]
    assert result["phi"].abs().max().item() > 0
    assert len(result["loss_history"]) == 1
    assert result["phi_inv"] is None


def test_worse_post_update_does_not_replace_best_field():
    moving = ramp()
    fixed = moving + 0.05
    model = Simple3DRegistration(max_iter=1, lr=20, tv_weight=0, similarity="mse")
    result = model(moving, fixed)
    assert torch.count_nonzero(result["phi"]).item() == 0
    assert result["loss"] == pytest.approx(result["loss_history"][0], abs=1e-7)
    assert torch.mean((result["warped"] - fixed) ** 2).item() == pytest.approx(result["loss"], abs=1e-7)


def test_registration_uses_internal_gradients_under_no_grad():
    model = Simple3DRegistration(max_iter=2, lr=0.1, tv_weight=0, similarity="mse")
    moving = ramp()
    with torch.no_grad():
        result = model(moving, moving + 0.05)
    assert np.isfinite(result["loss"])
    assert result["phi"].abs().max().item() > 0
    assert not result["warped"].requires_grad
    with torch.inference_mode(), pytest.raises(RuntimeError, match="internal gradients"):
        model(moving, moving)


@pytest.mark.parametrize("kwargs", [
    {"max_iter": 0}, {"max_iter": 1.5}, {"lr": 0}, {"lr": float("nan")},
    {"tv_weight": -1}, {"similarity": "invalid"},
])
def test_invalid_optimizer_parameters_fail_early(kwargs):
    with pytest.raises(ValueError):
        Simple3DRegistration(**kwargs)


def test_mutated_max_iter_and_invalid_inputs_fail_early():
    moving = ramp()
    model = Simple3DRegistration(max_iter=1)
    model.max_iter = 0
    with pytest.raises(ValueError, match="max_iter"):
        model(moving, moving)
    model.max_iter = 1
    with pytest.raises(ValueError, match="identical shapes"):
        model(moving, moving[..., :-1])
    with pytest.raises(ValueError, match="spatial dimension"):
        model(moving[..., :1], moving[..., :1])
    bad = moving.clone()
    bad[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(bad, moving)
    with pytest.raises(ValueError, match="positive weight"):
        model(moving, moving, torch.zeros_like(moving))


def test_masked_multichannel_ncc_is_per_sample_weighted_pearson():
    generator = torch.Generator().manual_seed(31)
    moving = torch.randn(2, 2, 3, 4, 5, generator=generator)
    fixed = torch.randn(2, 2, 3, 4, 5, generator=generator)
    weights = torch.rand(2, 1, 3, 4, 5, generator=generator)
    weights[:, :, 0] = 0
    model = Simple3DRegistration(max_iter=1, similarity="ncc")
    actual = model._ncc(moving, fixed, weights).item()
    reference = []
    for i in range(2):
        x = moving[i].numpy().astype(np.float64)
        y = fixed[i].numpy().astype(np.float64)
        w = np.broadcast_to(weights[i].numpy(), x.shape).astype(np.float64)
        xc = x - (x * w).sum() / w.sum()
        yc = y - (y * w).sum() / w.sum()
        reference.append((w * xc * yc).sum() / np.sqrt((w * xc ** 2).sum() * (w * yc ** 2).sum()))
    assert actual == pytest.approx(np.mean(reference), abs=1e-6)


def test_masked_mse_normalizes_broadcast_weights_per_case():
    warped = torch.empty(2, 2, 2, 2, 2)
    warped[0, 0], warped[0, 1] = 1, 3
    warped[1, 0], warped[1, 1] = 2, 4
    fixed = torch.zeros_like(warped)
    phi = torch.zeros(2, 3, 2, 2, 2)
    model = Simple3DRegistration(max_iter=1, tv_weight=0, similarity="mse")
    mask = torch.zeros(2, 1, 2, 2, 2)
    mask[0, :, 0, 0, 0] = 0.25
    mask[1] = 1
    actual = model._objective(warped, fixed, phi, mask).item()
    assert actual == pytest.approx((5 + 10) / 2)
    scaled = mask.clone()
    scaled[0] *= 10
    scaled[1] *= 0.001
    assert model._objective(warped, fixed, phi, scaled).item() == pytest.approx(actual)

    channel_mask = torch.empty_like(warped)
    channel_mask[0, 0], channel_mask[0, 1] = 0.2, 0.8
    channel_mask[1, 0], channel_mask[1, 1] = 0.75, 0.25
    expected = ((0.2 * 1 + 0.8 * 9) + (0.75 * 4 + 0.25 * 16)) / 2
    assert model._objective(warped, fixed, phi, channel_mask).item() == pytest.approx(expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_and_cuda_registration_agree_on_synthetic_input():
    generator = torch.Generator().manual_seed(29)
    moving = torch.randn(1, 1, 5, 6, 7, generator=generator)
    fixed = torch.randn(1, 1, 5, 6, 7, generator=generator)
    model = Simple3DRegistration(max_iter=3, lr=0.05, tv_weight=0.01, similarity="mse")
    cpu = model(moving, fixed)
    gpu = model(moving.cuda(), fixed.cuda())
    assert gpu["loss"] == pytest.approx(cpu["loss"], rel=1e-5, abs=1e-6)
    # The backends accumulate grid_sample gradients in different orders.
    torch.testing.assert_close(gpu["phi"].cpu(), cpu["phi"], rtol=5e-4, atol=1e-4)
    torch.testing.assert_close(gpu["warped"].cpu(), cpu["warped"], rtol=5e-4, atol=1e-4)
