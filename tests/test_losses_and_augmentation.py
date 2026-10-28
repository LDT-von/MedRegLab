"""Synthetic regression tests for loss semantics and spatial augmentations."""

import numpy as np
import pytest
import torch

from medreglab.core.loss_builder import LossBuilder
from medreglab.data.transforms import RandomElasticDeformation, RandomFlip3D, RandomRotate90


def test_loss_names_and_unimplemented_semantics_fail_closed():
    assert LossBuilder(loss_types=["mse", "ncc"]).losses == ["mse", "ncc"]
    for name in ("nmi", "impact"):
        with pytest.raises(NotImplementedError):
            LossBuilder(losses=[name])
    with pytest.raises(ValueError, match="Unknown loss"):
        LossBuilder(losses=["typo"])
    with pytest.raises(TypeError, match="Unsupported"):
        LossBuilder(losses=["mse"], misspelled_parameter=3)
    with pytest.raises(ValueError, match="displacement field"):
        LossBuilder(losses=["jacobian"])(torch.zeros(1, 1, 2, 3, 4), torch.zeros(1, 1, 2, 3, 4))
    with pytest.raises(NotImplementedError, match="IMPACT"):
        LossBuilder()(torch.zeros(1, 1, 2, 3, 4), torch.zeros(1, 1, 2, 3, 4), features_fixed=torch.zeros(1))


def test_ncc_is_per_sample_and_channel_and_differentiable():
    base = torch.arange(24, dtype=torch.float32).reshape(1, 1, 2, 3, 4)
    batch = torch.cat((base, 10 * base + 1000), dim=0).requires_grad_()
    objective = LossBuilder(losses=["ncc"], tv_weight=0, jacobian_weight=0)
    same = objective(batch, batch.detach())["total_tensor"]
    assert same.item() == pytest.approx(0, abs=1e-5)
    same.backward()
    assert torch.isfinite(batch.grad).all()
    opposite = objective(base, -base)["total_tensor"]
    assert opposite.item() == pytest.approx(2, abs=1e-5)
    constant = objective(torch.ones_like(base), torch.ones_like(base))["total_tensor"]
    assert torch.isfinite(constant)


def test_complete_jacobian_detects_axial_and_cross_shear_folds():
    objective = LossBuilder(losses=["mse"], tv_weight=0, jacobian_weight=0)
    identity = torch.zeros(2, 3, 4, 5, 6)
    assert objective._jacobian_regularization(identity).item() == 0
    assert objective._tv_regularization(identity).item() == 0
    reflection = identity.clone()
    reflection[:, 0] = -2 * torch.arange(6).reshape(1, 1, 1, 6)
    reflection.requires_grad_()
    penalty = objective._jacobian_regularization(reflection)
    assert penalty.item() == pytest.approx(1, abs=1e-5)
    penalty.backward()
    assert reflection.grad is not None and torch.isfinite(reflection.grad).all()
    shear = identity.clone()
    shear[:, 0] = 2 * torch.arange(5).reshape(1, 1, 5, 1)
    shear[:, 1] = torch.arange(6).reshape(1, 1, 1, 6)
    assert objective._jacobian_regularization(shear).item() == pytest.approx(1, abs=1e-5)
    reflection_2d = torch.zeros(1, 2, 5, 6)
    reflection_2d[:, 0] = -2 * torch.arange(6).reshape(1, 1, 6)
    assert objective._jacobian_regularization(reflection_2d).item() == pytest.approx(1, abs=1e-5)
    with pytest.raises(ValueError, match="Expected voxel displacement"):
        objective._jacobian_regularization(torch.zeros(1, 2, 4, 5, 6))


def test_weighted_loss_total_preserves_gradient():
    warped = torch.ones(1, 1, 3, 4, 5, requires_grad=True)
    fixed = torch.zeros_like(warped)
    phi = torch.zeros(1, 3, 3, 4, 5, requires_grad=True)
    objective = LossBuilder(losses=["mse", "tv", "jacobian"], weights={"mse": 2}, tv_weight=0.1)
    result = objective(warped, fixed, phi)
    assert result["total"] == pytest.approx(2)
    result["total_tensor"].backward()
    assert warped.grad is not None and phi.grad is not None
    assert torch.isfinite(warped.grad).all() and torch.isfinite(phi.grad).all()


def synthetic_sample(batch=1, device="cpu"):
    image = torch.zeros(batch, 1, 5, 6, 7, device=device)
    image[:, 0, 2, 3, 4] = 1
    # Dataset item uses 4D masks; collated batches use 5D masks.
    mask = image.to(torch.uint8) if batch > 1 else image[0].to(torch.uint8)
    return {"fixed": image.clone(), "moving": image.clone(),
            "seg_fixed": {"kidney": mask.clone()}, "seg_moving": {"kidney": mask.clone()}}


def assert_images_masks_aligned(sample):
    for image_key, mask_key in (("fixed", "seg_fixed"), ("moving", "seg_moving")):
        image = sample[image_key]
        mask = sample[mask_key]["kidney"]
        if mask.ndim == 4:
            mask = mask.unsqueeze(0)
        assert torch.equal(image > 0.5, mask.bool())


def test_flip_and_rotate_keep_both_modalities_and_masks_aligned(monkeypatch):
    sample = synthetic_sample()
    sample = RandomFlip3D(axes=[0, 2], p=1.0)(sample)
    assert sample["spatial_augmented"] is True
    assert_images_masks_aligned(sample)
    monkeypatch.setattr("medreglab.data.transforms.random.choice", lambda options: 1)
    sample = RandomRotate90(p=1.0)(sample)
    assert_images_masks_aligned(sample)
    assert sample["fixed"].shape[-2:] == (7, 6)


def test_elastic_uses_nearest_for_masks_and_supports_batches(monkeypatch):
    def controls(*shape, **kwargs):
        values = torch.zeros(*shape, **kwargs)
        values[0] = 0.75
        return values

    monkeypatch.setattr("medreglab.data.transforms.torch.randn", controls)
    sample = synthetic_sample(batch=2)
    output = RandomElasticDeformation(num_ctrl_points=2, deform_std=1.0, p=1.0)(sample)
    assert output["spatial_augmented"] is True
    assert_images_masks_aligned(output)
    assert output["seg_fixed"]["kidney"].dtype == torch.uint8
    assert set(torch.unique(output["seg_fixed"]["kidney"]).tolist()) == {0, 1}
    assert output["fixed"][0, 0, 2, 3, 3].item() == pytest.approx(0.75, abs=1e-5)
    assert output["fixed"][0, 0, 2, 3, 4].item() == pytest.approx(0.25, abs=1e-5)


def test_mismatched_batches_are_rejected():
    sample = synthetic_sample(batch=2)
    sample["seg_fixed"]["kidney"] = sample["seg_fixed"]["kidney"][0]
    with pytest.raises(ValueError, match="ambiguous or mismatched batch"):
        RandomFlip3D(axes=[0], p=1)(sample)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_elastic_supports_cuda_device_with_nearest_labels():
    sample = synthetic_sample(batch=2, device="cuda")
    output = RandomElasticDeformation(num_ctrl_points=2, deform_std=0, p=1)(sample)
    assert output["fixed"].device.type == "cuda"
    assert output["seg_fixed"]["kidney"].device.type == "cuda"
    assert_images_masks_aligned(output)
