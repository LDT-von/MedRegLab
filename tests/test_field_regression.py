"""Analytic voxel-field checks independent of patient data and image registration."""

import numpy as np
import pytest
import torch

from medreglab.models.deformation_field import DeformationField, JacobianField
from medreglab.utils.metrics import compute_jacobian_determinant


@pytest.mark.parametrize("size", [(5, 7), (4, 5, 7)])
def test_grid_and_composition_identity_and_translation(size):
    dims = len(size)
    grid = DeformationField.create_grid(size)
    assert grid.shape == (dims, *size)
    assert grid[0].flatten()[0].item() == -1
    assert grid[0].flatten()[size[-1] - 1].item() == 1
    generator = torch.Generator().manual_seed(51)
    field = torch.randn((2, dims, *size), generator=generator)
    zero = torch.zeros_like(field)
    torch.testing.assert_close(DeformationField.compose(field, zero), field, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(DeformationField.compose(zero, field), field, atol=2e-6, rtol=1e-6)
    original = field.clone()
    DeformationField.compose(zero, field)
    torch.testing.assert_close(field, original)
    first = torch.ones_like(field) * 2
    second = torch.ones_like(field) * -0.25
    torch.testing.assert_close(DeformationField.compose(first, second), first + second)


def test_3d_composition_samples_first_field_at_second_displacement():
    first = torch.zeros(1, 3, 4, 5, 7)
    first[:, 0] = 0.5 * torch.arange(7).view(1, 1, 1, 7)
    second = torch.zeros_like(first)
    second[:, 0] = 1
    composed = DeformationField.compose(first, second)
    expected = 1 + 0.5 * (torch.arange(6) + 1)
    torch.testing.assert_close(composed[0, 0, 2, 2, :6], expected)
    assert torch.count_nonzero(composed[:, 1:]) == 0


@pytest.mark.parametrize("size", [(5, 7), (4, 5, 7)])
def test_inverse_translation_and_noninvertible_field(size):
    dims = len(size)
    translation = torch.zeros(1, dims, *size)
    for axis in range(dims):
        translation[:, axis] = (axis + 1) * 0.25
    inverse = DeformationField.invert(translation)
    torch.testing.assert_close(inverse, -translation, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(DeformationField.compose(translation, inverse), torch.zeros_like(translation), atol=1e-6, rtol=0)
    folded = torch.zeros_like(translation)
    folded[:, 0] = -2 * torch.arange(size[-1]).reshape(*([1] * (dims - 1)), size[-1])
    with pytest.raises(RuntimeError, match="did not converge"):
        DeformationField.invert(folded, num_iter=10)


@pytest.mark.parametrize("size", [(7, 17), (5, 7, 17)])
def test_inverse_spatially_varying_field_is_not_just_negation(size):
    dims = len(size)
    field = torch.zeros(1, dims, *size)
    x = torch.arange(size[-1], dtype=torch.float32)
    field[:, 0] = 0.1 * torch.sin(torch.pi * x / (size[-1] - 1))
    inverse = DeformationField.invert(field, num_iter=30, tolerance=1e-3)
    assert (inverse + field).abs().max().item() > 1e-4
    assert DeformationField.compose(field, inverse).abs().max().item() < 1e-3
    assert DeformationField.compose(inverse, field).abs().max().item() < 1e-3


@pytest.mark.parametrize("size", [(7, 9), (5, 7, 9)])
def test_smooth_preserves_constants_and_smooths_each_channel(size):
    dims = len(size)
    field = torch.zeros(2, dims, *size)
    field[0, 0] = 2
    field[0, 1] = -3
    center = tuple(length // 2 for length in size)
    field[(1, 0, *center)] = 1
    smoothed = DeformationField.smooth(field, sigma=1)
    torch.testing.assert_close(smoothed[0, 0], torch.full(size, 2.0))
    torch.testing.assert_close(smoothed[0, 1], torch.full(size, -3.0))
    assert 0 < smoothed[(1, 0, *center)].item() < 1
    neighbor = (*center[:-1], center[-1] + 1)
    assert smoothed[(1, 0, *neighbor)].item() > 0
    assert torch.count_nonzero(smoothed[1, 1:]) == 0


def test_add_bias_applies_to_channels_not_spatial_axis():
    field = torch.zeros(2, 3, 4, 5, 7)
    result = DeformationField.add_bias(field, torch.tensor([[1., 2., 3.], [-1., -2., -3.]]))
    assert result[:, :, 0, 0, 0].tolist() == [[1., 2., 3.], [-1., -2., -3.]]
    assert result[0, 1].min().item() == 2 and result[0, 1].max().item() == 2
    with pytest.raises(ValueError):
        DeformationField.add_bias(field, torch.zeros(2))


def test_2d_jacobian_uses_xy_axes_and_cross_derivatives():
    y, x = torch.meshgrid(torch.arange(5.), torch.arange(7.), indexing="ij")
    field = torch.stack((2 * y, x)).unsqueeze(0)
    determinant = JacobianField.compute(field)
    torch.testing.assert_close(determinant, -torch.ones(1, 1, 4, 6))
    assert JacobianField.check_folding(determinant)["fold_ratio"] == 1
    assert JacobianField.check_folding(torch.tensor([1., 0., -1.]))["fold_ratio"] == pytest.approx(2 / 3)


def test_3d_jacobian_matches_affine_matrix_with_shear():
    z, y, x = torch.meshgrid(torch.arange(4.), torch.arange(5.), torch.arange(7.), indexing="ij")
    displacement = torch.stack((0.2*x + 0.5*y - 0.1*z,
                                0.3*x - 0.2*y + 0.4*z,
                                -0.1*x + 0.7*y + 0.3*z)).unsqueeze(0)
    displacement.requires_grad_()
    determinant = JacobianField.compute(displacement)
    matrix = np.array([[1.2, 0.5, -0.1], [0.3, 0.8, 0.4], [-0.1, 0.7, 1.3]])
    expected = torch.full((1, 3, 4, 6), float(np.linalg.det(matrix)))
    torch.testing.assert_close(determinant, expected, atol=2e-6, rtol=2e-6)
    determinant.mean().backward()
    assert torch.isfinite(displacement.grad).all()
    identity = torch.zeros_like(displacement)
    torch.testing.assert_close(JacobianField.compute(identity), torch.ones_like(determinant))


def test_physical_jacobian_3d_keeps_full_resolution_and_spacing():
    spacing = (2.0, 3.0, 5.0)  # x, y, z
    z, y, x = np.meshgrid(np.arange(4) * spacing[2], np.arange(5) * spacing[1],
                          np.arange(7) * spacing[0], indexing="ij")
    field = np.stack((0.2*x + 0.5*y - 0.1*z, 0.3*x - 0.2*y + 0.4*z,
                      -0.1*x + 0.7*y + 0.3*z)).astype(np.float64)
    matrix = np.array([[1.2, 0.5, -0.1], [0.3, 0.8, 0.4], [-0.1, 0.7, 1.3]])
    determinant = compute_jacobian_determinant(field, spacing=spacing)
    assert determinant.shape == (4, 5, 7) and determinant.dtype == np.float64
    np.testing.assert_allclose(determinant, np.linalg.det(matrix), atol=1e-12)
    tensor = torch.tensor(field, requires_grad=True)
    tensor_det = compute_jacobian_determinant(tensor, spacing=spacing)
    assert tensor_det.device == tensor.device and tensor_det.dtype == tensor.dtype
    tensor_det.mean().backward()
    assert torch.isfinite(tensor.grad).all()


def test_physical_jacobian_2d_batch_and_ambiguous_shape():
    spacing = (2.0, 3.0)
    y, x = torch.meshgrid(torch.arange(5.) * spacing[1], torch.arange(7.) * spacing[0], indexing="ij")
    field = torch.stack((0.2*x + 0.5*y, 0.3*x - 0.2*y))
    determinant = compute_jacobian_determinant(field, spacing=spacing)
    expected = float(np.linalg.det(np.array([[1.2, 0.5], [0.3, 0.8]])))
    torch.testing.assert_close(determinant, torch.full((5, 7), expected), atol=1e-6, rtol=1e-6)
    ambiguous = field.unsqueeze(0).repeat(3, 1, 1, 1)
    with pytest.raises(ValueError, match="ambiguous"):
        compute_jacobian_determinant(ambiguous, spacing=spacing)
    batch = compute_jacobian_determinant(ambiguous, spacing=spacing, spatial_dims=2)
    assert batch.shape == (3, 5, 7)
    torch.testing.assert_close(batch[0], determinant)
    depth_two_3d = torch.zeros(3, 2, 5, 7)
    volume = compute_jacobian_determinant(depth_two_3d, spatial_dims=3)
    torch.testing.assert_close(volume, torch.ones(2, 5, 7))
    with pytest.raises(ValueError, match="spacing"):
        compute_jacobian_determinant(field, spacing=(2., 3., 4.))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_physical_jacobian_preserves_cuda_device_and_gradient():
    field = torch.zeros(3, 4, 5, 7, device="cuda", requires_grad=True)
    determinant = compute_jacobian_determinant(field, spacing=(2., 3., 5.))
    assert determinant.is_cuda and determinant.shape == (4, 5, 7)
    torch.testing.assert_close(determinant, torch.ones_like(determinant))
    determinant.mean().backward()
    assert field.grad.is_cuda and torch.isfinite(field.grad).all()
