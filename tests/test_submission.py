"""Geometry checks independent of model quality and patient data."""

import numpy as np
import SimpleITK as sitk
import torch

from medreglab.utils.submission import (
    displacement_to_fixed_grid, warp_original_moving, grid_folding_statistics,
)


def test_export_preserves_index_mapping_across_geometries():
    fixed = sitk.Image([9, 7, 5], sitk.sitkFloat32)
    fixed.SetSpacing([0.7, 1.2, 4.0])
    fixed.SetOrigin([10.0, -20.0, 100.0])
    fixed.SetDirection([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    moving = sitk.Image([13, 11, 9], sitk.sitkFloat32)
    moving.SetSpacing([1.5, 2.0, 3.0])
    moving.SetOrigin([-30.0, 15.0, -80.0])
    moving.SetDirection([-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
    phi = torch.zeros(1, 3, 4, 6, 8)
    phi[:, 0], phi[:, 1], phi[:, 2] = 0.5, -0.25, 0.75
    field = displacement_to_fixed_grid(phi, fixed, moving)
    assert field.GetSize() == fixed.GetSize()
    assert field.GetSpacing() == fixed.GetSpacing()
    assert field.GetDirection() == fixed.GetDirection()
    for point_index in [(0, 0, 0), (4, 3, 2), (8, 6, 4)]:
        moving_index = (
            np.asarray(point_index) / (np.asarray(fixed.GetSize()) - 1)
            + np.array([0.5, -0.25, 0.75]) / np.array([7, 5, 3])
        ) * (np.asarray(moving.GetSize()) - 1)
        expected = np.asarray(moving.TransformContinuousIndexToPhysicalPoint(moving_index.tolist()))
        fixed_point = np.asarray(fixed.TransformIndexToPhysicalPoint(point_index))
        np.testing.assert_allclose(fixed_point + np.asarray(field[point_index]), expected, atol=2e-5)


def test_export_identity_and_warp_original_intensities():
    array = np.arange(5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing([0.8, 1.4, 3.0])
    image.SetOrigin([-5.0, 17.0, 200.0])
    phi = torch.zeros(1, 3, 4, 6, 8)
    field = displacement_to_fixed_grid(phi, image, image)
    np.testing.assert_allclose(sitk.GetArrayFromImage(field), 0.0, atol=1e-6)
    warped = warp_original_moving(image, image, field)
    np.testing.assert_allclose(sitk.GetArrayFromImage(warped), array, atol=1e-6)


def test_jacobian_uses_cross_derivatives_and_xyz_axes():
    phi = torch.zeros(1, 3, 5, 7, 9)
    stats = grid_folding_statistics(phi)
    assert stats["fold_ratio"] == 0
    assert stats["min_jacobian"] == 1
    # u_x = -2*x gives an orientation-reversing x map, det = -1.
    phi[:, 0] = -2 * torch.arange(9).view(1, 1, 9)
    stats = grid_folding_statistics(phi)
    assert stats["fold_ratio"] == 1
    assert stats["min_jacobian"] == -1


def test_official_transform_export_keeps_physical_vectors_and_ct_geometry():
    from experiments.package_submission import official_field_from_transform

    ct = sitk.Image([9, 7, 5], sitk.sitkFloat32)
    ct.SetSpacing([0.7, 1.2, 4.0])
    ct.SetOrigin([10.0, -20.0, 100.0])
    ct.SetDirection([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    vectors = np.empty((5, 7, 9, 3), dtype=np.float32)
    vectors[:] = [2.5, -3.25, 4.75]
    source = sitk.GetImageFromArray(vectors, isVector=True)
    source.CopyInformation(ct)
    exported = official_field_from_transform(source, ct)
    assert exported.GetSize() == ct.GetSize()
    assert exported.GetSpacing() == ct.GetSpacing()
    assert exported.GetOrigin() == ct.GetOrigin()
    assert exported.GetDirection() == ct.GetDirection()
    assert exported.GetPixelID() == sitk.sitkVectorFloat32
    np.testing.assert_allclose(sitk.GetArrayViewFromImage(exported), vectors, atol=1e-6)
