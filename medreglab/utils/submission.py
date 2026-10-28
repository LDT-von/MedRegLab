"""Export index-grid registration results as physical CT-to-MR sampling maps."""

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


def displacement_to_fixed_grid(phi, fixed, moving):
    """Return an LPS-mm displacement field on the original fixed image grid.

    ``phi`` contains x/y/z displacements in the resized model grid, with
    align_corners=True. The returned pull transform maps a fixed physical
    point to the moving physical point to sample; it includes the mapping
    between the two original image geometries.
    """
    if phi.ndim != 5 or phi.shape[:2] != (1, 3):
        raise ValueError("Expected phi with shape [1, 3, D, H, W]")
    if not torch.isfinite(phi).all():
        raise ValueError("Non-finite model displacement")
    model_xyz = np.asarray(phi.shape[2:][::-1], dtype=np.float64)
    fixed_xyz = np.asarray(fixed.GetSize(), dtype=np.float64)
    moving_xyz = np.asarray(moving.GetSize(), dtype=np.float64)
    if np.any(model_xyz < 2) or np.any(fixed_xyz < 2) or np.any(moving_xyz < 2):
        raise ValueError("All spatial dimensions must be at least two")
    native = F.interpolate(
        phi.detach().float().cpu(),
        size=tuple(int(n) for n in fixed_xyz[::-1]),
        mode="trilinear", align_corners=True,
    )[0].numpy()
    fixed_matrix = np.asarray(fixed.GetDirection()).reshape(3, 3) @ np.diag(fixed.GetSpacing())
    moving_matrix = np.asarray(moving.GetDirection()).reshape(3, 3) @ np.diag(moving.GetSpacing())
    origin_delta = np.asarray(moving.GetOrigin()) - np.asarray(fixed.GetOrigin())
    index_scale = (moving_xyz - 1) / (fixed_xyz - 1)
    phi_scale = (moving_xyz - 1) / (model_xyz - 1)
    z, y, x = np.ogrid[:int(fixed_xyz[2]), :int(fixed_xyz[1]), :int(fixed_xyz[0])]
    indices = (x, y, z)
    displacement = np.empty((*native.shape[1:], 3), dtype=np.float32)
    for physical_axis in range(3):
        component = displacement[..., physical_axis]
        component.fill(origin_delta[physical_axis])
        for index_axis in range(3):
            component += (
                (moving_matrix[physical_axis, index_axis] * index_scale[index_axis]
                 - fixed_matrix[physical_axis, index_axis]) * indices[index_axis]
            )
            coefficient = moving_matrix[physical_axis, index_axis] * phi_scale[index_axis]
            if coefficient:
                component += coefficient * native[index_axis]
    field = sitk.GetImageFromArray(displacement, isVector=True)
    field.CopyInformation(fixed)
    return field


def warp_original_moving(moving, fixed, field):
    """Resample original MR intensities onto the original CT grid."""
    transform = sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))
    return sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)


def grid_folding_statistics(phi):
    """Compute det(I + du/dx), including all cross derivatives, on model grid."""
    array = phi.detach().float().cpu().numpy()[0]
    derivatives = [np.gradient(component, axis=(2, 1, 0)) for component in array]
    a, b, c = derivatives[0]
    d, e, f = derivatives[1]
    g, h, i = derivatives[2]
    a, e, i = a + 1, e + 1, i + 1
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return {
        "fold_ratio": float(np.mean(determinant <= 0)),
        "min_jacobian": float(determinant.min()),
        "mean_jacobian": float(determinant.mean()),
    }
