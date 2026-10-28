"""Synthetic checks for the rigid pre-alignment helpers.

Everything here is synthetic. No patient data is read.

The important regression is the composition of a rigid map with a deformable
pull field.  An earlier version reordered the deformable array with a double
axis flip, which cancels out for pure translations and therefore stayed
invisible in translation-only tests, but produced a wrong field as soon as the
rigid part contained a rotation.  The tests below compare the composed field
against a brute-force per-voxel reference so that any axis mistake shows up.
"""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from medreglab.evaluation.dev_eval import (
    ALL_CONFIGS,
    DEFAULT_CONFIGS,
    PREALIGN_CONFIGS,
    ROBUST_RIGID_CONFIGS,
    config_by_name,
)
from medreglab.evaluation.prealign import (
    PreAlignmentError,
    compose_rigid_and_deformable,
    field_statistics,
    normalize_ct,
    normalize_mr,
    resample_to_fixed_grid,
    transform_to_displacement_field,
)

_DIRECTION_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _fixed(size_xyz=(4, 5, 6), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """A non-cubic CT-like reference image (z, y, x ordering in numpy)."""
    image = sitk.GetImageFromArray(np.zeros(tuple(reversed(size_xyz)), dtype=np.float32))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(_DIRECTION_IDENTITY)
    return image


def _field(array_zyx, reference):
    """Build a vector field with the reference's geometry from a (z, y, x, c) array."""
    assert array_zyx.shape[:3] == tuple(reversed(reference.GetSize()))
    field = sitk.GetImageFromArray(np.ascontiguousarray(array_zyx.astype(np.float32)), isVector=True)
    field.CopyInformation(reference)
    return field


def _rotation_about_origin(angle):
    transform = sitk.Euler3DTransform()
    transform.SetCenter((0.0, 0.0, 0.0))
    transform.SetRotation(0.0, 0.0, angle)
    return transform


def test_composition_matches_brute_force_under_rotation():
    """A rotated rigid map composed with a varying field must match the reference.

    The brute-force reference transforms every voxel centre individually, so an
    axis swap or a silent double flip cannot hide.
    """
    fixed = _fixed()
    deformable = (np.random.RandomState(7).rand(*reversed(fixed.GetSize()), 3) - 0.5) * 4.0
    rigid = _rotation_about_origin(np.pi / 2.0)

    composed = compose_rigid_and_deformable(rigid, _field(deformable, fixed), fixed)
    actual = np.array(sitk.GetArrayViewFromImage(composed), dtype=np.float64)

    for z in range(fixed.GetSize()[2]):
        for y in range(fixed.GetSize()[1]):
            for x in range(fixed.GetSize()[0]):
                point = fixed.TransformContinuousIndexToPhysicalPoint(
                    (float(x), float(y), float(z))
                )
                shifted = np.asarray(point, dtype=np.float64) + deformable[z, y, x]
                mapped = np.asarray(rigid.TransformPoint(tuple(shifted)), dtype=np.float64)
                expected = mapped - np.asarray(point, dtype=np.float64)
                assert np.allclose(actual[z, y, x], expected, atol=1e-4), (
                    f"voxel (x={x}, y={y}, z={z}) mismatch: "
                    f"{actual[z, y, x].tolist()} != {expected.tolist()}"
                )


def test_composition_with_pure_translation_adds_the_deformable_field():
    """With a pure translation the composed field is the translation plus u."""
    fixed = _fixed(spacing=(1.5, 2.0, 3.0), origin=(11.0, -7.0, 42.0))
    offset = np.array([2.0, -3.0, 5.0])
    rigid = sitk.TranslationTransform(3)
    rigid.SetOffset(tuple(offset))

    deformable = np.zeros((*reversed(fixed.GetSize()), 3), dtype=np.float32)
    for x in range(fixed.GetSize()[0]):
        deformable[:, :, x, 0] = 0.5 * x
    composed = compose_rigid_and_deformable(rigid, _field(deformable, fixed), fixed)
    actual = np.array(sitk.GetArrayViewFromImage(composed), dtype=np.float64)

    expected = deformable.astype(np.float64) + offset
    assert np.allclose(actual, expected, atol=1e-4)


def test_composition_rejects_a_field_on_a_different_grid():
    fixed = _fixed()
    wrong = sitk.GetImageFromArray(np.zeros((3, 4, 5, 3), dtype=np.float32), isVector=True)
    with pytest.raises(PreAlignmentError, match="grid does not match"):
        compose_rigid_and_deformable(sitk.TranslationTransform(3), wrong, fixed)


def test_displacement_field_reproduces_the_rigid_resampling():
    """Resampling through the pull field must equal resampling through the transform."""
    fixed = _fixed(size_xyz=(6, 6, 6), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    content = np.zeros((6, 6, 6), dtype=np.float32)
    content[2:4, 2:4, 2:4] = 100.0
    moving = sitk.GetImageFromArray(content)
    moving.CopyInformation(fixed)

    rigid = sitk.TranslationTransform(3)
    rigid.SetOffset((1.0, -2.0, 0.5))

    field = transform_to_displacement_field(rigid, fixed)
    stats = field_statistics(field)
    assert stats["mean_displacement_mm"] == pytest.approx([1.0, -2.0, 0.5], abs=1e-4)

    through_field = sitk.Resample(
        moving, fixed, sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64)),
        sitk.sitkLinear, 0.0, sitk.sitkFloat32,
    )
    through_transform = resample_to_fixed_grid(moving, fixed, rigid)
    assert np.allclose(
        np.array(sitk.GetArrayViewFromImage(through_field)),
        np.array(sitk.GetArrayViewFromImage(through_transform)),
        atol=1e-4,
    )


def test_normalization_matches_the_dataset_convention():
    ct = np.array([[-1000.0, 0.0, 600.0]], dtype=np.float32)
    expected_ct = (np.clip(ct, -200.0, 500.0) - (-200.0)) / (500.0 - (-200.0))
    assert np.allclose(normalize_ct(ct), expected_ct)
    mr = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    normalized = normalize_mr(mr)
    assert normalized.dtype == np.float32
    assert float(normalized.max()) == pytest.approx(1.0)
    constant = normalize_mr(np.full((2, 2), 5.0, dtype=np.float32))
    assert np.allclose(constant, 0.0)


def test_prealign_configurations_are_opt_in():
    """The pre-alignment configurations are extra, never part of the fixed set."""
    assert [config.name for config in DEFAULT_CONFIGS] == [
        "A_zero_physical",
        "B_zero_model_identity",
        "C1_current",
        "C2_tv005",
        "C3_tv005_lr005",
        "C4_iter100",
    ]
    assert len(ALL_CONFIGS) == len(DEFAULT_CONFIGS) + len(PREALIGN_CONFIGS) + len(
        ROBUST_RIGID_CONFIGS
    )
    assert config_by_name("P1_prealign").mode == "prealign"
    assert config_by_name("P2_prealign_C1").mode == "prealign_model"
