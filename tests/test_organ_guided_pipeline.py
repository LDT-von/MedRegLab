"""Synthetic checks for the organ-guided inference and submission verification.

Everything here is synthetic. No patient data is read and no registration is
run: the tests exercise the geometry contract and the failure handling only.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from experiments.organ_guided_inference import (
    CASE_IDS,
    InferenceError,
    _assert_native_geometry,
)
from experiments.verify_submission import (
    VerificationError,
    verify_field_against_ct,
    verify_point_mapping,
)


def _image(size_xyz, spacing, origin, direction=None):
    array = np.zeros((*reversed(size_xyz), 1), dtype=np.float32)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    if direction is not None:
        image.SetDirection(direction)
    return image


def _field(size_xyz, spacing, origin, values, direction=None):
    array = np.asarray(values, dtype=np.float32)
    field = sitk.GetImageFromArray(array, isVector=True)
    field.SetSpacing(spacing)
    field.SetOrigin(origin)
    if direction is not None:
        field.SetDirection(direction)
    return field


def test_assert_native_geometry_accepts_matching_field():
    ct = _image((4, 5, 6), (1.0, 2.0, 3.0), (10.0, 20.0, 30.0))
    values = np.zeros((*reversed(ct.GetSize()), 3), dtype=np.float32)
    values[..., 0] = 1.5
    field = _field(ct.GetSize(), ct.GetSpacing(), ct.GetOrigin(), values)
    _assert_native_geometry(field, ct, "001")


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda f, ct: f.SetSpacing((9.0, 9.0, 9.0)), id="spacing"),
        pytest.param(lambda f, ct: f.SetOrigin((0.0, 0.0, 0.0)), id="origin"),
        pytest.param(lambda f, ct: f.SetDirection((0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)), id="direction"),
    ],
)
def test_assert_native_geometry_rejects_mismatch(mutate):
    ct = _image((4, 5, 6), (1.0, 2.0, 3.0), (10.0, 20.0, 30.0))
    values = np.zeros((*reversed(ct.GetSize()), 3), dtype=np.float32)
    field = _field(ct.GetSize(), ct.GetSpacing(), ct.GetOrigin(), values)
    mutate(field, ct)
    with pytest.raises(InferenceError):
        _assert_native_geometry(field, ct, "001")


def test_assert_native_geometry_rejects_size_mismatch():
    ct = _image((4, 5, 6), (1.0, 2.0, 3.0), (10.0, 20.0, 30.0))
    values = np.zeros((*reversed((2, 2, 2)), 3), dtype=np.float32)
    field = _field((2, 2, 2), ct.GetSpacing(), ct.GetOrigin(), values)
    with pytest.raises(InferenceError):
        _assert_native_geometry(field, ct, "001")


def test_assert_native_geometry_rejects_non_finite():
    ct = _image((4, 5, 6), (1.0, 2.0, 3.0), (10.0, 20.0, 30.0))
    values = np.zeros((*reversed(ct.GetSize()), 3), dtype=np.float32)
    values[0, 0, 0, 0] = np.nan
    field = _field(ct.GetSize(), ct.GetSpacing(), ct.GetOrigin(), values)
    with pytest.raises(InferenceError):
        _assert_native_geometry(field, ct, "001")


def test_verify_field_against_ct_reports_magnitudes():
    ct = _image((4, 5, 6), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    values = np.zeros((*reversed(ct.GetSize()), 3), dtype=np.float32)
    values[..., 0] = 3.0
    values[..., 1] = 4.0
    field = _field(ct.GetSize(), ct.GetSpacing(), ct.GetOrigin(), values)
    report = verify_field_against_ct(field, ct, "025")
    assert report["max_magnitude_mm"] == pytest.approx(5.0)
    assert report["mean_magnitude_mm"] == pytest.approx(5.0)


def test_verify_point_mapping_rejects_constant_field():
    """A pure constant offset is not a registration and must be rejected."""
    ct = _image((4, 5, 6), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    mr = _image((4, 5, 6), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    values = np.zeros((*reversed(ct.GetSize()), 3), dtype=np.float32)
    values[..., 2] = 12.0
    field = _field(ct.GetSize(), ct.GetSpacing(), ct.GetOrigin(), values)
    with pytest.raises(VerificationError):
        verify_point_mapping(field, ct, mr, "025")


def test_verify_point_mapping_detects_outside_mr():
    """A field that maps the CT centre outside the MR extent must be flagged."""
    ct = _image((4, 5, 6), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    mr = _image((4, 5, 6), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    values = np.zeros((*reversed(ct.GetSize()), 3), dtype=np.float32)
    values[..., 2] = 5000.0  # far outside the MR
    values[0, 0, 0, 2] = 4999.0  # but not constant
    field = _field(ct.GetSize(), ct.GetSpacing(), ct.GetOrigin(), values)
    report = verify_point_mapping(field, ct, mr, "025")
    assert report["mapped_inside_mr_extent"] is False


def test_verify_point_mapping_accepts_translation_into_mr():
    ct = _image((4, 5, 6), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    mr = _image((4, 5, 6), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    values = np.zeros((*reversed(ct.GetSize()), 3), dtype=np.float32)
    values[..., 0] = 1.0
    values[0, 0, 0, 0] = 1.5
    field = _field(ct.GetSize(), ct.GetSpacing(), ct.GetOrigin(), values)
    report = verify_point_mapping(field, ct, mr, "025")
    assert report["mapped_inside_mr_extent"] is True
    assert report["max_deviation_from_mean_mm"] > 0.0


def test_case_ids_cover_the_official_test_range():
    assert CASE_IDS == tuple(f"{case:03d}" for case in range(25, 33))
