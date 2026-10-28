"""Synthetic checks for the baseline-B test-set field generator.

Everything here is synthetic: a fake unlabelled test split is written to a
temporary directory and the generator is driven over it.  No patient data is
read, and no real submission is produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from scripts.baseline_test_fields import generate_baseline_fields

_DIRECTION_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _make_unlabelled_test_split(root: Path, num_cases: int = 2) -> None:
    """Write a tiny test split with CT/MR pairs and no segmentation files."""
    test = root / "test"
    test.mkdir(parents=True, exist_ok=True)
    for index in range(num_cases):
        case = test / f"{index + 25:03d}"
        case.mkdir(exist_ok=True)
        ct = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
        mr = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6) * 0.5
        for name, array, origin in (
            ("ct.nii", ct, (0.0, 0.0, 0.0)),
            ("mr.nii", mr, (0.5, 0.0, 0.0)),
        ):
            image = sitk.GetImageFromArray(array)
            image.SetSpacing((1.0, 1.0, 1.0))
            image.SetOrigin(origin)
            image.SetDirection(_DIRECTION_IDENTITY)
            sitk.WriteImage(image, str(case / name), True)


def test_baseline_b_fields_are_written_without_labels(tmp_path):
    _make_unlabelled_test_split(tmp_path, num_cases=2)
    output = tmp_path / "out"
    summary = generate_baseline_fields(tmp_path, output, split="test", target_size=(4, 5, 6))

    assert summary["num_processed"] == 2
    assert summary["num_errors"] == 0
    assert summary["baseline"] == "B_zero_model_identity"
    assert summary["field_units"] == "mm"
    assert summary["field_grid"] == "original fixed CT"
    assert summary["transform"] == "pull: fixed CT physical point -> moving MR physical point"

    written = sorted(p.name for p in (output / "deformation_fields").glob("*.nii.gz"))
    assert written == ["025_deformation_field.nii.gz", "026_deformation_field.nii.gz"]

    saved = json.loads((output / "inference_summary.json").read_text())
    assert saved["num_processed"] == 2
    for record in saved["cases"]:
        assert len(record["field_sha256"]) == 64
        assert len(record["ct_input_sha256"]) == 64
        assert len(record["mr_input_sha256"]) == 64


def test_baseline_b_field_is_not_zero_when_geometry_differs(tmp_path):
    """Baseline B must carry the CT/MR geometric offset; that is its whole point."""
    _make_unlabelled_test_split(tmp_path, num_cases=1)
    output = tmp_path / "out"
    generate_baseline_fields(tmp_path, output, split="test", target_size=(4, 5, 6))

    field_path = output / "deformation_fields" / "025_deformation_field.nii.gz"
    field = sitk.ReadImage(str(field_path))
    array = sitk.GetArrayViewFromImage(field)
    assert field.GetNumberOfComponentsPerPixel() == 3
    # The MR origin is offset by 0.5 mm in x, so the field must not be zero.
    assert not np.allclose(array, 0.0)
    assert float(np.abs(array[..., 0]).mean()) == pytest.approx(0.5, abs=1e-4)


def test_baseline_b_field_geometry_matches_the_original_ct(tmp_path):
    _make_unlabelled_test_split(tmp_path, num_cases=1)
    output = tmp_path / "out"
    generate_baseline_fields(tmp_path, output, split="test", target_size=(4, 5, 6))

    field = sitk.ReadImage(str(output / "deformation_fields" / "025_deformation_field.nii.gz"))
    ct = sitk.ReadImage(str(tmp_path / "test" / "025" / "ct.nii"))
    for getter in ("GetSize", "GetSpacing", "GetOrigin", "GetDirection"):
        assert getattr(field, getter)() == getattr(ct, getter)()


def test_baseline_b_refuses_to_overwrite_and_reports_missing_cases(tmp_path):
    _make_unlabelled_test_split(tmp_path, num_cases=1)
    output = tmp_path / "out"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        generate_baseline_fields(tmp_path, output, split="test", target_size=(4, 5, 6))

    with pytest.raises(ValueError, match="not in the dataset"):
        generate_baseline_fields(
            tmp_path, tmp_path / "other", split="test",
            target_size=(4, 5, 6), case_ids=["099"],
        )
