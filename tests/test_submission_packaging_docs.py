"""Checks that the packaged submission describes the scheme it actually contains.

A baseline package must not claim to be a registered one, and the human-readable
README must stay in step with the JSON report.  Everything here is synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from experiments.package_submission import CASE_IDS, create_submission


def _write_ct(path: Path) -> None:
    """Write a tiny CT image with non-trivial geometry."""
    array = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.5, 2.0, 3.0))
    image.SetOrigin((11.0, -7.0, 42.0))
    sitk.WriteImage(image, str(path), True)


def _write_fake_inference(root: Path, model_name: str, baseline: str | None) -> Path:
    """Create a fake inference directory with eight native-grid fields."""
    inference = root / "inference"
    fields = inference / "deformation_fields"
    fields.mkdir(parents=True)

    data_root = root / "data"
    cases = []
    for case_id in CASE_IDS:
        case = data_root / "test" / case_id
        case.mkdir(parents=True)
        _write_ct(case / "ct.nii")
        _write_ct(case / "mr.nii")

        # A zero field on the CT grid is enough for the packaging checks.
        ct = sitk.ReadImage(str(case / "ct.nii"))
        field = sitk.GetImageFromArray(
            np.zeros((*reversed(ct.GetSize()), 3), dtype=np.float32), isVector=True
        )
        field.CopyInformation(ct)
        sitk.WriteImage(field, str(fields / f"{case_id}_deformation_field.nii.gz"), True)
        cases.append(
            {
                "case_id": case_id,
                "ct_input_sha256": "",
                "mr_input_sha256": "",
                "field_sha256": "",
            }
        )

    summary = {
        "model": model_name,
        "split": "test",
        "num_processed": 8,
        "num_errors": 0,
        "field_units": "mm",
        "field_grid": "original fixed CT",
        "field_components": "physical LPS x,y,z",
        "transform": "pull: fixed CT physical point -> moving MR physical point",
        "cases": cases,
    }
    if baseline is not None:
        summary["baseline"] = baseline
    (inference / "inference_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return inference


def _fill_hashes(root: Path, inference: Path) -> None:
    """Fill the recorded hashes so the packaging provenance checks pass."""
    from experiments.package_submission import sha256

    summary_path = inference / "inference_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for record in summary["cases"]:
        case = record["case_id"]
        record["ct_input_sha256"] = sha256(root / "data" / "test" / case / "ct.nii")
        record["mr_input_sha256"] = sha256(root / "data" / "test" / case / "mr.nii")
        record["field_sha256"] = sha256(
            inference / "deformation_fields" / f"{case}_deformation_field.nii.gz"
        )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def test_baseline_package_readme_names_the_baseline_not_the_model(tmp_path):
    """A baseline package must not describe itself as a registered submission."""
    inference = _write_fake_inference(tmp_path, "B_zero_model_identity", "B_zero_model_identity")
    _fill_hashes(tmp_path, inference)
    output = tmp_path / "submission_out"
    create_submission(inference, tmp_path / "data", output)

    readme = (output / "README.txt").read_text(encoding="utf-8")
    assert "B_zero_model_identity" in readme
    assert "模型：convex_3d" not in readme

    report = json.loads((output / "format_check.json").read_text(encoding="utf-8"))
    assert report["model"] == "B_zero_model_identity"


def test_registered_package_readme_names_the_model(tmp_path):
    inference = _write_fake_inference(tmp_path, "convex_3d", None)
    _fill_hashes(tmp_path, inference)
    output = tmp_path / "submission_out"
    create_submission(inference, tmp_path / "data", output)

    readme = (output / "README.txt").read_text(encoding="utf-8")
    assert "方案：convex_3d" in readme
    assert "基线B" not in readme


def test_readme_stays_outside_the_zip(tmp_path):
    """README, JSON report and SHA256 must not be submitted alongside the fields."""
    import zipfile

    inference = _write_fake_inference(tmp_path, "B_zero_model_identity", "B_zero_model_identity")
    _fill_hashes(tmp_path, inference)
    output = tmp_path / "submission_out"
    create_submission(inference, tmp_path / "data", output)

    with zipfile.ZipFile(output / "submission.zip") as archive:
        names = {entry.filename for entry in archive.infolist()}
    assert "README.txt" not in names
    assert "format_check.json" not in names
    assert "submission.zip.sha256" not in names
    assert names == {
        "submission/", *(f"submission/{case}/" for case in CASE_IDS),
        *(f"submission/{case}/displacement_field.nii.gz" for case in CASE_IDS),
    }
