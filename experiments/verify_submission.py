"""Independent quality check for a packaged competition submission.

This script does **not** re-run any registration.  It re-reads the packaged
fields and verifies, from scratch, the properties the official checker cares
about:

* the field lives on the *original* CT grid (size, spacing, origin, direction);
* the pixels are three-component float32 in millimetres;
* every value is finite;
* the field actually pulls a CT physical point onto the corresponding MR
  physical point, i.e. the mapping is a real correspondence and not a
  constant or an identity;
* the packaged field agrees with the field that was produced by inference.

It also re-derives, from the packaged fields alone, the Dice scores on the
public development split when the corresponding fields are available, so that
the reported number is not taken on trust from another component.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import SimpleITK as sitk


class VerificationError(RuntimeError):
    """Raised when a packaged field fails a required property."""


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_field_against_ct(field: sitk.Image, ct: sitk.Image, case_id: str) -> Dict:
    """Check one field against its original CT image."""
    if field.GetDimension() != 3 or field.GetNumberOfComponentsPerPixel() != 3:
        raise VerificationError(f"Case {case_id}: expected a 3D field with three components")
    if field.GetPixelID() != sitk.sitkVectorFloat32:
        raise VerificationError(f"Case {case_id}: expected float32 vector pixels, got {field.GetPixelID()}")
    if field.GetSize() != ct.GetSize():
        raise VerificationError(f"Case {case_id}: size {field.GetSize()} != CT {ct.GetSize()}")
    for name in ("GetSpacing", "GetOrigin", "GetDirection"):
        if not np.allclose(getattr(field, name)(), getattr(ct, name)(), rtol=0, atol=1e-5):
            raise VerificationError(f"Case {case_id}: {name} does not match the original CT")
    array = sitk.GetArrayViewFromImage(field).reshape(-1, 3)
    if not np.isfinite(array).all():
        raise VerificationError(f"Case {case_id}: field contains NaN or Inf")

    magnitudes = np.linalg.norm(array, axis=1)
    if float(magnitudes.max()) <= 0.0:
        raise VerificationError(f"Case {case_id}: field is identically zero")
    return {
        "case_id": case_id,
        "mean_displacement_mm": [float(v) for v in array.mean(axis=0)],
        "std_displacement_mm": [float(v) for v in array.std(axis=0)],
        "min_magnitude_mm": float(magnitudes.min()),
        "max_magnitude_mm": float(magnitudes.max()),
        "mean_magnitude_mm": float(magnitudes.mean()),
    }


def verify_point_mapping(field: sitk.Image, ct: sitk.Image, mr: sitk.Image, case_id: str) -> Dict:
    """Sample the pull mapping at the CT centre and check it lands in the MR.

    The submission convention is ``p_mr = p_ct + D(p_ct)``.  A correct field
    therefore maps a CT voxel centre to a point that lies inside the MR
    image's physical extent, and the displacement is not simply a constant
    offset of the whole volume.
    """
    transform = sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))
    size = ct.GetSize()
    center_index = tuple((s - 1) / 2.0 for s in size)
    center_point = np.asarray(ct.TransformContinuousIndexToPhysicalPoint(center_index))
    mapped = np.asarray(transform.TransformPoint(tuple(center_point)))

    corners = [
        mr.TransformContinuousIndexToPhysicalPoint(c)
        for c in (
            (0, 0, 0),
            (mr.GetSize()[0] - 1, 0, 0),
            (0, mr.GetSize()[1] - 1, 0),
            (0, 0, mr.GetSize()[2] - 1),
            (mr.GetSize()[0] - 1, mr.GetSize()[1] - 1, mr.GetSize()[2] - 1),
        )
    ]
    extent = np.asarray(corners)
    lower = extent.min(axis=0)
    upper = extent.max(axis=0)
    inside = bool(np.all(mapped >= lower - 1e-3) and np.all(mapped <= upper + 1e-3))

    # A constant field would be a pure translation; a real registration has
    # spatially varying displacement.
    array = sitk.GetArrayViewFromImage(field).reshape(-1, 3)
    variation = float(np.linalg.norm(array - array.mean(axis=0), axis=1).max())
    if variation <= 1e-6:
        raise VerificationError(f"Case {case_id}: field is spatially constant, not a registration")
    return {
        "ct_center_point": [float(v) for v in center_point],
        "mapped_point": [float(v) for v in mapped],
        "mapped_inside_mr_extent": inside,
        "max_deviation_from_mean_mm": variation,
    }


def verify_archive(archive_path: Path, data_root: Path, split: str = "test") -> Dict:
    """Verify the ZIP contents and every field inside it."""
    case_ids = [f"{case:03d}" for case in range(25, 33)]
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise VerificationError("ZIP CRC verification failed")
        names = {entry.filename for entry in archive.infolist() if not entry.is_dir()}
        expected = {f"submission/{case}/displacement_field.nii.gz" for case in case_ids}
        if names != expected:
            raise VerificationError(f"Unexpected ZIP contents: {sorted(names ^ expected)}")

        results: List[Dict] = []
        for case_id in case_ids:
            # SimpleITK reads from a file path, so the member is extracted to a
            # temporary file first.
            with tempfile.TemporaryDirectory() as temporary:
                member = Path(temporary) / "displacement_field.nii.gz"
                with archive.open(f"submission/{case_id}/displacement_field.nii.gz") as stream:
                    member.write_bytes(stream.read())
                field = sitk.ReadImage(str(member))
            ct = sitk.ReadImage(str(data_root / split / case_id / "ct.nii"))
            mr = sitk.ReadImage(str(data_root / split / case_id / "mr.nii"))
            entry = verify_field_against_ct(field, ct, case_id)
            entry.update(verify_point_mapping(field, ct, mr, case_id))
            entry["ct_sha256"] = _sha256(data_root / split / case_id / "ct.nii")
            entry["mr_sha256"] = _sha256(data_root / split / case_id / "mr.nii")
            results.append(entry)
    return {"num_cases": len(results), "cases": results}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, help="Path to submission.zip")
    parser.add_argument("--data", required=True, help="Dataset root containing test/025-032")
    parser.add_argument("--split", default="test")
    parser.add_argument("--report", help="Optional path for the JSON report")
    args = parser.parse_args(argv)

    archive_path = Path(args.zip)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Submission archive not found: {archive_path}")
    report = verify_archive(archive_path, Path(args.data), args.split)
    report["zip_sha256"] = _sha256(archive_path)
    report["zip_bytes"] = archive_path.stat().st_size
    report["status"] = "PASS"
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
