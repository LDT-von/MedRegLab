"""Generate rigidly pre-aligned displacement fields for the eight test cases.

Why a separate script
---------------------
The competition submission needs one physical pull field per case:

    p_mr = p_ct + D(p_ct),      D on the original CT grid, in millimetres.

Two producers of such a field already exist in this repository:

* ``experiments/infer_competition.py`` runs the deformable ``convex_3d``
  optimizer.  On the development set that optimizer scored a *lower* mean DSC
  than doing nothing on the cases whose CT and MR grids describe different
  physical frames, because the model has to learn a rigid error it cannot
  represent well on a resized grid.
* ``scripts/baseline_test_fields.py`` writes baseline B (``phi = 0`` pushed
  through the geometric mapping), which carries the CT/MR geometry but no
  anatomical alignment at all.

This script adds the missing third producer: an **image-driven rigid**
pre-alignment.  A Mattes mutual-information rigid search estimates a CT->MR
transform from the images alone; the transform is then rasterised into the
physical pull field the submission format expects.

No organ label is read anywhere in this script.  Labels are used only for
scoring in ``medreglab.evaluation.dev_eval``.

Output layout
-------------
The directory matches what ``run_competition_inference`` produces, so
``experiments/package_submission.py`` can package it unchanged::

    <output>/deformation_fields/025_deformation_field.nii.gz
    ...
    <output>/deformation_fields/032_deformation_field.nii.gz
    <output>/inference_summary.json

The script fails closed: if any case cannot be processed, no
``inference_summary.json`` is written and the packaging step refuses to run.

Usage:
    python scripts/rigid_test_fields.py --data Datasets --output outputs/rigid_test
    python experiments/package_submission.py --inference outputs/rigid_test \
        --data Datasets --output outputs/submission_rigid
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medreglab.evaluation.prealign import (  # noqa: E402
    estimate_rigid_transform_robust,
    field_statistics,
    transform_to_displacement_field,
)

CASE_IDS = tuple(f"{case:03d}" for case in range(25, 33))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_submittable_field(field: sitk.Image, reference: sitk.Image, case_id: str) -> None:
    """Fail closed unless the field satisfies the packaging preconditions."""
    if field.GetDimension() != 3 or field.GetNumberOfComponentsPerPixel() != 3:
        raise ValueError(f"Case {case_id}: field must have three vector components")
    if field.GetPixelID() != sitk.sitkVectorFloat32:
        raise ValueError(f"Case {case_id}: field must use float32 vector pixels")
    for getter in ("GetSize", "GetSpacing", "GetOrigin", "GetDirection"):
        if getattr(field, getter)() != getattr(reference, getter)():
            raise ValueError(
                f"Case {case_id}: field {getter} does not exactly match the original CT"
            )
    if not np.isfinite(sitk.GetArrayViewFromImage(field)).all():
        raise ValueError(f"Case {case_id}: field contains NaN or Inf")


def _test_case_paths(data_root: Path, case_id: str) -> Dict[str, Path]:
    """Locate the CT and MR volumes of one test case."""
    case_dir = data_root / "test" / case_id
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case {case_id}: no directory at {case_dir}")
    ct = case_dir / "ct.nii"
    mr = case_dir / "mr.nii"
    if not ct.is_file():
        raise FileNotFoundError(f"Case {case_id}: missing {ct}")
    if not mr.is_file():
        raise FileNotFoundError(f"Case {case_id}: missing {mr}")
    return {"ct": ct, "mr": mr}


def _generate_one_case(
    data_root: Path,
    output_dir: Path,
    case_id: str,
) -> Dict[str, Any]:
    """Estimate one rigid field.  Runs in a worker process."""
    started = time.perf_counter()
    paths = _test_case_paths(data_root, case_id)
    ct_path, mr_path = paths["ct"], paths["mr"]
    ct_hash_before = sha256_file(ct_path)
    mr_hash_before = sha256_file(mr_path)

    fixed = sitk.ReadImage(str(ct_path))
    moving = sitk.ReadImage(str(mr_path))

    # Image-driven only: no organ label is opened in this function.
    transform, report = estimate_rigid_transform_robust(fixed, moving)
    field = transform_to_displacement_field(transform, fixed)
    _assert_submittable_field(field, fixed, case_id)

    field_path = output_dir / "deformation_fields" / f"{case_id}_deformation_field.nii.gz"
    sitk.WriteImage(field, str(field_path), True)

    # The recorded hashes must describe the inputs that produced the field.
    if sha256_file(ct_path) != ct_hash_before or sha256_file(mr_path) != mr_hash_before:
        raise RuntimeError("CT or MR input changed while the field was generated")

    stats = field_statistics(field)
    return {
        "case_id": case_id,
        "fixed_size_xyz": list(fixed.GetSize()),
        "moving_size_xyz": list(moving.GetSize()),
        "fixed_spacing": [float(v) for v in fixed.GetSpacing()],
        "moving_spacing": [float(v) for v in moving.GetSpacing()],
        "fixed_origin": [float(v) for v in fixed.GetOrigin()],
        "moving_origin": [float(v) for v in moving.GetOrigin()],
        "ct_input_sha256": ct_hash_before,
        "mr_input_sha256": mr_hash_before,
        "field_sha256": sha256_file(field_path),
        "field_path": str(field_path.relative_to(output_dir)),
        "mean_displacement_mm": stats["mean_displacement_mm"],
        "std_displacement_mm": stats["std_displacement_mm"],
        "rigid_magnitude_mm": stats["rigid_magnitude_mm"],
        "max_displacement_mm": stats["max_displacement_mm"],
        "rigid_selection": report,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def generate_rigid_fields(
    data_root: Path,
    output_dir: Path,
    split: str = "test",
    case_ids: Optional[Sequence[str]] = None,
    jobs: int = 1,
) -> Dict[str, Any]:
    """Write rigidly aligned fields plus an ``inference_summary.json``."""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Refusing to overwrite a nonempty output directory: {output_dir}")

    wanted = list(case_ids) if case_ids else list(CASE_IDS)
    field_dir = output_dir / "deformation_fields"
    field_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    if jobs > 1 and len(wanted) > 1:
        # One process per case: SimpleITK's rigid search is CPU bound and the
        # eight test cases are independent, so they run concurrently.
        with ProcessPoolExecutor(max_workers=min(jobs, len(wanted))) as pool:
            futures = {
                pool.submit(_generate_one_case, data_root, output_dir, case_id): case_id
                for case_id in wanted
            }
            for future in as_completed(futures):
                case_id = futures[future]
                try:
                    entry = future.result()
                except Exception as error:  # noqa: BLE001 - recorded and reported
                    errors.append({"case_id": case_id, "error": f"{type(error).__name__}: {error}"})
                    print(f"[rigid] {case_id}: FAILED {type(error).__name__}: {error}", flush=True)
                    continue
                results.append(entry)
                print(
                    f"[rigid] {entry['case_id']}: field written, "
                    f"|mean D|={entry['rigid_magnitude_mm']:.1f} mm, "
                    f"max |D|={entry['max_displacement_mm']:.1f} mm, "
                    f"selected={entry['rigid_selection']['selected']} "
                    f"({entry['elapsed_seconds']:.1f}s)",
                    flush=True,
                )
    else:
        for case_id in wanted:
            try:
                entry = _generate_one_case(data_root, output_dir, case_id)
            except Exception as error:  # noqa: BLE001 - recorded and reported
                errors.append({"case_id": case_id, "error": f"{type(error).__name__}: {error}"})
                print(f"[rigid] {case_id}: FAILED {type(error).__name__}: {error}", flush=True)
                continue
            results.append(entry)
            print(
                f"[rigid] {entry['case_id']}: field written, "
                f"|mean D|={entry['rigid_magnitude_mm']:.1f} mm, "
                f"max |D|={entry['max_displacement_mm']:.1f} mm, "
                f"selected={entry['rigid_selection']['selected']} "
                f"({entry['elapsed_seconds']:.1f}s)",
                flush=True,
            )

    results.sort(key=lambda entry: entry["case_id"])
    if errors:
        (output_dir / "errors.json").write_text(
            json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise RuntimeError(
            f"Rigid generation failed for {len(errors)} case(s): "
            f"{[entry['case_id'] for entry in errors]}. "
            "No inference_summary.json was written, so packaging will refuse to run."
        )

    summary = {
        "model": "rigid_prealign",
        "baseline": "rigid_prealign",
        "split": split,
        "num_processed": len(results),
        "num_errors": 0,
        "deform_dir": str(field_dir),
        "max_iter": 0,
        "target_size_dhw": [],
        "field_grid": "original fixed CT",
        "field_units": "mm",
        "field_components": "physical LPS x,y,z",
        "transform": "pull: fixed CT physical point -> moving MR physical point",
        "pre_alignment": "rigid image-driven (Mattes MI), multi-candidate selection",
        "cases": results,
    }
    (output_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset root")
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument("--split", default="test", help="Split to process")
    parser.add_argument(
        "--cases", nargs="+", default=None,
        help="Optional subset of case IDs (default: 025 through 032)",
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Number of cases to process concurrently (the rigid search is CPU bound)",
    )
    args = parser.parse_args()

    try:
        summary = generate_rigid_fields(
            data_root=Path(args.data),
            output_dir=Path(args.output),
            split=args.split,
            case_ids=args.cases,
            jobs=args.jobs,
        )
    except (RuntimeError, ValueError, FileExistsError, FileNotFoundError) as error:
        print(f"rigid generation failed: {error}", file=sys.stderr)
        return 1

    print()
    print(f"cases written: {summary['num_processed']} (errors: {summary['num_errors']})")
    print(f"fields: {Path(args.output) / 'deformation_fields'}")
    print(f"summary: {Path(args.output) / 'inference_summary.json'}")
    print("next: python experiments/package_submission.py "
          f"--inference {args.output} --data {args.data} --output <NEW_DIR>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
