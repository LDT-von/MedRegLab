"""Generate baseline-B displacement fields for the eight competition test cases.

What baseline B is
------------------
``B_zero_model_identity`` is the field the model reduces to **when it learns
nothing**: an identity field ``phi = 0`` on the model's resized grid, pushed
through :func:`medreglab.utils.submission.displacement_to_fixed_grid` onto the
original CT grid.

The resulting physical field is **not** zero.  It carries the CT/MR geometric
mapping (size, spacing, origin and direction differences) but **no** learned
deformation.  That is what distinguishes it from baseline A, which is a
literally zero physical field ``D(p) = 0`` on the CT grid.

Because baseline B needs no labels, it can be produced for the hidden test
split.  (Baseline A can too, but it is the weaker option: on the development
set A scored 0.4358 mean DSC versus B's 0.5096, because A scores exactly 0.0
on cases whose CT and MR physical extents do not overlap.)

Output layout
-------------
The directory is laid out exactly like ``run_competition_inference`` produces,
so ``experiments/package_submission.py`` can package it unchanged::

    <output>/deformation_fields/025_deformation_field.nii.gz
    ...
    <output>/deformation_fields/032_deformation_field.nii.gz
    <output>/inference_summary.json

The script fails closed: if any case cannot be processed, no
``inference_summary.json`` is written and the packaging step refuses to run.

Usage:
    python scripts/baseline_test_fields.py --data Datasets --output outputs/baseline_B_test
    python experiments/package_submission.py --inference outputs/baseline_B_test \
        --data Datasets --output outputs/submission_B_test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medreglab.data import CompetitionDataset  # noqa: E402
from medreglab.evaluation.dev_eval import zero_model_identity_field  # noqa: E402
from medreglab.evaluation.prealign import field_statistics  # noqa: E402

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


def generate_baseline_fields(
    data_root: Path,
    output_dir: Path,
    split: str = "test",
    target_size: Sequence[int] = (128, 128, 128),
    case_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Write baseline-B fields plus an ``inference_summary.json`` for packaging."""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Refusing to overwrite a nonempty output directory: {output_dir}")

    dataset = CompetitionDataset(root=data_root, split=split)
    if not len(dataset):
        raise ValueError(f"No cases found for split {split!r} under {data_root}")

    wanted = set(case_ids) if case_ids else None
    if wanted is not None:
        available = {
            str(dataset.get_sample_info(index).get("case_id"))
            for index in range(len(dataset))
        }
        missing = wanted - available
        if missing:
            raise ValueError(f"Requested cases are not in the dataset: {sorted(missing)}")

    field_dir = output_dir / "deformation_fields"
    field_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for index in range(len(dataset)):
        info = dataset.get_sample_info(index)
        case_id = str(info.get("case_id"))
        if wanted is not None and case_id not in wanted:
            continue
        try:
            ct_path = Path(info["fixed_path"])
            mr_path = Path(info["moving_path"])
            ct_hash_before = sha256_file(ct_path)
            mr_hash_before = sha256_file(mr_path)

            fixed = sitk.ReadImage(str(ct_path))
            moving = sitk.ReadImage(str(mr_path))
            field = zero_model_identity_field(fixed, moving, target_size)
            _assert_submittable_field(field, fixed, case_id)

            field_path = field_dir / f"{case_id}_deformation_field.nii.gz"
            sitk.WriteImage(field, str(field_path), True)

            # The recorded hashes must describe the inputs that produced the field.
            if sha256_file(ct_path) != ct_hash_before or sha256_file(mr_path) != mr_hash_before:
                raise RuntimeError("CT or MR input changed while the field was generated")

            stats = field_statistics(field)
            results.append(
                {
                    "case_id": case_id,
                    "fixed_size_xyz": list(fixed.GetSize()),
                    "moving_size_xyz": list(moving.GetSize()),
                    "ct_input_sha256": ct_hash_before,
                    "mr_input_sha256": mr_hash_before,
                    "field_sha256": sha256_file(field_path),
                    "field_path": str(field_path.relative_to(output_dir)),
                    "mean_displacement_mm": stats["mean_displacement_mm"],
                    "std_displacement_mm": stats["std_displacement_mm"],
                    "rigid_magnitude_mm": stats["rigid_magnitude_mm"],
                    "max_displacement_mm": stats["max_displacement_mm"],
                }
            )
            print(
                f"[baseline-B] {case_id}: field written, "
                f"|mean D|={stats['rigid_magnitude_mm']:.1f} mm, "
                f"max |D|={stats['max_displacement_mm']:.1f} mm",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - recorded and reported
            errors.append({"case_id": case_id, "error": f"{type(error).__name__}: {error}"})

    if errors:
        (output_dir / "errors.json").write_text(
            json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise RuntimeError(
            f"Baseline-B generation failed for {len(errors)} case(s): "
            f"{[entry['case_id'] for entry in errors]}. "
            "No inference_summary.json was written, so packaging will refuse to run."
        )

    summary = {
        "model": "B_zero_model_identity",
        "baseline": "B_zero_model_identity",
        "split": split,
        "num_processed": len(results),
        "num_errors": 0,
        "deform_dir": str(field_dir),
        "max_iter": 0,
        "target_size_dhw": [int(v) for v in target_size],
        "field_grid": "original fixed CT",
        "field_units": "mm",
        "field_components": "physical LPS x,y,z",
        "transform": "pull: fixed CT physical point -> moving MR physical point",
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
        "--target_size", type=int, nargs="+", default=[128, 128, 128],
        help="Model grid size (D H W); baseline B is independent of this value",
    )
    parser.add_argument(
        "--cases", nargs="+", default=None,
        help="Optional subset of case IDs",
    )
    args = parser.parse_args()

    try:
        summary = generate_baseline_fields(
            data_root=Path(args.data),
            output_dir=Path(args.output),
            split=args.split,
            target_size=tuple(args.target_size),
            case_ids=args.cases,
        )
    except (RuntimeError, ValueError, FileExistsError) as error:
        print(f"baseline-B generation failed: {error}", file=sys.stderr)
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
