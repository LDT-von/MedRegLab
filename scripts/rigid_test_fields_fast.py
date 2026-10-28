"""Fast rigid candidates: RSGD only, no LBFGSB.

Why this exists
---------------
``estimate_rigid_transform_robust`` evaluates four candidates.  Two of them
use the LBFGSB optimiser, which is single threaded and, on the thick-slice test
cases (CT Z depth 130 and 142 slices against 35-46 for the others), needs
roughly three times as long as every other case.

Across the cases that have already been produced, the LBFGSB candidate was
**never** the one that won: the winners were always the two regular-step
gradient-descent candidates, ``mi_body_init_body`` and ``mi_body_init_geometry``.

This module therefore offers the same selection procedure with the LBFGSB
candidates removed.  It is used to produce the remaining fields without
waiting for the slow optimiser.  The result is not identical to the full
search, so the provenance of every field records which search produced it.

As everywhere else in this repository, the selection score is computed from
the images only.  **No organ label is read.**

Usage:
    python scripts/rigid_test_fields_fast.py --data Datasets \
        --output outputs/rigid_test_fast --cases 030 031 --jobs 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medreglab.evaluation.prealign import (  # noqa: E402
    PreAlignmentError,
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


def _initial_rigid(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    """Centre the moving image on the fixed image (body-centre init)."""
    transform = sitk.Euler3DTransform()
    fixed_center = fixed.TransformContinuousIndexToPhysicalPoint(
        [(size - 1) / 2.0 for size in fixed.GetSize()]
    )
    moving_center = moving.TransformContinuousIndexToPhysicalPoint(
        [(size - 1) / 2.0 for size in moving.GetSize()]
    )
    transform.SetCenter(fixed_center)
    transform.SetTranslation(
        tuple(m - f for m, f in zip(moving_center, fixed_center))
    )
    return transform


def _geometry_initial(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    """Initial transform that maps the moving grid origin onto the fixed one."""
    transform = sitk.Euler3DTransform()
    transform.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint((0.0, 0.0, 0.0)))
    transform.SetTranslation(
        tuple(
            m - f
            for m, f in zip(moving.GetOrigin(), fixed.GetOrigin())
        )
    )
    return transform


def _metric_transform(
    fixed: sitk.Image,
    moving: sitk.Image,
    transform: sitk.Transform,
    samples: float,
    seed: int,
) -> float:
    """Mattes MI between the resampled pair.  Lower is better."""
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(samples, seed)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetInitialTransform(transform, inPlace=False)
    return float(registration.Execute(fixed, moving))


def estimate_rigid_rsgd_only(
    fixed: sitk.Image,
    moving: sitk.Image,
    levels: int = 3,
    iterations: int = 200,
    samples: float = 0.2,
    seed: int = 42,
    selection_samples: float = 0.2,
    selection_seed: int = 1234,
) -> Tuple[sitk.Transform, dict]:
    """Same candidate selection as the robust estimator, without LBFGSB."""
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    def _mi(initial, iters, lev):
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(samples, seed)
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=2.0, minStep=1e-4, numberOfIterations=iters,
            relaxationFactor=0.5, gradientMagnitudeTolerance=1e-8,
        )
        registration.SetOptimizerScalesFromPhysicalShift()
        shrink = [4, 2, 1][:lev]
        registration.SetShrinkFactorsPerLevel(shrink)
        registration.SetSmoothingSigmasPerLevel([shrink_v * 0.5 for shrink_v in shrink])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration.SetInitialTransform(initial, inPlace=False)
        return registration.Execute(fixed_f, moving_f)

    candidates: List[Tuple[str, sitk.Transform]] = [
        ("init_body", _initial_rigid(fixed_f, moving_f)),
        ("init_geometry", _geometry_initial(fixed_f, moving_f)),
    ]
    for name, initial in list(candidates):
        try:
            candidates.append((f"mi_body_{name}", _mi(initial, iterations, levels)))
        except RuntimeError:
            continue

    scored = []
    for name, transform in candidates:
        try:
            value = _metric_transform(fixed_f, moving_f, transform, selection_samples, selection_seed)
        except RuntimeError as error:
            raise PreAlignmentError(f"metric evaluation failed for candidate {name}: {error}") from error
        scored.append({"name": name, "metric": value, "transform": transform})
    finite = [entry for entry in scored if math.isfinite(entry["metric"])]
    if not finite:
        raise PreAlignmentError("no rigid candidate produced a finite metric score")
    best = min(finite, key=lambda entry: entry["metric"])
    report = {
        "selected": best["name"],
        "selected_metric": best["metric"],
        "search": "rsgd_only",
        "candidates": [
            {"name": entry["name"], "metric": entry["metric"]} for entry in scored
        ],
    }
    return best["transform"], report


def _assert_submittable_field(field: sitk.Image, reference: sitk.Image, case_id: str) -> None:
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


def _generate_one_case(data_root: Path, output_dir: Path, case_id: str) -> Dict[str, Any]:
    started = time.perf_counter()
    case_dir = data_root / "test" / case_id
    ct_path = case_dir / "ct.nii"
    mr_path = case_dir / "mr.nii"
    for path in (ct_path, mr_path):
        if not path.is_file():
            raise FileNotFoundError(f"Case {case_id}: missing {path}")
    ct_hash_before = sha256_file(ct_path)
    mr_hash_before = sha256_file(mr_path)

    fixed = sitk.ReadImage(str(ct_path))
    moving = sitk.ReadImage(str(mr_path))

    transform, report = estimate_rigid_rsgd_only(fixed, moving)
    field = transform_to_displacement_field(transform, fixed)
    _assert_submittable_field(field, fixed, case_id)

    field_path = output_dir / "deformation_fields" / f"{case_id}_deformation_field.nii.gz"
    sitk.WriteImage(field, str(field_path), True)

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        print(f"refusing to overwrite nonempty output: {output_dir}", file=sys.stderr)
        return 1
    (output_dir / "deformation_fields").mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(_generate_one_case, Path(args.data), output_dir, case_id): case_id
            for case_id in args.cases
        }
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                entry = future.result()
            except Exception as error:  # noqa: BLE001
                errors.append({"case_id": case_id, "error": f"{type(error).__name__}: {error}"})
                print(f"[fast] {case_id}: FAILED {type(error).__name__}: {error}", flush=True)
                continue
            results.append(entry)
            print(
                f"[fast] {entry['case_id']}: field written, "
                f"|mean D|={entry['rigid_magnitude_mm']:.1f} mm, "
                f"selected={entry['rigid_selection']['selected']} "
                f"({entry['elapsed_seconds']:.1f}s)",
                flush=True,
            )

    results.sort(key=lambda entry: entry["case_id"])
    if errors:
        (output_dir / "errors.json").write_text(
            json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"failed cases: {[e['case_id'] for e in errors]}", file=sys.stderr)
        return 1
    (output_dir / "fast_summary.json").write_text(
        json.dumps({"cases": results}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"done: {len(results)} case(s) -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
