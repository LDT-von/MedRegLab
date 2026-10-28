"""Run the organ-probability driven registration and export a competition field.

Why this exists
---------------
The intensity-only pipeline in ``experiments/infer_competition.py`` optimises a
displacement field against a cross-modal intensity similarity.  On this dataset
that similarity is poorly conditioned: the CT and MR acquisitions of cases
017-032 live in *different physical frames* (their origins differ by hundreds
of millimetres), so a single gradient search either never leaves the flat part
of the metric or converges to a wrong basin.

The approach implemented here fixes the frame first and then optimises against
*organ evidence* instead of raw intensity:

1.  A single-modality organ segmentation network predicts liver / spleen /
    kidney probability maps for the CT and for the MR of the same case.  The
    networks are trained on the public training split only; the hidden test
    labels are never read.
2.  The organ centroids give the initial translation that brings the two
    anatomies on top of each other.
3.  A coarse-to-fine search refines the transform against a modality-invariant
    MIND feature representation, with the organ probabilities added as extra
    feature channels.
4.  A dense displacement field is optimised on top of that initialisation.

The resulting field is written in exactly the format the competition expects:
a three-component float32 field in millimetres on the *original CT grid*,
following the pull convention ``p_mr = p_ct + D(p_ct)``.

The segmentation network and the search implementation live in a sibling
project (``ctmr``); this module is the MedRegLab-side driver that runs them,
verifies the output geometry and records provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import SimpleITK as sitk

CASE_IDS = tuple(f"{case:03d}" for case in range(25, 33))


class InferenceError(RuntimeError):
    """Raised when the registration cannot produce a trustworthy field."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_native_geometry(field: sitk.Image, reference: sitk.Image, case_id: str) -> None:
    """The submission field must match the original CT grid exactly."""
    if field.GetDimension() != 3 or field.GetNumberOfComponentsPerPixel() != 3:
        raise InferenceError(f"Case {case_id}: field must have three vector components")
    if field.GetSize() != reference.GetSize():
        raise InferenceError(f"Case {case_id}: field size {field.GetSize()} != CT {reference.GetSize()}")
    for name in ("GetSpacing", "GetOrigin", "GetDirection"):
        if not np.allclose(
            getattr(field, name)(), getattr(reference, name)(), rtol=0, atol=1e-5
        ):
            raise InferenceError(f"Case {case_id}: field {name} does not match the original CT")
    if not np.isfinite(sitk.GetArrayViewFromImage(field)).all():
        raise InferenceError(f"Case {case_id}: field contains NaN or Inf")


def run_organ_guided_registration(
    data_root: Path,
    output_dir: Path,
    engine_root: Path,
    prediction_dir: Path,
    split: str = "test",
    case_ids: Sequence[str] = CASE_IDS,
    gpu: int = 0,
    python_executable: Optional[str] = None,
    settings: Sequence[str] = (
        "seg_source=pred",
        "init=seg",
        "seg_weight=1",
        "affine_mode=none",
        "io_lambda=10",
    ),
    timeout_seconds: Optional[float] = None,
    extra_settings: Sequence[str] = (),
) -> Dict:
    """Run the organ-probability driven registration and export native fields.

    The heavy lifting happens in the ``ctmr`` engine; this function drives it,
    then re-verifies every produced field against the original CT geometry
    before it is accepted.
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    engine_root = Path(engine_root)
    prediction_dir = Path(prediction_dir)
    if not engine_root.is_dir():
        raise InferenceError(f"Registration engine not found: {engine_root}")
    if not prediction_dir.is_dir():
        raise InferenceError(f"Organ probability maps not found: {prediction_dir}")
    python_executable = python_executable or sys.executable

    output_dir.mkdir(parents=True, exist_ok=True)
    field_dir = output_dir / "deformation_fields"
    field_dir.mkdir(parents=True, exist_ok=True)

    engine_out = (output_dir / "engine").resolve()
    command = [
        python_executable,
        "-m",
        "ctmr.v2.register",
        "run",
        "--split",
        split,
        "--cases",
        "-".join([case_ids[0], case_ids[-1]]),
        "--out",
        str(engine_out),
        "--gpu",
        str(gpu),
        "--save-fields",
        "--set",
        f"pred_dir={prediction_dir}",
        *settings,
        *extra_settings,
    ]
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        f"{engine_root}{os.pathsep}{existing}" if existing else str(engine_root)
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(engine_root),
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    elapsed = round(time.perf_counter() - started, 1)
    (output_dir / "engine_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "engine_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise InferenceError(
            f"Registration engine failed with exit code {completed.returncode}; "
            f"see {output_dir / 'engine_stderr.log'}"
        )

    # Collect and verify the fields the engine produced.
    produced = engine_out / "fields"
    records: List[Dict] = []
    for case_id in case_ids:
        source = produced / case_id / "displacement_field.nii.gz"
        if not source.is_file():
            raise InferenceError(f"Registration engine produced no field for case {case_id}")
        ct_path = data_root / split / case_id / "ct.nii"
        mr_path = data_root / split / case_id / "mr.nii"
        if not ct_path.is_file() or not mr_path.is_file():
            raise InferenceError(f"Case {case_id}: missing CT or MR input")
        field = sitk.ReadImage(str(source))
        reference = sitk.ReadImage(str(ct_path))
        _assert_native_geometry(field, reference, case_id)

        destination = field_dir / f"{case_id}_deformation_field.nii.gz"
        sitk.WriteImage(field, str(destination), True)
        array = sitk.GetArrayViewFromImage(field).reshape(-1, 3)
        records.append({
            "case_id": case_id,
            "field_path": str(destination.relative_to(output_dir)),
            "field_sha256": _sha256(destination),
            "ct_input_sha256": _sha256(ct_path),
            "mr_input_sha256": _sha256(mr_path),
            "fixed_size_xyz": list(reference.GetSize()),
            "fixed_spacing": list(reference.GetSpacing()),
            "fixed_origin": list(reference.GetOrigin()),
            "mean_displacement_mm": [float(v) for v in array.mean(axis=0)],
            "max_displacement_mm": float(np.linalg.norm(array, axis=1).max()),
        })

    summary = {
        "model": "organ_guided_registration",
        "baseline": "organ_guided_registration",
        "split": split,
        "num_processed": len(records),
        "num_errors": 0,
        "deform_dir": str(field_dir),
        "field_grid": "original fixed CT",
        "field_units": "mm",
        "field_components": "physical LPS x,y,z",
        "transform": "pull: fixed CT physical point -> moving MR physical point",
        "pre_alignment": "organ-probability centroid translation + MIND coarse-to-fine search",
        "engine": str(engine_root),
        "engine_elapsed_seconds": elapsed,
        "cases": records,
    }
    (output_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "status": "OK",
        "num_cases": len(records),
        "elapsed_seconds": elapsed,
    }, ensure_ascii=False), flush=True)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset root containing test/025-032")
    parser.add_argument("--output", required=True, help="Output directory for the fields")
    parser.add_argument("--engine", required=True, help="Path to the ctmr registration engine")
    parser.add_argument("--predictions", required=True, help="Directory with organ probability maps")
    parser.add_argument("--split", default="test")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args(argv)

    run_organ_guided_registration(
        data_root=Path(args.data),
        output_dir=Path(args.output),
        engine_root=Path(args.engine),
        prediction_dir=Path(args.predictions),
        split=args.split,
        gpu=args.gpu,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
