"""End-to-end checks for the development-set evaluation harness.

Everything here is synthetic: a small fake dataset is written to a temporary
directory and the harness is driven over it. No patient data is touched.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from medreglab.evaluation.dev_eval import (
    DEFAULT_CONFIGS,
    EvalConfig,
    EvaluationError,
    build_comparison,
    collect_development_cases,
    config_by_name,
    preflight,
    run_configurations,
    score_case,
    zero_model_identity_field,
    zero_physical_field,
)


def _write_image(path: Path, array: np.ndarray, spacing, origin, direction) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    sitk.WriteImage(image, str(path), True)
    return image


def _make_dataset(root: Path, num_cases: int = 4, identical_geometry: bool = False) -> None:
    """Write a tiny labelled dataset with two organs and one kidney pair."""
    train = root / "train"
    train.mkdir(parents=True, exist_ok=True)
    direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    for index in range(num_cases):
        case = train / f"{index + 1:03d}"
        case.mkdir(exist_ok=True)
        ct = np.zeros((6, 5, 4), dtype=np.float32)
        mr = np.zeros((6, 5, 4), dtype=np.float32)
        ct[1:3, 1:3, 1:3] = 40.0
        mr[1:3, 1:3, 1:3] = 0.5
        ct_seg = np.zeros((6, 5, 4), dtype=np.uint8)
        mr_seg = np.zeros((6, 5, 4), dtype=np.uint8)
        ct_seg[1, 1, 1] = 1          # liver
        ct_seg[2, 2, 2] = 2          # spleen
        ct_seg[3, 3, 1] = 3          # kidney (left)
        ct_seg[4, 4, 2] = 4          # kidney (right)
        mr_seg[1, 1, 1] = 1
        mr_seg[2, 2, 2] = 2
        mr_seg[3, 3, 1] = 3
        mr_seg[4, 4, 2] = 4
        mr_origin = (0.0, 0.0, 0.0) if identical_geometry else (0.5, 0.0, 0.0)
        _write_image(case / "ct.nii.gz", ct, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), direction)
        _write_image(case / "mr.nii.gz", mr, (1.0, 1.0, 1.0), mr_origin, direction)
        _write_image(case / "seg_ct.nii.gz", ct_seg, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), direction)
        _write_image(case / "seg_mr.nii.gz", mr_seg, (1.0, 1.0, 1.0), mr_origin, direction)


def test_collect_development_cases_reports_geometry_and_hashes(tmp_path):
    _make_dataset(tmp_path, num_cases=3)
    cases = collect_development_cases(tmp_path, split="train")
    assert len(cases) == 3
    assert [case.case_id for case in cases] == ["001", "002", "003"]
    for case in cases:
        assert len(case.fixed_sha256) == 64
        assert len(case.seg_moving_sha256) == 64
        assert case.fixed_size_xyz == (4, 5, 6)
        assert case.moving_size_xyz == (4, 5, 6)
        assert set(case.ct_labels) == {0, 1, 2, 3, 4}


def test_preflight_flags_differing_geometry(tmp_path):
    _make_dataset(tmp_path, num_cases=2, identical_geometry=False)
    report = preflight(tmp_path, split="train")
    assert report["num_cases"] == 2
    assert all(not case["same_geometry"] for case in report["cases"])
    identical = preflight(tmp_path, split="train")
    assert identical["num_cases"] == 2


def test_zero_physical_and_model_identity_baselines_differ(tmp_path):
    """Baseline A (D=0) and baseline B (model identity) must not be conflated."""
    _make_dataset(tmp_path, num_cases=1, identical_geometry=False)
    case = collect_development_cases(tmp_path, split="train")[0]
    fixed = sitk.ReadImage(str(case.fixed_path))
    moving = sitk.ReadImage(str(case.moving_path))

    field_a = zero_physical_field(fixed)
    field_b = zero_model_identity_field(fixed, moving, (4, 5, 6))
    array_a = sitk.GetArrayViewFromImage(field_a)
    array_b = sitk.GetArrayViewFromImage(field_b)
    assert np.allclose(array_a, 0.0)
    assert not np.allclose(array_b, 0.0), "baseline B must carry the CT/MR geometric mapping"

    # Identical geometry collapses the two baselines onto each other.
    fixed_same = sitk.ReadImage(str(case.fixed_path))
    moving_same = sitk.ReadImage(str(case.moving_path))
    moving_same.SetOrigin(fixed_same.GetOrigin())
    field_b_same = zero_model_identity_field(fixed_same, moving_same, (4, 5, 6))
    array_b_same = sitk.GetArrayViewFromImage(field_b_same)
    assert np.allclose(array_b_same, 0.0, atol=1e-6)


def test_zero_physical_baseline_scores_perfectly_on_identical_geometry(tmp_path):
    _make_dataset(tmp_path, num_cases=1, identical_geometry=True)
    case = collect_development_cases(tmp_path, split="train")[0]
    fixed = sitk.ReadImage(str(case.fixed_path))
    scores = score_case(case, zero_physical_field(fixed))
    assert scores["liver"] == pytest.approx(1.0)
    assert scores["spleen"] == pytest.approx(1.0)
    assert scores["kidney"] == pytest.approx(1.0)
    assert scores["mean"] == pytest.approx(1.0)


def test_run_configurations_produces_paired_comparison(tmp_path):
    _make_dataset(tmp_path, num_cases=3)
    output = tmp_path / "out"
    configs = (
        EvalConfig("A_zero_physical", 0, 0.0, 0.0, "mse", mode="zero_physical"),
        EvalConfig("B_zero_model_identity", 0, 0.0, 0.0, "mse", mode="zero_model_identity"),
        EvalConfig("C1_current", 2, 0.1, 0.01, "mse", mode="model"),
    )
    comparison = run_configurations(
        configs=configs, data_root=tmp_path, output_dir=output,
        target_size=(8, 8, 8), split="train", device="cpu",
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["num_cases"] == 3
    assert manifest["case_ids"] == ["001", "002", "003"]
    assert manifest["split"] == "train"
    assert len(manifest["configurations"]) == 3

    for name in ("A_zero_physical", "B_zero_model_identity", "C1_current"):
        summary = comparison["configs"][name]["summary"]
        assert summary["num_cases_scored"] == 3
        assert summary["num_cases_failed"] == 0
        assert 0.0 <= summary["mean_dsc"] <= 1.0
        # ceil(3 * 0.3) == 1
        assert summary["dsc30_num_worst"] == 1
        assert summary["dsc30"] == pytest.approx(summary["mean_dsc"])

    # The model configuration records the parameters that were actually used.
    effective = comparison["configs"]["C1_current"]["summary"]["effective_parameters"]
    assert effective["max_iter"] == 2
    assert effective["lr"] == pytest.approx(0.1)
    assert effective["tv_weight"] == pytest.approx(0.01)
    assert effective["similarity"] == "mse"
    assert effective["target_size_dhw"] == [8, 8, 8]

    # Per-case deltas are present for every scored case.
    deltas = comparison["configs"]["C1_current"]["per_case_deltas_vs_A_zero_physical"]
    assert set(deltas) == {"001", "002", "003"}
    for case_id, value in deltas.items():
        assert isinstance(value, float)

    # The per-case file keeps the hashes and geometry.
    per_case = json.loads((output / "C1_current" / "per_case.json").read_text())
    assert set(per_case["per_case"]) == {"001", "002", "003"}
    for entry in per_case["per_case"].values():
        assert len(entry["field_sha256"]) == 64
        assert entry["fixed_size_xyz"] == [4, 5, 6]
        assert "fold_ratio" in entry


def test_run_configurations_refuses_nonempty_output(tmp_path):
    _make_dataset(tmp_path, num_cases=1)
    output = tmp_path / "out"
    output.mkdir()
    (output / "existing.txt").write_text("data")
    with pytest.raises(EvaluationError, match="nonempty"):
        run_configurations(
            configs=DEFAULT_CONFIGS[:1], data_root=tmp_path, output_dir=output,
            target_size=(8, 8, 8), split="train", device="cpu",
        )


def test_run_configurations_reports_missing_segmentation(tmp_path):
    _make_dataset(tmp_path, num_cases=2)
    # Remove one case's MR segmentation: the run must fail loudly.
    (tmp_path / "train" / "002" / "seg_mr.nii.gz").unlink()
    output = tmp_path / "out"
    with pytest.raises(EvaluationError):
        run_configurations(
            configs=DEFAULT_CONFIGS[:1], data_root=tmp_path, output_dir=output,
            target_size=(8, 8, 8), split="train", device="cpu",
        )


def test_dsc30_uses_ceil_over_effective_case_count():
    from medreglab.evaluation import DSC30_Scorer

    scorer = DSC30_Scorer()
    for index in range(24):
        scorer.add_case(f"{index:03d}", 0.5)
    assert scorer.compute()["num_worst"] == 8
    assert scorer.compute()["num_worst"] == math.ceil(24 * 0.3)


def test_build_comparison_reports_deltas():
    class Fake:
        def __init__(self, name, values):
            self.name = name
            self.mode = "model"
            self.effective = {}
            self.per_case = {
                case_id: {"mean_dsc": value} for case_id, value in values.items()
            }
            self.errors = []

        def summary(self):
            return {
                "name": self.name, "mode": self.mode, "effective_parameters": {},
                "num_cases_scored": len(self.per_case), "num_cases_failed": 0,
                "mean_dsc": float(np.mean([v["mean_dsc"] for v in self.per_case.values()])),
                "dsc30": None, "dsc30_num_worst": 0,
                "mean_fold_ratio": None, "max_fold_ratio": None, "errors": [],
            }

    base = Fake("A_zero_physical", {"001": 0.4, "002": 0.5})
    other = Fake("C1_current", {"001": 0.45, "002": 0.5})
    comparison = build_comparison([base, other], ["001", "002"])
    deltas = comparison["configs"]["C1_current"]["per_case_deltas_vs_A_zero_physical"]
    assert deltas["001"] == pytest.approx(0.05)
    assert deltas["002"] == pytest.approx(0.0)
