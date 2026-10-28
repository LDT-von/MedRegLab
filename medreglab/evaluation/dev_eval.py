"""Development-set evaluation harness for the convex_3d per-case optimizer.

This module is deliberately separate from ``experiments/infer_competition.py``:
it evaluates a *development set* (the labelled ``train`` split) instead of
producing a competition submission, and it compares several fixed
registration configurations against each other on exactly the same cases.

Two different baselines are exposed and must not be conflated:

``zero_physical``
    The physical displacement field ``D(p) = 0`` on the original CT grid.
    This resamples the MR straight onto the CT grid with no motion at all.

``zero_model_identity``
    The identity field ``phi = 0`` on the model's *resized* grid, pushed
    through :func:`medreglab.utils.submission.displacement_to_fixed_grid`.
    Because that conversion also carries the geometric mapping between the CT
    and MR grids (size, spacing, origin and direction differences), the
    resulting physical field is generally **not** zero. It only reduces to
    ``zero_physical`` when the two modalities share the same geometry.

Scoring always happens on the original CT grid with the same code path the
submission uses, so the numbers are directly comparable with the official
protocol: three organs (liver, spleen and combined kidney), no smoothing
term, an empty CT organ scores 1.0, and DSC30 keeps
``ceil(total * 0.3)`` cases.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch

from medreglab.data import CompetitionDataset, competition_masks_from_labels
from medreglab.evaluation import competition_case_dice
from medreglab.evaluation.prealign import (
    compose_rigid_and_deformable,
    estimate_rigid_transform,
    normalize_ct,
    normalize_mr,
    resample_to_fixed_grid,
    transform_to_displacement_field,
)
from medreglab.utils.submission import displacement_to_fixed_grid, grid_folding_statistics

ORGANS = ("liver", "spleen", "kidney")


class EvaluationError(RuntimeError):
    """Raised when the evaluation cannot produce a trustworthy result."""


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalConfig:
    """One registration configuration.

    The four optimisation knobs mirror ``Simple3DRegistration`` exactly, and
    the effective values are recorded in the report so that a silently
    ignored parameter cannot go unnoticed.
    """

    name: str
    max_iter: int
    lr: float
    tv_weight: float
    similarity: str
    mode: str = "model"  # "model" | "zero_physical" | "zero_model_identity"
                         # | "prealign" | "prealign_model"

    def validate(self) -> None:
        if self.mode in ("model", "prealign_model"):
            if not isinstance(self.max_iter, int) or isinstance(self.max_iter, bool) or self.max_iter < 1:
                raise EvaluationError(f"{self.name}: max_iter must be a positive integer")
            if not math.isfinite(self.lr) or self.lr <= 0:
                raise EvaluationError(f"{self.name}: lr must be finite and positive")
            if not math.isfinite(self.tv_weight) or self.tv_weight < 0:
                raise EvaluationError(f"{self.name}: tv_weight must be finite and non-negative")
            if self.similarity not in ("mse", "ncc"):
                raise EvaluationError(f"{self.name}: similarity must be 'mse' or 'ncc'")
        elif self.mode not in ("zero_physical", "zero_model_identity", "prealign"):
            raise EvaluationError(f"{self.name}: unknown mode {self.mode!r}")


#: The configurations compared by :func:`run_configurations`. They are fixed
#: before any run so that results are not selected after the fact.
DEFAULT_CONFIGS: Tuple[EvalConfig, ...] = (
    EvalConfig("A_zero_physical", 0, 0.0, 0.0, "mse", mode="zero_physical"),
    EvalConfig("B_zero_model_identity", 0, 0.0, 0.0, "mse", mode="zero_model_identity"),
    EvalConfig("C1_current", 200, 0.1, 0.01, "mse", mode="model"),
    EvalConfig("C2_tv005", 200, 0.1, 0.05, "mse", mode="model"),
    EvalConfig("C3_tv005_lr005", 200, 0.05, 0.05, "mse", mode="model"),
    EvalConfig("C4_iter100", 100, 0.1, 0.01, "mse", mode="model"),
)

#: Configurations that add an image-driven rigid pre-alignment in front of the
#: deformable optimizer.  ``P1_prealign`` is rigid only; ``P2_prealign_C1``
#: runs the same C1 deformable settings on the rigidly aligned pair.
PREALIGN_CONFIGS: Tuple[EvalConfig, ...] = (
    EvalConfig("P1_prealign", 0, 0.0, 0.0, "mse", mode="prealign"),
    EvalConfig("P2_prealign_C1", 200, 0.1, 0.01, "mse", mode="prealign_model"),
)

#: Every configuration the harness can run.
ALL_CONFIGS: Tuple[EvalConfig, ...] = DEFAULT_CONFIGS + PREALIGN_CONFIGS


def config_by_name(name: str, configs: Sequence[EvalConfig] = ALL_CONFIGS) -> EvalConfig:
    for config in configs:
        if config.name == name:
            return config
    raise EvaluationError(
        f"Unknown configuration {name!r}; available: {[c.name for c in configs]}"
    )


# --------------------------------------------------------------------------
# Provenance helpers
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def git_dirty(repo_root: Path) -> Optional[bool]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


# --------------------------------------------------------------------------
# Case discovery
# --------------------------------------------------------------------------

@dataclass
class DevelopmentCase:
    """One labelled development case with its geometry and file hashes."""

    case_id: str
    fixed_path: Path
    moving_path: Path
    seg_fixed_path: Path
    seg_moving_path: Path
    fixed_size_xyz: Tuple[int, int, int]
    moving_size_xyz: Tuple[int, int, int]
    fixed_spacing: Tuple[float, float, float]
    moving_spacing: Tuple[float, float, float]
    fixed_origin: Tuple[float, float, float]
    moving_origin: Tuple[float, float, float]
    fixed_direction: Tuple[float, ...]
    moving_direction: Tuple[float, ...]
    fixed_sha256: str
    moving_sha256: str
    seg_fixed_sha256: str
    seg_moving_sha256: str
    ct_labels: List[int]
    mr_labels: List[int]


def _image_geometry(image: sitk.Image) -> Dict[str, Any]:
    return {
        "size_xyz": list(image.GetSize()),
        "spacing": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
    }


def collect_development_cases(
    data_root: Path,
    split: str = "train",
    require_labels: bool = True,
) -> List[DevelopmentCase]:
    """Collect labelled cases from a split, verifying geometry and hashing files."""
    dataset = CompetitionDataset(root=data_root, split=split)
    cases: List[DevelopmentCase] = []
    for index in range(len(dataset)):
        info = dataset.get_sample_info(index)
        case_id = str(info.get("case_id"))
        fixed_path = Path(info["fixed_path"])
        moving_path = Path(info["moving_path"])
        seg_fixed_path = info.get("seg_fixed_path")
        seg_moving_path = info.get("seg_moving_path")
        if require_labels and not (seg_fixed_path and seg_moving_path):
            raise EvaluationError(
                f"Case {case_id} has no paired segmentation; development evaluation "
                "requires both CT and MR labels"
            )
        if not (fixed_path.exists() and moving_path.exists()):
            raise EvaluationError(f"Case {case_id} is missing image files")
        seg_fixed_path = Path(seg_fixed_path)
        seg_moving_path = Path(seg_moving_path)
        if not (seg_fixed_path.exists() and seg_moving_path.exists()):
            raise EvaluationError(f"Case {case_id} is missing segmentation files")

        fixed = sitk.ReadImage(str(fixed_path))
        moving = sitk.ReadImage(str(moving_path))
        seg_fixed = sitk.ReadImage(str(seg_fixed_path))
        seg_moving = sitk.ReadImage(str(seg_moving_path))
        for label_image, image, modality in (
            (seg_fixed, fixed, "CT"), (seg_moving, moving, "MR"),
        ):
            if label_image.GetSize() != image.GetSize() or any(
                not np.allclose(getattr(label_image, name)(), getattr(image, name)(), rtol=0, atol=1e-5)
                for name in ("GetSpacing", "GetOrigin", "GetDirection")
            ):
                raise EvaluationError(
                    f"Case {case_id}: {modality} segmentation geometry does not match its image"
                )
        cases.append(
            DevelopmentCase(
                case_id=case_id,
                fixed_path=fixed_path,
                moving_path=moving_path,
                seg_fixed_path=seg_fixed_path,
                seg_moving_path=seg_moving_path,
                fixed_size_xyz=tuple(fixed.GetSize()),
                moving_size_xyz=tuple(moving.GetSize()),
                fixed_spacing=tuple(fixed.GetSpacing()),
                moving_spacing=tuple(moving.GetSpacing()),
                fixed_origin=tuple(fixed.GetOrigin()),
                moving_origin=tuple(moving.GetOrigin()),
                fixed_direction=tuple(fixed.GetDirection()),
                moving_direction=tuple(moving.GetDirection()),
                fixed_sha256=sha256_file(fixed_path),
                moving_sha256=sha256_file(moving_path),
                seg_fixed_sha256=sha256_file(seg_fixed_path),
                seg_moving_sha256=sha256_file(seg_moving_path),
                ct_labels=[int(v) for v in np.unique(sitk.GetArrayViewFromImage(seg_fixed)).tolist()],
                mr_labels=[int(v) for v in np.unique(sitk.GetArrayViewFromImage(seg_moving)).tolist()],
            )
        )
    if not cases:
        raise EvaluationError(f"No cases found for split {split!r} under {data_root}")
    return cases


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score_case(
    case: DevelopmentCase,
    field: sitk.Image,
) -> Dict[str, Any]:
    """Score one case on the original CT grid with the submission code path."""
    fixed_seg = sitk.ReadImage(str(case.seg_fixed_path))
    moving_seg = sitk.ReadImage(str(case.seg_moving_path))
    fixed = sitk.ReadImage(str(case.fixed_path))
    transform = sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))
    warped_seg = sitk.Resample(
        moving_seg, fixed, transform, sitk.sitkNearestNeighbor, 0, moving_seg.GetPixelID()
    )
    fixed_masks = competition_masks_from_labels(
        sitk.GetArrayViewFromImage(fixed_seg), case.case_id
    )
    warped_masks = competition_masks_from_labels(
        sitk.GetArrayViewFromImage(warped_seg), case.case_id
    )
    scores = competition_case_dice(fixed_masks, warped_masks)
    return {
        "liver": float(scores["liver"]),
        "spleen": float(scores["spleen"]),
        "kidney": float(scores["kidney"]),
        "mean": float(scores["mean"]),
    }


def zero_physical_field(fixed: sitk.Image) -> sitk.Image:
    """A constant-zero physical displacement field on the CT grid (baseline A)."""
    array = np.zeros((*reversed(fixed.GetSize()), 3), dtype=np.float32)
    field = sitk.GetImageFromArray(array, isVector=True)
    field.CopyInformation(fixed)
    return field


def zero_model_identity_field(
    fixed: sitk.Image,
    moving: sitk.Image,
    model_size_dhw: Sequence[int],
) -> sitk.Image:
    """The model's identity field pushed to the CT grid (baseline B)."""
    phi = torch.zeros((1, 3) + tuple(int(v) for v in model_size_dhw))
    return displacement_to_fixed_grid(phi, fixed, moving)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

@dataclass
class ConfigResult:
    name: str
    mode: str
    effective: Dict[str, Any]
    per_case: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        scored = [v for v in self.per_case.values() if "mean_dsc" in v]
        means = sorted(v["mean_dsc"] for v in scored)
        dsc30 = None
        num_worst = 0
        if means:
            num_worst = max(1, int(math.ceil(len(means) * 0.3)))
            dsc30 = float(np.mean(means[:num_worst]))
        folds = [v["fold_ratio"] for v in scored if "fold_ratio" in v]
        return {
            "name": self.name,
            "mode": self.mode,
            "effective_parameters": self.effective,
            "num_cases_scored": len(scored),
            "num_cases_failed": len(self.errors),
            "mean_dsc": float(np.mean([v["mean_dsc"] for v in scored])) if scored else None,
            "dsc30": dsc30,
            "dsc30_num_worst": num_worst,
            "mean_fold_ratio": float(np.mean(folds)) if folds else None,
            "max_fold_ratio": float(np.max(folds)) if folds else None,
            "errors": self.errors,
        }


def run_configuration(
    config: EvalConfig,
    cases: Sequence[DevelopmentCase],
    data_root: Path,
    output_dir: Path,
    target_size: Sequence[int],
    device: str,
    dataset: Optional[CompetitionDataset] = None,
) -> ConfigResult:
    """Run one configuration over every case and record per-case results."""
    config.validate()
    if config.mode in ("model", "prealign_model") and dataset is None:
        raise EvaluationError("A dataset is required for model configurations")

    result = ConfigResult(name=config.name, mode=config.mode, effective={})
    field_dir = output_dir / config.name / "deformation_fields"
    field_dir.mkdir(parents=True, exist_ok=True)

    model = None
    if config.mode in ("model", "prealign_model"):
        from medreglab import load_model

        model = load_model("convex_3d")
        model.max_iter = config.max_iter
        model.lr = config.lr
        model.tv_weight = config.tv_weight
        model.similarity = config.similarity
        model = model.to(device)
        model.eval()
        # Record what the model actually uses, not what we asked for.
        result.effective = {
            "max_iter": int(model.max_iter),
            "lr": float(model.lr),
            "tv_weight": float(model.tv_weight),
            "similarity": str(model.similarity),
            "target_size_dhw": [int(v) for v in target_size],
            "device": device,
        }
        if config.mode == "prealign_model":
            result.effective["pre_alignment"] = "rigid image-driven (Mattes MI)"
    else:
        result.effective = {
            "mode": config.mode,
            "target_size_dhw": [int(v) for v in target_size],
            "device": device,
        }
        if config.mode == "prealign":
            result.effective["pre_alignment"] = "rigid image-driven (Mattes MI)"

    case_index = {case.case_id: index for index, case in enumerate(cases)}
    for case in cases:
        case_start = time.perf_counter()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            fixed = sitk.ReadImage(str(case.fixed_path))
            moving = sitk.ReadImage(str(case.moving_path))
            rigid_field = None
            if config.mode == "zero_physical":
                field = zero_physical_field(fixed)
                fold_stats = None
            elif config.mode == "zero_model_identity":
                field = zero_model_identity_field(fixed, moving, target_size)
                fold_stats = None
            elif config.mode in ("prealign", "prealign_model"):
                # Rigid, image-driven (no labels are read here).
                transform = estimate_rigid_transform(fixed, moving)
                rigid_field = transform_to_displacement_field(transform, fixed)
                if config.mode == "prealign":
                    field = rigid_field
                    fold_stats = None
                else:
                    # Deformable refinement on the rigidly aligned pair.
                    aligned = resample_to_fixed_grid(moving, fixed, transform)
                    fixed_array = normalize_ct(sitk.GetArrayViewFromImage(fixed))
                    moving_array = normalize_mr(sitk.GetArrayViewFromImage(aligned))
                    fixed_tensor = torch.from_numpy(fixed_array)[None, None].float()
                    moving_tensor = torch.from_numpy(moving_array)[None, None].float()
                    if tuple(fixed_tensor.shape[2:]) != tuple(target_size):
                        fixed_tensor = torch.nn.functional.interpolate(
                            fixed_tensor, size=tuple(target_size),
                            mode="trilinear", align_corners=True,
                        )
                        moving_tensor = torch.nn.functional.interpolate(
                            moving_tensor, size=tuple(target_size),
                            mode="trilinear", align_corners=True,
                        )
                    fixed_tensor = fixed_tensor.to(device)
                    moving_tensor = moving_tensor.to(device)
                    with torch.enable_grad():
                        model_output = model(moving_tensor, fixed_tensor)
                    phi = model_output["phi"].detach().cpu()
                    # The deformable field now lives in the CT frame, because the
                    # model was optimised on the rigidly aligned pair.  Converting
                    # it with the *original* MR geometry would re-introduce the
                    # frame error that the rigid step just removed.
                    learned = displacement_to_fixed_grid(phi, fixed, aligned)
                    field = compose_rigid_and_deformable(transform, learned, fixed)
                    fold_stats = grid_folding_statistics(phi)
            else:
                sample = dataset[case_index[case.case_id]]
                if sample.get("spatial_augmented"):
                    raise EvaluationError(
                        f"Case {case.case_id}: spatially augmented tensors cannot be scored "
                        "against the original geometry"
                    )
                fixed_tensor = sample["fixed"].to(device)
                moving_tensor = sample["moving"].to(device)
                if fixed_tensor.dim() == 6:
                    fixed_tensor = fixed_tensor.squeeze(0)
                if moving_tensor.dim() == 6:
                    moving_tensor = moving_tensor.squeeze(0)
                with torch.enable_grad():
                    model_output = model(moving_tensor, fixed_tensor)
                phi = model_output["phi"].detach().cpu()
                field = displacement_to_fixed_grid(phi, fixed, moving)
                fold_stats = grid_folding_statistics(phi)

            field_path = field_dir / f"{case.case_id}_deformation_field.nii.gz"
            sitk.WriteImage(field, str(field_path), True)

            scores = score_case(case, field)
            entry: Dict[str, Any] = {
                "case_id": case.case_id,
                "mean_dsc": scores["mean"],
                "dsc": scores,
                "field_sha256": sha256_file(field_path),
                "field_path": str(field_path.relative_to(output_dir)),
                "fixed_size_xyz": list(case.fixed_size_xyz),
                "moving_size_xyz": list(case.moving_size_xyz),
                "fixed_spacing": list(case.fixed_spacing),
                "moving_spacing": list(case.moving_spacing),
                "fixed_origin": list(case.fixed_origin),
                "moving_origin": list(case.moving_origin),
            }
            if fold_stats is not None:
                entry.update(fold_stats)
            if config.mode in ("model", "prealign_model"):
                entry["loss"] = float(model_output["loss"])
            if rigid_field is not None:
                from medreglab.evaluation.prealign import field_statistics

                rigid_stats = field_statistics(rigid_field)
                entry["rigid_magnitude_mm"] = rigid_stats["rigid_magnitude_mm"]
                entry["rigid_mean_displacement_mm"] = rigid_stats["mean_displacement_mm"]
            entry["elapsed_seconds"] = round(time.perf_counter() - case_start, 3)
            if device == "cuda" and torch.cuda.is_available():
                entry["peak_gpu_gib"] = round(
                    torch.cuda.max_memory_allocated() / (1024 ** 3), 4
                )
            result.per_case[case.case_id] = entry
            print(
                f"[dev-eval] {config.name} case {case.case_id}: "
                f"mean_dsc={scores['mean']:.4f} "
                f"({entry['elapsed_seconds']:.1f}s"
                + (f", {entry['peak_gpu_gib']:.2f} GiB)" if "peak_gpu_gib" in entry else ")"),
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - recorded, then re-raised by the caller
            result.errors.append({"case_id": case.case_id, "error": f"{type(error).__name__}: {error}"})
            raise EvaluationError(
                f"Configuration {config.name} failed on case {case.case_id}: {error}"
            ) from error
    return result


def run_configurations(
    configs: Sequence[EvalConfig],
    data_root: Path,
    output_dir: Path,
    target_size: Sequence[int] = (128, 128, 128),
    split: str = "train",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run every configuration over the same cases and build the comparison."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise EvaluationError(f"Refusing to write into a nonempty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cases = collect_development_cases(Path(data_root), split=split)
    case_ids = [case.case_id for case in cases]

    manifest = {
        "purpose": "development-set evaluation (NOT an independent validation set)",
        "split": split,
        "data_root": str(data_root),
        "num_cases": len(cases),
        "case_ids": case_ids,
        "target_size_dhw": [int(v) for v in target_size],
        "device": device,
        "git_commit": git_commit(Path(__file__).resolve().parents[2]),
        "git_dirty": git_dirty(Path(__file__).resolve().parents[2]),
        "cases": [
            {
                "case_id": case.case_id,
                "fixed_path": str(case.fixed_path),
                "moving_path": str(case.moving_path),
                "seg_fixed_path": str(case.seg_fixed_path),
                "seg_moving_path": str(case.seg_moving_path),
                "fixed_size_xyz": list(case.fixed_size_xyz),
                "moving_size_xyz": list(case.moving_size_xyz),
                "fixed_spacing": list(case.fixed_spacing),
                "moving_spacing": list(case.moving_spacing),
                "fixed_origin": list(case.fixed_origin),
                "moving_origin": list(case.moving_origin),
                "fixed_direction": list(case.fixed_direction),
                "moving_direction": list(case.moving_direction),
                "fixed_sha256": case.fixed_sha256,
                "moving_sha256": case.moving_sha256,
                "seg_fixed_sha256": case.seg_fixed_sha256,
                "seg_moving_sha256": case.seg_moving_sha256,
                "ct_labels": case.ct_labels,
                "mr_labels": case.mr_labels,
            }
            for case in cases
        ],
        "configurations": [asdict(config) for config in configs],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    dataset = CompetitionDataset(root=data_root, split=split, target_size=tuple(target_size))
    results: List[ConfigResult] = []
    for config in configs:
        print(f"[dev-eval] running configuration {config.name}", flush=True)
        result = run_configuration(
            config, cases, Path(data_root), output_dir, target_size, device, dataset=dataset
        )
        results.append(result)
        payload = {
            "summary": result.summary(),
            "per_case": result.per_case,
            "errors": result.errors,
        }
        (output_dir / config.name / "per_case.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary = result.summary()
        print(
            f"[dev-eval] {config.name}: mean DSC={summary['mean_dsc']} "
            f"DSC30={summary['dsc30']} (n={summary['dsc30_num_worst']})",
            flush=True,
        )

    comparison = build_comparison(results, case_ids)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return comparison


def build_comparison(
    results: Sequence[ConfigResult],
    case_ids: Sequence[str],
) -> Dict[str, Any]:
    """Build the paired per-case comparison table."""
    by_name = {result.name: result for result in results}
    paired: Dict[str, Any] = {}
    for result in results:
        deltas = {}
        for case_id in case_ids:
            entry = result.per_case.get(case_id)
            if entry is None:
                continue
            base = by_name["A_zero_physical"].per_case.get(case_id) if "A_zero_physical" in by_name else None
            if base is not None and "mean_dsc" in base:
                deltas[case_id] = float(entry["mean_dsc"]) - float(base["mean_dsc"])
        paired[result.name] = {
            "summary": result.summary(),
            "per_case_deltas_vs_A_zero_physical": deltas,
            "per_case_mean_dsc": {
                case_id: result.per_case[case_id]["mean_dsc"]
                for case_id in case_ids
                if case_id in result.per_case
            },
        }
    return {
        "note": (
            "Development-set results only. Selecting a configuration on these cases is "
            "not an independent validation and the hidden test labels were never used."
        ),
        "dsc30_rule": "ceil(total * 0.3) lowest cases are averaged",
        "case_ids": list(case_ids),
        "configs": paired,
    }


def preflight(data_root: Path, split: str = "train") -> Dict[str, Any]:
    """Check that the development split is usable before any registration runs."""
    data_root = Path(data_root)
    cases = collect_development_cases(data_root, split=split)
    label_histogram: Dict[str, int] = {}
    for case in cases:
        for label in case.ct_labels:
            key = f"ct_label_{label}"
            label_histogram[key] = label_histogram.get(key, 0) + 1
    return {
        "data_root": str(data_root),
        "split": split,
        "num_cases": len(cases),
        "case_ids": [case.case_id for case in cases],
        "label_histogram": label_histogram,
        "cases": [
            {
                "case_id": case.case_id,
                "fixed_size_xyz": list(case.fixed_size_xyz),
                "moving_size_xyz": list(case.moving_size_xyz),
                "fixed_spacing": list(case.fixed_spacing),
                "moving_spacing": list(case.moving_spacing),
                "same_geometry": (
                    case.fixed_size_xyz == case.moving_size_xyz
                    and np.allclose(case.fixed_spacing, case.moving_spacing)
                    and np.allclose(case.fixed_origin, case.moving_origin)
                    and np.allclose(case.fixed_direction, case.moving_direction)
                ),
                "ct_labels": case.ct_labels,
                "mr_labels": case.mr_labels,
            }
            for case in cases
        ],
    }
