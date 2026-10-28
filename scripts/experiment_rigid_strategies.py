"""Experiment: which rigid strategy gets the CT/MR pair aligned best?

Every strategy is image-driven: no organ label is ever read.  Each strategy
produces a CT->MR transform, and the candidates are compared by a Mattes
mutual-information score computed on the full-resolution pair with a fixed
sampling seed, so the selection is reproducible and label-free.

The diagnostic prints, for reference only, the distance between the produced
transform and the organ-centroid "oracle" translation.  That number is a
diagnostic ceiling, not an input to any decision.
"""
from __future__ import annotations

import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medreglab.data.dataset import competition_masks_from_labels  # noqa: E402
from medreglab.evaluation.dsc_evaluator import competition_case_dice  # noqa: E402
from medreglab.evaluation.prealign import (  # noqa: E402
    _geometry_center,
    _initial_rigid,
    body_centroid,
    transform_to_displacement_field,
)

DIRECTION_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _pick(case_dir, want_ct, want_seg):
    hits = []
    for path in sorted(glob.glob(os.path.join(case_dir, "*"))):
        name = os.path.basename(path).lower()
        if ("ct" in name) != want_ct:
            continue
        is_seg = ("seg" in name) or ("label" in name)
        if is_seg == want_seg:
            hits.append(path)
    if len(hits) != 1:
        raise RuntimeError(f"ambiguous selection in {case_dir}: {hits}")
    return hits[0]


def _mask(image, modality):
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    if modality == "CT":
        return array > -300.0
    positive = array[array > 0]
    if positive.size == 0:
        return np.ones(array.shape, dtype=bool)
    return array > float(np.percentile(positive, 40.0))


def _run_mi(fixed, moving, initial, levels=3, iterations=200, samples=0.2,
            seed=42, optimizer="rsgd", mask=None):
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(samples, seed)
    reg.SetInterpolator(sitk.sitkLinear)
    if optimizer == "rsgd":
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=2.0, minStep=1e-4, numberOfIterations=iterations,
            relaxationFactor=0.5, gradientMagnitudeTolerance=1e-8,
        )
    else:
        reg.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-6, numberOfIterations=iterations,
            maximumNumberOfCorrections=10, maximumNumberOfFunctionEvaluations=2000,
            costFunctionConvergenceFactor=1e7,
        )
    reg.SetOptimizerScalesFromPhysicalShift()
    shrink = [4, 2, 1][:levels]
    reg.SetShrinkFactorsPerLevel(shrink)
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0][:levels])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=False)
    return reg.Execute(fixed_f, moving_f)


def _score_mi(fixed, moving, transform, seed=1234, samples=0.2):
    """Mattes MI between fixed and moving resampled through the transform.

    For Mattes mutual information SimpleITK returns the *negated* mutual
    information, so a lower value means a better match.
    """
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(samples, seed)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(transform, inPlace=False)
    try:
        value = float(reg.MetricEvaluate(fixed_f, moving_f))
    except RuntimeError:
        return float("inf")
    return value if np.isfinite(value) else float("inf")


def _translation_init(fixed, moving, kind):
    t = sitk.Euler3DTransform()
    t.SetCenter(_geometry_center(fixed))
    if kind == "body":
        fc, mc = body_centroid(fixed, "CT"), body_centroid(moving, "MR")
        if fc is not None and mc is not None:
            t.SetTranslation((mc - fc).tolist())
    elif kind == "geometry":
        offset = np.asarray(_geometry_center(moving)) - np.asarray(_geometry_center(fixed))
        t.SetTranslation(offset.tolist())
    return t


def _field_translation(fixed, transform):
    field = transform_to_displacement_field(transform, fixed)
    arr = sitk.GetArrayFromImage(field).astype(np.float64)
    return arr.reshape(-1, 3).mean(axis=0)


def _score_case(case_id, root="Datasets"):
    case_dir = os.path.join(root, "train", case_id)
    fixed = sitk.ReadImage(_pick(case_dir, True, False))
    moving = sitk.ReadImage(_pick(case_dir, False, False))
    ct_seg = sitk.ReadImage(_pick(case_dir, True, True))
    mr_seg = sitk.ReadImage(_pick(case_dir, False, True))

    # --- diagnostic reference (labels; never used for selection) ---
    def centroids(image, labels):
        masks = competition_masks_from_labels(labels, case_id)
        out = {}
        for organ, mask in masks.items():
            if not mask.any():
                continue
            idx = np.argwhere(mask).astype(np.float64)
            c = idx.mean(axis=0)[::-1]
            out[organ] = np.asarray(image.TransformContinuousIndexToPhysicalPoint(
                (float(c[0]), float(c[1]), float(c[2]))), dtype=np.float64)
        return out
    ct_c = centroids(fixed, sitk.GetArrayFromImage(ct_seg))
    mr_c = centroids(moving, sitk.GetArrayFromImage(mr_seg))
    oracle = np.mean([mr_c[o] - ct_c[o] for o in ("liver", "spleen", "kidney")
                      if o in ct_c and o in mr_c], axis=0)

    def dsc_of(transform):
        field = transform_to_displacement_field(transform, fixed)
        t = sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))
        warped = sitk.Resample(mr_seg, fixed, t, sitk.sitkNearestNeighbor, 0, mr_seg.GetPixelID())
        fm = competition_masks_from_labels(sitk.GetArrayFromImage(ct_seg), case_id)
        wm = competition_masks_from_labels(sitk.GetArrayFromImage(warped), case_id)
        return competition_case_dice(fm, wm)["mean"]

    candidates = {}
    candidates["init_body"] = _translation_init(fixed, moving, "body")
    candidates["init_geometry"] = _translation_init(fixed, moving, "geometry")

    def _try(name, fn):
        try:
            candidates[name] = fn()
        except RuntimeError as error:
            # A candidate that fails to optimise is simply dropped; the
            # remaining candidates are still scored.
            print(f"    [skip] {name}: {str(error).splitlines()[0][:90]}", flush=True)

    _try("mi_body", lambda: _run_mi(fixed, moving, _initial_rigid(fixed, moving)))
    _try("mi_geometry", lambda: _run_mi(fixed, moving, _translation_init(fixed, moving, "geometry")))
    _try("mi_identity", lambda: _run_mi(fixed, moving, _translation_init(fixed, moving, "none")))
    _try("lbfgs_body", lambda: _run_mi(fixed, moving, _initial_rigid(fixed, moving), optimizer="lbfgs"))
    _try("mi_body_fine", lambda: _run_mi(fixed, moving, _initial_rigid(fixed, moving), levels=4,
                                        iterations=300))

    rows = []
    for name, transform in candidates.items():
        err = float(np.linalg.norm(_field_translation(fixed, transform) - oracle))
        rows.append({
            "name": name,
            "dsc": dsc_of(transform),
            "err_mm": err,
            "metric": _score_mi(fixed, moving, transform),
        })
    best_metric = min(rows, key=lambda r: r["metric"])
    best_err = min(rows, key=lambda r: r["err_mm"])
    best_dsc = max(rows, key=lambda r: r["dsc"])
    return {
        "case_id": case_id,
        "rows": rows,
        "picked_by_metric": best_metric["name"],
        "picked_by_metric_dsc": best_metric["dsc"],
        "picked_by_metric_err": best_metric["err_mm"],
        "best_dsc": best_dsc["dsc"],
        "best_dsc_name": best_dsc["name"],
        "oracle_err": float(np.linalg.norm(oracle)),
    }


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "Datasets"
    cases = sys.argv[2:] or [f"{i:03d}" for i in range(17, 25)]
    with ProcessPoolExecutor(max_workers=min(8, len(cases))) as pool:
        results = list(pool.map(_score_case, cases))
    results.sort(key=lambda r: r["case_id"])

    names = [r["name"] for r in results[0]["rows"]]
    print(f"{'case':6s} " + " ".join(f"{n:>15s}" for n in names) + "   pick(metric)  pick->dsc  best")
    for res in results:
        by = {r["name"]: r for r in res["rows"]}
        cells = " ".join(f"{by[n]['dsc']:.3f}/{by[n]['err_mm']:5.0f}mm" for n in names)
        print(f"{res['case_id']:6s} {cells}   {res['picked_by_metric']:>13s} "
              f"{res['picked_by_metric_dsc']:.3f}   {res['best_dsc']:.3f}({res['best_dsc_name']})")
    print()
    for n in names:
        vals = [r for res in results for r in res["rows"] if r["name"] == n]
        print(f"  {n:16s} mean_dsc={np.mean([v['dsc'] for v in vals]):.4f} "
              f"mean_err={np.mean([v['err_mm'] for v in vals]):6.1f}mm")
    picked = [res["picked_by_metric_dsc"] for res in results]
    best = [res["best_dsc"] for res in results]
    print(f"\n  selection by metric : mean={np.mean(picked):.4f} "
          f"dsc30={np.mean(np.sort(picked)[:max(1,int(np.ceil(len(picked)*0.3)))]):.4f}")
    print(f"  oracle best per case : mean={np.mean(best):.4f} "
          f"dsc30={np.mean(np.sort(best)[:max(1,int(np.ceil(len(best)*0.3)))]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
