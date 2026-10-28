"""Read-only diagnostic ceiling: what would a *perfect* rigid alignment score?

The rigid residual is nearly rigid (organ centroids agree to 1.5-15.2 mm), so
the whole score is governed by rigid accuracy.  This script measures the
ceiling of a rigid-only solution by applying the translation that best
aligns the labelled organ centroids.

WARNING: the translation here is derived from the *labels*.  It is a
diagnostic ceiling only.  It must never be used to build a submission, and it
is not an independent validation of anything.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medreglab.data.dataset import competition_masks_from_labels  # noqa: E402
from medreglab.evaluation.dsc_evaluator import competition_case_dice  # noqa: E402


def _pick(case_dir: str, want_ct: bool, want_seg: bool) -> str:
    hits = []
    for path in sorted(glob.glob(os.path.join(case_dir, "*"))):
        name = os.path.basename(path).lower()
        if ("ct" in name) != want_ct:
            continue
        is_seg = ("seg" in name) or ("label" in name)
        if is_seg == want_seg:
            hits.append(path)
    if len(hits) != 1:
        raise RuntimeError(f"ambiguous file selection in {case_dir}: {hits}")
    return hits[0]


def organ_centroids(image: sitk.Image, labels: np.ndarray, case_id: str):
    masks = competition_masks_from_labels(labels, case_id)
    out = {}
    for organ, mask in masks.items():
        if not mask.any():
            out[organ] = None
            continue
        idx = np.argwhere(mask).astype(np.float64)
        continuous = idx.mean(axis=0)[::-1]
        point = image.TransformContinuousIndexToPhysicalPoint(
            (float(continuous[0]), float(continuous[1]), float(continuous[2]))
        )
        out[organ] = np.array(point, dtype=np.float64)
    return out


def constant_field(fixed: sitk.Image, translation) -> sitk.Image:
    array = np.zeros((*reversed(fixed.GetSize()), 3), dtype=np.float32)
    array[..., 0], array[..., 1], array[..., 2] = translation
    field = sitk.GetImageFromArray(array, isVector=True)
    field.CopyInformation(fixed)
    return field


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "Datasets"
    cases = sys.argv[2:] or [f"{i:03d}" for i in range(17, 25)]
    scores = []
    for case_id in cases:
        case_dir = os.path.join(root, "train", case_id)
        ct = sitk.ReadImage(_pick(case_dir, True, False))
        mr = sitk.ReadImage(_pick(case_dir, False, False))
        ct_seg = sitk.ReadImage(_pick(case_dir, True, True))
        mr_seg = sitk.ReadImage(_pick(case_dir, False, True))
        ct_c = organ_centroids(ct, sitk.GetArrayFromImage(ct_seg), case_id)
        mr_c = organ_centroids(mr, sitk.GetArrayFromImage(mr_seg), case_id)
        deltas = [mr_c[o] - ct_c[o] for o in ("liver", "spleen", "kidney")
                  if ct_c.get(o) is not None and mr_c.get(o) is not None]
        translation = np.mean(deltas, axis=0)

        field = constant_field(ct, translation)
        transform = sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))
        warped = sitk.Resample(mr_seg, ct, transform, sitk.sitkNearestNeighbor,
                               0, mr_seg.GetPixelID())
        fixed_masks = competition_masks_from_labels(sitk.GetArrayFromImage(ct_seg), case_id)
        warped_masks = competition_masks_from_labels(sitk.GetArrayFromImage(warped), case_id)
        result = competition_case_dice(fixed_masks, warped_masks)
        scores.append(result["mean"])
        print(f"{case_id}: oracle-translation mean_dsc={result['mean']:.4f} "
              f"(L={result['liver']:.3f} S={result['spleen']:.3f} K={result['kidney']:.3f}) "
              f"t={np.round(translation, 1)}")
    if scores:
        arr = np.array(scores)
        k = max(1, int(np.ceil(len(arr) * 0.3)))
        print(f"\nORACLE ceiling: mean={arr.mean():.4f} "
              f"dsc30={np.mean(np.sort(arr)[:k]):.4f} (n_worst={k})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
