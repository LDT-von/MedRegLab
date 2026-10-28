"""Read-only diagnostic: quality of the rigid (MI) estimate.

The pull convention is p_mr = p_ct + D(p_ct), so the transform that the rigid
stage estimates must satisfy T(p_ct) = p_mr.  For a pure translation that
means the translation vector equals (MR organ centroid - CT organ centroid).

This script compares three translations per case:
  * ``init``   - the body-mask centroid difference used to seed the search
  * ``oracle`` - the mean organ-centroid difference (uses labels; diagnostic
                 ceiling only, never used to produce a submission)
  * ``mi``     - the translation the Mattes-MI registration actually produced

and reports how far each is from the oracle.  Nothing is written and no
deformable stage is run.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medreglab.data.dataset import competition_masks_from_labels  # noqa: E402
from medreglab.evaluation.prealign import (  # noqa: E402
    _initial_rigid,
    body_centroid,
    estimate_rigid_transform,
    transform_to_displacement_field,
)


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


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "Datasets"
    cases = sys.argv[2:] or [f"{i:03d}" for i in range(17, 25)]
    for case_id in cases:
        case_dir = os.path.join(root, "train", case_id)
        ct = sitk.ReadImage(_pick(case_dir, True, False))
        mr = sitk.ReadImage(_pick(case_dir, False, False))
        ct_c = organ_centroids(ct, sitk.GetArrayFromImage(
            sitk.ReadImage(_pick(case_dir, True, True))), case_id)
        mr_c = organ_centroids(mr, sitk.GetArrayFromImage(
            sitk.ReadImage(_pick(case_dir, False, True))), case_id)

        deltas = [mr_c[o] - ct_c[o] for o in ("liver", "spleen", "kidney")
                  if ct_c.get(o) is not None and mr_c.get(o) is not None]
        oracle = np.mean(deltas, axis=0)

        init = _initial_rigid(ct, mr)
        init_t = np.array(init.GetTranslation(), dtype=np.float64)

        transform = estimate_rigid_transform(ct, mr)
        # The multi-level search returns a CompositeTransform, so the effective
        # rigid offset is measured directly from the physical displacement field.
        mi_field = transform_to_displacement_field(transform, ct)
        mi_array = sitk.GetArrayFromImage(mi_field).astype(np.float64)
        mi_t = mi_array.reshape(-1, 3).mean(axis=0)

        print(f"--- case {case_id}")
        print(f"    oracle  {np.round(oracle, 1)}  |{np.linalg.norm(oracle):7.1f}|")
        print(f"    init    {np.round(init_t, 1)}  err={np.linalg.norm(init_t - oracle):7.1f} mm")
        print(f"    mi      {np.round(mi_t, 1)}  err={np.linalg.norm(mi_t - oracle):7.1f} mm")
        print(f"    mi spread (std over grid) {np.round(mi_array.reshape(-1, 3).std(axis=0), 2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
