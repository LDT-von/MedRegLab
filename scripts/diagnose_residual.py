"""Read-only diagnostic: where does the residual error live after rigid alignment?

For every case in the geometry-different group this compares, per organ, the
physical displacement that maps the MR organ centroid onto the CT organ
centroid.  If the three organs need the *same* displacement the residual is
still rigid; if they need *different* displacements the residual is
non-rigid and a deformable stage is the right tool.

No registration is run and no file is written.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medreglab.data.dataset import competition_masks_from_labels  # noqa: E402


def organ_centroids(image: sitk.Image, labels: np.ndarray, case_id: str):
    masks = competition_masks_from_labels(labels, case_id)
    out = {}
    for organ, mask in masks.items():
        if not mask.any():
            out[organ] = None
            continue
        idx = np.argwhere(mask).astype(np.float64)  # (z, y, x)
        continuous = idx.mean(axis=0)[::-1]  # (x, y, z)
        point = image.TransformContinuousIndexToPhysicalPoint(
            (float(continuous[0]), float(continuous[1]), float(continuous[2]))
        )
        out[organ] = np.array(point, dtype=np.float64)
    return out


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


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "Datasets"
    cases = sys.argv[2:] or [f"{i:03d}" for i in range(17, 25)]
    for case_id in cases:
        case_dir = os.path.join(root, "train", case_id)
        ct_path = _pick(case_dir, want_ct=True, want_seg=False)
        mr_path = _pick(case_dir, want_ct=False, want_seg=False)
        ct_seg = _pick(case_dir, want_ct=True, want_seg=True)
        mr_seg = _pick(case_dir, want_ct=False, want_seg=True)
        ct = sitk.ReadImage(ct_path)
        mr = sitk.ReadImage(mr_path)
        ct_labels = sitk.GetArrayFromImage(sitk.ReadImage(ct_seg))
        mr_labels = sitk.GetArrayFromImage(sitk.ReadImage(mr_seg))
        ct_c = organ_centroids(ct, ct_labels, case_id)
        mr_c = organ_centroids(mr, mr_labels, case_id)
        print(f"--- case {case_id}  CT{ct.GetSize()} MR{mr.GetSize()}")
        deltas = []
        for organ in ("liver", "spleen", "kidney"):
            if ct_c.get(organ) is None or mr_c.get(organ) is None:
                print(f"    {organ:7s} missing on one side")
                continue
            d = ct_c[organ] - mr_c[organ]
            deltas.append(d)
            print(f"    {organ:7s} CTcent={np.round(ct_c[organ],1)} MRcent={np.round(mr_c[organ],1)} "
                  f"delta={np.round(d,1)} |d|={np.linalg.norm(d):.1f}mm")
        if len(deltas) >= 2:
            arr = np.stack(deltas)
            spread = np.linalg.norm(arr - arr.mean(axis=0), axis=1)
            print(f"    -> centroid disagreement across organs: {np.round(spread,1)} mm "
                  f"(max {spread.max():.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
