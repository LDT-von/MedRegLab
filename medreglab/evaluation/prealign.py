"""Image-driven rigid pre-alignment for CT/MR pairs.

Why this exists
---------------
The deformable model in this repository optimises a displacement field on a
resized grid and converts it to physical millimetres with
``displacement_to_fixed_grid``.  That conversion starts from an *index-based*
correspondence between the CT and the MR grid.  When the two acquisitions
share a physical frame that correspondence is nearly right; when they do not,
it is not an anatomical correspondence at all and the converted field carries a
large rigid error that no amount of TV regularisation can remove.

This module estimates a rigid transform from the CT and the MR **images only**
and expresses it as the same physical pull displacement field that the
submission format expects:

    p_mr = p_ct + D(p_ct),   D(p_ct) = T(p_ct) - p_ct

Organ labels are never read here.  They are used only for scoring, in
``medreglab.evaluation.dev_eval``.

The transform maps a CT physical point to the MR physical point that should be
sampled, which is the pull convention used by the competition submission
path (``experiments/infer_competition``) and by ``score_case``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import math


class PreAlignmentError(RuntimeError):
    """Raised when the rigid pre-alignment cannot be estimated."""


def _foreground_mask(image: sitk.Image, modality: str) -> np.ndarray:
    """A crude body mask, used only to seed the rigid search."""
    array = sitk.GetArrayViewFromImage(image).astype(np.float32)
    if modality == "CT":
        # Air sits near -1000 HU; body tissue is well above -300 HU.
        return array > -300.0
    positive = array[array > 0]
    if positive.size == 0:
        return np.ones(array.shape, dtype=bool)
    threshold = float(np.percentile(positive, 40.0))
    return array > threshold


def body_centroid(image: sitk.Image, modality: str) -> Optional[np.ndarray]:
    """Physical centroid of the body cross-section, or None if undetectable."""
    mask = _foreground_mask(image, modality)
    if not mask.any():
        return None
    indices = np.argwhere(mask)
    # np.argwhere yields (z, y, x); SimpleITK wants (x, y, z).
    continuous = indices.mean(axis=0)[::-1]
    point = image.TransformContinuousIndexToPhysicalPoint(
        (float(continuous[0]), float(continuous[1]), float(continuous[2]))
    )
    return np.array(point, dtype=np.float64)


def _geometry_center(image: sitk.Image) -> Tuple[float, float, float]:
    return tuple(float(v) for v in image.TransformContinuousIndexToPhysicalPoint(
        (image.GetSize()[0] / 2.0 - 0.5, image.GetSize()[1] / 2.0 - 0.5,
         image.GetSize()[2] / 2.0 - 0.5)
    ))


def _initial_rigid(fixed: sitk.Image, moving: sitk.Image) -> sitk.Euler3DTransform:
    """A translation-only starting point that puts the two bodies on top of
    each other.  The rotation part stays at identity, which is correct for
    the axis-aligned DICOM-derived grids in this dataset."""
    transform = sitk.Euler3DTransform()
    transform.SetCenter(_geometry_center(fixed))
    fixed_centroid = body_centroid(fixed, "CT")
    moving_centroid = body_centroid(moving, "MR")
    if fixed_centroid is not None and moving_centroid is not None:
        transform.SetTranslation((moving_centroid - fixed_centroid).tolist())
    return transform


def _metric_transform(
    fixed: sitk.Image,
    moving: sitk.Image,
    transform: sitk.Transform,
    samples: float = 0.2,
    seed: int = 1234,
) -> float:
    """Score an existing CT->MR transform with Mattes mutual information.

    SimpleITK's Mattes metric returns the *negated* mutual information, so a
    **lower** value means a better match.  The transform is scored without
    optimising it, which makes the score a pure comparison value.

    The sampling seed is fixed, so repeated calls on the same pair return the
    same number and candidate selection is reproducible.
    """
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)
    metric = sitk.ImageRegistrationMethod()
    metric.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    metric.SetMetricSamplingStrategy(metric.RANDOM)
    metric.SetMetricSamplingPercentage(samples, seed)
    metric.SetInterpolator(sitk.sitkLinear)
    metric.SetInitialTransform(transform, inPlace=False)
    try:
        value = metric.MetricEvaluate(fixed_f, moving_f)
    except RuntimeError:
        return float("inf")
    return float(value) if math.isfinite(value) else float("inf")


def estimate_rigid_transform(
    fixed: sitk.Image,
    moving: sitk.Image,
    levels: int = 3,
    iterations: int = 200,
    samples: float = 0.2,
    seed: int = 42,
) -> sitk.Transform:
    """Estimate a rigid CT->MR transform from image intensities only.

    Mattes mutual information is used because CT and MR intensities are not
    linearly related.  The transform maps CT physical points to MR physical
    points, matching the pull convention of the submission format.

    This is the single-candidate estimator.  :func:`estimate_rigid_transform_robust`
    additionally compares several candidates and keeps the best one, which is
    what the competition path uses: a gradient-descent MI search can walk away
    from a good initialisation when the metric is flat.
    """
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(samples, seed)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=iterations,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-8,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    shrink = [4, 2, 1][:levels]
    registration.SetShrinkFactorsPerLevel(shrink)
    registration.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0][:levels])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(_initial_rigid(fixed_f, moving_f), inPlace=False)

    try:
        return registration.Execute(fixed_f, moving_f)
    except RuntimeError as error:  # pragma: no cover - defensive
        raise PreAlignmentError(f"rigid registration failed: {error}") from error


def estimate_rigid_transform_robust(
    fixed: sitk.Image,
    moving: sitk.Image,
    levels: int = 3,
    iterations: int = 200,
    samples: float = 0.2,
    seed: int = 42,
    selection_samples: float = 0.2,
    selection_seed: int = 1234,
) -> Tuple[sitk.Transform, dict]:
    """Estimate a rigid CT->MR transform and keep the best of several candidates.

    A single gradient-descent MI search is not reliable here.  The CT and MR
    grids in this dataset are hundreds of millimetres apart, the metric is flat
    far from the optimum, and the optimiser occasionally moves *away* from a
    good initialisation (observed: a 13.9 mm initialisation becoming 41.5 mm).

    The fix is to generate several independent candidates -- different
    initialisations and two different optimisers -- and to keep the one that
    scores best under the same metric.  The score is computed from the images
    only; **no organ label is read**, so the selection remains a legitimate
    image-driven step.

    Returns:
        The winning transform and a dictionary describing the candidates.
    """
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    def _mi(initial, opt, iters, lev):
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(samples, seed)
        registration.SetInterpolator(sitk.sitkLinear)
        if opt == "lbfgs":
            registration.SetOptimizerAsLBFGSB(
                gradientConvergenceTolerance=1e-6,
                numberOfIterations=iters,
                maximumNumberOfCorrections=10,
                maximumNumberOfFunctionEvaluations=2000,
                costFunctionConvergenceFactor=1e7,
            )
        else:
            registration.SetOptimizerAsRegularStepGradientDescent(
                learningRate=2.0, minStep=1e-4, numberOfIterations=iters,
                relaxationFactor=0.5, gradientMagnitudeTolerance=1e-8,
            )
        registration.SetOptimizerScalesFromPhysicalShift()
        shrink = [4, 2, 1][:lev]
        registration.SetShrinkFactorsPerLevel(shrink)
        registration.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0][:lev])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration.SetInitialTransform(initial, inPlace=False)
        return registration.Execute(fixed_f, moving_f)

    def _geometry_initial():
        transform = sitk.Euler3DTransform()
        transform.SetCenter(_geometry_center(fixed_f))
        offset = np.asarray(_geometry_center(moving_f)) - np.asarray(_geometry_center(fixed_f))
        transform.SetTranslation(offset.tolist())
        return transform

    candidates: List[Tuple[str, sitk.Transform]] = [
        ("init_body", _initial_rigid(fixed_f, moving_f)),
        ("init_geometry", _geometry_initial()),
    ]
    for name, initial in list(candidates):
        try:
            candidates.append((f"mi_body_{name}", _mi(initial, "rsgd", iterations, levels)))
        except RuntimeError:
            continue
    try:
        candidates.append(("mi_geometry_rsgd", _mi(_geometry_initial(), "rsgd", iterations, levels)))
    except RuntimeError:
        pass
    try:
        candidates.append(("mi_body_lbfgs", _mi(_initial_rigid(fixed_f, moving_f), "lbfgs", iterations, levels)))
    except RuntimeError:
        pass

    scored = []
    for name, transform in candidates:
        value = _metric_transform(fixed_f, moving_f, transform, selection_samples, selection_seed)
        scored.append({"name": name, "metric": value, "transform": transform})
    finite = [entry for entry in scored if math.isfinite(entry["metric"])]
    if not finite:
        raise PreAlignmentError("no rigid candidate produced a finite metric score")
    best = min(finite, key=lambda entry: entry["metric"])
    report = {
        "selected": best["name"],
        "selected_metric": best["metric"],
        "candidates": [
            {"name": entry["name"], "metric": entry["metric"]} for entry in scored
        ],
    }
    return best["transform"], report


def transform_to_displacement_field(
    transform: sitk.Transform,
    fixed: sitk.Image,
) -> sitk.Image:
    """Convert a CT->MR transform into a physical pull field on the CT grid.

    ``D(p) = T(p) - p`` so that ``p_mr = p_ct + D(p_ct)``, the same convention
    as ``displacement_to_fixed_grid``.
    """
    generator = sitk.TransformToDisplacementFieldFilter()
    generator.SetReferenceImage(fixed)
    generator.SetOutputPixelType(sitk.sitkVectorFloat64)
    try:
        field = generator.Execute(transform)
    except RuntimeError as error:  # pragma: no cover - defensive
        raise PreAlignmentError(f"displacement field conversion failed: {error}") from error
    field = sitk.Cast(field, sitk.sitkVectorFloat32)
    field.CopyInformation(fixed)
    return field


def field_statistics(field: sitk.Image) -> dict:
    """Summary of a displacement field, used to verify the conversion."""
    array = sitk.GetArrayViewFromImage(field)
    flat = array.reshape(-1, 3)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    return {
        "mean_displacement_mm": [float(v) for v in mean],
        "std_displacement_mm": [float(v) for v in std],
        "rigid_magnitude_mm": float(np.linalg.norm(mean)),
        "learned_magnitude_mm": float(np.linalg.norm(std)),
        "max_displacement_mm": float(np.linalg.norm(flat, axis=1).max()),
    }


def resample_to_fixed_grid(
    moving: sitk.Image,
    fixed: sitk.Image,
    transform: sitk.Transform,
) -> sitk.Image:
    """Resample the moving image onto the fixed grid through ``transform``.

    The transform maps CT physical points to MR physical points, so the result
    is the MR image already aligned to the CT frame.  Linear interpolation is
    used for intensities; labels are never touched by this function.
    """
    return sitk.Resample(
        moving,
        fixed,
        transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )


def normalize_ct(array: np.ndarray, ct_clip=(-200.0, 500.0)) -> np.ndarray:
    """CT intensity normalization, matching ``CompetitionDataset``."""
    clipped = np.clip(array, ct_clip[0], ct_clip[1])
    return ((clipped - ct_clip[0]) / (ct_clip[1] - ct_clip[0])).astype(np.float32)


def normalize_mr(array: np.ndarray, percentile=(0.5, 99.5)) -> np.ndarray:
    """MR intensity normalization, matching ``CompetitionDataset``."""
    low, high = np.percentile(array, list(percentile))
    clipped = np.clip(array, low, high)
    if high > low:
        clipped = (clipped - low) / (high - low)
    else:
        clipped = np.zeros_like(clipped)
    return clipped.astype(np.float32)


def _affine_matrix(transform: sitk.Transform) -> np.ndarray:
    """Return the 4x4 homogeneous matrix of an affine transform.

    The matrix is obtained by probing the transform at the origin and the unit
    basis vectors.  This works for any affine transform regardless of its
    SimpleITK subclass: ``Euler3DTransform.GetMatrix`` returns only the 3x3
    rotation about its centre, and ``CompositeTransform`` has no ``GetMatrix``
    at all, so neither can be used directly.
    """
    origin = np.asarray(transform.TransformPoint((0.0, 0.0, 0.0)), dtype=np.float64)
    axes = []
    for unit in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        moved = np.asarray(transform.TransformPoint(unit), dtype=np.float64)
        axes.append(moved - origin)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.stack(axes, axis=1)
    matrix[:3, 3] = origin
    return matrix


def compose_rigid_and_deformable(
    rigid_transform: sitk.Transform,
    deformable_field: sitk.Image,
    fixed: sitk.Image,
) -> sitk.Image:
    """Exact composition of a rigid map and a deformable pull field.

    The rigid transform ``R`` maps CT physical points to MR physical points.
    The deformable field ``u`` lives in the *already aligned* (CT) frame and
    follows the same pull convention as the rest of the pipeline, so the
    aligned image satisfies ``A(y) = MR(R(y))`` and the deformable step gives
    ``warped(y) = A(y + u(y)) = MR(R(y + u(y)))``.

    The submission field ``D`` must reproduce that when the MR is resampled
    with ``p + D(p)``, so the total field is

        D(p) = R(p + u(p)) - p

    This is evaluated explicitly here rather than through a
    ``CompositeTransform`` so that the application order cannot be ambiguous.
    """
    deformable_array = sitk.GetArrayViewFromImage(deformable_field).astype(np.float64)
    if deformable_array.shape[:3] != tuple(reversed(fixed.GetSize())):
        raise PreAlignmentError(
            "Deformable field grid does not match the fixed image grid"
        )
    size_xyz = fixed.GetSize()

    # Physical coordinate of every voxel, vectorised over the whole grid.
    direction = np.asarray(fixed.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(fixed.GetSpacing(), dtype=np.float64)
    origin = np.asarray(fixed.GetOrigin(), dtype=np.float64)
    zz, yy, xx = np.meshgrid(
        np.arange(size_xyz[2]), np.arange(size_xyz[1]), np.arange(size_xyz[0]),
        indexing="ij",
    )
    index_xyz = np.stack([xx, yy, zz], axis=-1).astype(np.float64)
    points = origin + index_xyz @ (direction * spacing).T

    # Both arrays stay in numpy (z, y, x, component) order: points[z, y, x]
    # is the physical coordinate of that voxel and deformable_array[z, y, x]
    # is its (x, y, z) displacement.  No axis flip or transpose is needed;
    # a double [::-1] flip would cancel out for pure translations and hide
    # the bug, but it breaks under rotation.
    u = deformable_array
    shifted = points + u

    # Apply the rigid map R to every shifted point, vectorised.
    matrix = _affine_matrix(rigid_transform)
    linear = matrix[:3, :3]
    offset = matrix[:3, 3]
    mapped = shifted @ linear.T + offset

    total = mapped - points
    field = sitk.GetImageFromArray(
        np.ascontiguousarray(total.astype(np.float32)), isVector=True
    )
    field.CopyInformation(fixed)
    return field

