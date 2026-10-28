"""Synthetic, label-only checks of the supplied competition scoring rules."""

import math

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from medreglab.data import CompetitionDataset, competition_masks_from_labels
from medreglab.evaluation import CompetitionEvaluator, DSC30_Scorer, competition_case_dice
from medreglab.evaluation.dsc_evaluator import compute_dsc_for_pair, submit_deformation_fields


def test_kidney_label_mapping_and_absent_organs():
    labels = np.array([[[0, 1, 2, 3, 4]]], dtype=np.uint8)
    masks = competition_masks_from_labels(labels, "001")
    assert set(masks) == {"liver", "spleen", "kidney"}
    assert masks["kidney"].tolist() == [[[False, False, False, True, True]]]
    with pytest.raises(ValueError, match="label 4"):
        competition_masks_from_labels(labels, "009")
    later = competition_masks_from_labels(np.array([[[0, 3, 0]]]), "009")
    assert later["kidney"].sum() == 1
    assert not later["liver"].any() and not later["spleen"].any()
    unknown = competition_masks_from_labels(labels, "img0002_tcia")
    assert unknown["kidney"].sum() == 2
    with pytest.raises(ValueError, match="integers"):
        competition_masks_from_labels(np.array([[[1.2]]]), "001")
    with pytest.raises(ValueError, match="finite 3D"):
        competition_masks_from_labels(np.empty((0, 1, 1)), "001")


def test_fixed_empty_organ_and_three_organ_mean():
    fixed = {
        "liver": np.array([1, 1, 0, 0]),
        "spleen": np.array([0, 0, 0, 0]),
        "kidney": torch.tensor([1, 0, 0, 0]),
    }
    warped = {
        "liver": np.array([1, 0, 0, 0]),
        "spleen": np.array([1, 0, 0, 0]),
        "kidney": torch.tensor([0, 0, 0, 0]),
    }
    scores = competition_case_dice(fixed, warped)
    assert scores == pytest.approx({"liver": 2 / 3, "spleen": 1, "kidney": 0, "mean": 5 / 9})
    with pytest.raises(ValueError, match="required"):
        competition_case_dice(fixed, {"liver": warped["liver"]})
    with pytest.raises(ValueError, match="shapes differ"):
        competition_case_dice(fixed, {**warped, "liver": np.zeros(5)})
    with pytest.raises(ValueError, match="NaN or Inf"):
        competition_case_dice(fixed, {**warped, "liver": np.array([0, 0, np.nan, 0])})


def test_dsc30_uses_lowest_three_of_eight_and_no_invented_composite():
    evaluator = CompetitionEvaluator()
    with pytest.raises(ValueError, match="No labeled cases"):
        evaluator.evaluate(save=False)
    assert DSC30_Scorer().compute_per_organ("kidney")["kidney_dsc30"] is None
    scores = [0.8, 0.2, 0.7, 0.1, 0.9, 0.3, 0.6, 0.4]
    for index, score in enumerate(scores):
        evaluator.add_case(str(index), {"liver": score, "spleen": score, "kidney": score})
    result = evaluator.evaluate(save=False)
    assert result["DSC30"]["num_worst"] == 3
    assert result["DSC30"]["dsc30"] == pytest.approx(0.2)
    assert result["DSC30"]["worst_cases"] == ["3", "1", "5"]
    assert result["ranking_rule"] == {"primary": "DSC", "secondary": "DSC30"}
    assert "composite_score" not in result
    with pytest.raises(ValueError, match="Duplicate"):
        evaluator.add_case("0", {"liver": 1, "spleen": 1, "kidney": 1})
    with pytest.raises(ValueError, match="Missing organ"):
        evaluator.add_case("new", {"liver": 1})


def test_dsc30_rounds_case_count_up():
    """DSC30 takes ceil(total * 0.3) cases: 8 -> 3, 24 -> 8."""
    assert math.ceil(8 * 0.3) == 3
    assert math.ceil(24 * 0.3) == 8

    # 24 cases: the lowest 8 are averaged, not the lowest 7.
    evaluator = CompetitionEvaluator()
    scores = [round(0.02 * index, 4) for index in range(24)]
    for index, score in enumerate(scores):
        evaluator.add_case(f"{index:03d}", {"liver": score, "spleen": score, "kidney": score})
    result = evaluator.evaluate(save=False)
    assert result["DSC30"]["num_worst"] == 8
    assert result["DSC30"]["total"] == 24
    assert result["DSC30"]["worst_cases"] == [f"{index:03d}" for index in range(8)]
    assert result["DSC30"]["dsc30"] == pytest.approx(float(np.mean(scores[:8])))
    # A floor-based count would have averaged only 7 cases.
    assert result["DSC30"]["dsc30"] != pytest.approx(float(np.mean(scores[:7])))

    # A single case still yields one worst case rather than an empty selection.
    single = CompetitionEvaluator()
    single.add_case("000", {"liver": 0.5, "spleen": 0.5, "kidney": 0.5})
    assert single.evaluate(save=False)["DSC30"]["num_worst"] == 1

    # Seven cases: ceil(7 * 0.3) == 3.
    seven = CompetitionEvaluator()
    for index, score in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]):
        seven.add_case(str(index), {"liver": score, "spleen": score, "kidney": score})
    assert seven.evaluate(save=False)["DSC30"]["num_worst"] == 3


def test_missing_val_or_test_does_not_fall_back_to_training(tmp_path):
    (tmp_path / "train").mkdir()
    with pytest.raises(FileNotFoundError, match="Validation"):
        CompetitionDataset(tmp_path, split="val")
    with pytest.raises(FileNotFoundError, match="Test"):
        CompetitionDataset(tmp_path, split="test")
    with pytest.raises(ValueError, match="Unsupported"):
        CompetitionDataset(tmp_path, split="typo")


def test_segmentation_resize_uses_image_grid_and_checks_geometry(monkeypatch, tmp_path):
    (tmp_path / "train").mkdir()
    with pytest.warns(UserWarning, match="No samples"):
        dataset = CompetitionDataset(tmp_path, split="train", target_size=(3, 4, 5))
    labels = np.zeros((5, 7, 9), dtype=np.uint8)
    labels[2, 2, 4] = 3
    labels[4, 6, 8] = 4
    image = sitk.GetImageFromArray(labels)
    image.SetSpacing((0.7, 1.2, 2.3))
    monkeypatch.setattr(sitk, "ReadImage", lambda path: image)
    masks = dataset._load_segmentation("in-memory", "001", image)
    assert set(masks) == {"liver", "spleen", "kidney"}
    assert masks["kidney"].shape == (1, 3, 4, 5)
    assert masks["kidney"][0, 1, 1, 2] == 1
    assert masks["kidney"][0, 2, 3, 4] == 1
    assert masks["liver"].sum() == 0
    reference = sitk.Image(image)
    reference.SetOrigin((1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="geometry differs"):
        dataset._load_segmentation("in-memory", "001", reference)


def test_physical_pull_uses_moving_labels_and_nearest_interpolation():
    reference = sitk.Image([6, 6, 6], sitk.sitkFloat32)
    reference.SetSpacing((1.0, 1.0, 2.0))
    reference.SetOrigin((-4.0, -3.0, 10.0))
    fixed_arrays = {organ: np.zeros((6, 6, 6), dtype=np.uint8) for organ in ("liver", "spleen", "kidney")}
    moving_arrays = {organ: np.zeros((6, 6, 6), dtype=np.uint8) for organ in fixed_arrays}
    fixed_arrays["liver"][2, 2, 2] = 1
    moving_arrays["liver"][2, 2, 3] = 1
    fixed_arrays["spleen"][2, 2, 2] = 1
    moving_arrays["kidney"][2, 2, 3] = 1

    def as_image(array):
        image = sitk.GetImageFromArray(array)
        image.CopyInformation(reference)
        return image

    fixed_masks = {key: as_image(value) for key, value in fixed_arrays.items()}
    moving_masks = {key: as_image(value) for key, value in moving_arrays.items()}
    displacement = np.zeros((6, 6, 6, 3), dtype=np.float64)
    displacement[..., 0] = 1.0
    field = sitk.GetImageFromArray(displacement, isVector=True)
    field.CopyInformation(reference)
    with pytest.raises(ValueError, match="Both CT and MR"):
        compute_dsc_for_pair(reference, reference, field, fixed_masks)
    result = compute_dsc_for_pair(reference, reference, field, fixed_masks, moving_segmentations=moving_masks)
    assert result == pytest.approx({"liver": 1.0, "spleen": 0.0, "kidney": 1.0, "mean": 2 / 3})


def test_legacy_flat_submission_fails_closed(tmp_path):
    output = tmp_path / "wrong_submission"
    with pytest.raises(RuntimeError, match="package_submission"):
        submit_deformation_fields([], [], output)
    assert not output.exists()
