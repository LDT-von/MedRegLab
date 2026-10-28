"""Synthetic integration tests for inference, geometry and official packaging."""

import json
import importlib
from pathlib import Path
import zipfile

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from experiments.infer_competition import (
    _apply_deformation_to_label,
    _compute_dsc_for_sample,
    generate_submission,
    run_competition_inference,
)
from medreglab.data import CompetitionDataset

# LPS direction matrices: identity and a 90 degree rotation about the LPS z axis.
_DIRECTION_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
_DIRECTION_ROTATED_Z = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)

# Synthetic anatomy: label value -> CT voxel index (x, y, z). The positions are
# deliberately asymmetric and spread out so that swapping two axes or
# mis-mapping a physical direction cannot accidentally reproduce them.
_SYNTHETIC_ORGANS = {
    1: (4, 1, 2),
    2: (1, 3, 3),
    3: (3, 4, 1),
    4: (2, 2, 4),
}


def _label_image(size_xyz, spacing, origin, direction, marks):
    """Build a label image from {(x, y, z): label_value}."""
    array = np.zeros(tuple(reversed(size_xyz)), dtype=np.uint8)
    for (x, y, z), value in marks.items():
        array[z, y, x] = value
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    return image


def _constant_displacement_field(reference, displacement_xyz):
    """A constant LPS displacement field on the reference image's grid."""
    array = np.zeros((*reversed(reference.GetSize()), 3), dtype=np.float32)
    array[..., 0] = displacement_xyz[0]
    array[..., 1] = displacement_xyz[1]
    array[..., 2] = displacement_xyz[2]
    field = sitk.GetImageFromArray(array, isVector=True)
    field.CopyInformation(reference)
    return field


def test_native_dsc_resamples_each_physical_axis_to_the_expected_voxel(tmp_path, monkeypatch):
    """The physical scoring path must pull MR labels along x, y and z correctly.

    Each case places identical anatomy in CT and in a twice-finer MR grid whose
    origin is offset by a known physical displacement. The MR organ therefore
    sits exactly where the pull transform is expected to sample it, so the
    warped MR label must land back on the original CT voxel. The three
    geometries cover a cubic isotropic grid, a non-cubic anisotropic grid and
    a non-cubic grid with a non-identity direction matrix.
    """
    monkeypatch.chdir(tmp_path)
    base_origin = np.array((11.0, -7.0, 42.0))
    geometries = {
        "cubic_isotropic": ((5, 5, 5), (1.0, 1.0, 1.0), _DIRECTION_IDENTITY),
        "non_cubic_anisotropic": ((7, 6, 5), (1.5, 2.0, 3.0), _DIRECTION_IDENTITY),
        "non_cubic_rotated": ((7, 6, 5), (1.5, 2.0, 3.0), _DIRECTION_ROTATED_Z),
    }
    for name, (size_xyz, spacing, direction) in geometries.items():
        ct_marks = {index: label for label, index in _SYNTHETIC_ORGANS.items()}
        fixed_label = _label_image(size_xyz, spacing, tuple(base_origin), direction, ct_marks)
        mr_size = tuple(2 * extent for extent in size_xyz)
        mr_spacing = tuple(extent / 2 for extent in spacing)

        for axis_name, component in (("x", 0), ("y", 1), ("z", 2)):
            displacement = np.zeros(3)
            displacement[component] = spacing[component]
            mr_origin = tuple(base_origin + displacement)
            mr_marks = {
                tuple(2 * coordinate for coordinate in index): label
                for label, index in _SYNTHETIC_ORGANS.items()
            }
            moving_label = _label_image(mr_size, mr_spacing, mr_origin, direction, mr_marks)

            fixed_path = f"fixed-{name}-{axis_name}.nii.gz"
            moving_path = f"moving-{name}-{axis_name}.nii.gz"
            sitk.WriteImage(fixed_label, fixed_path)
            sitk.WriteImage(moving_label, moving_path)

            field = _constant_displacement_field(fixed_label, displacement)
            sample = {
                "case_id": "001",
                "fixed_sitk": fixed_label,
                "moving_sitk": moving_label,
            }
            result = _compute_dsc_for_sample(
                sample, field, fixed_path, moving_path, ["liver", "spleen", "kidney"]
            )
            # Every organ returns to its original CT voxel, so every DSC is 1.0.
            assert result["liver"] == pytest.approx(1.0), f"{name}/{axis_name} liver"
            assert result["spleen"] == pytest.approx(1.0), f"{name}/{axis_name} spleen"
            assert result["kidney"] == pytest.approx(1.0), f"{name}/{axis_name} kidney"
            assert result["mean"] == pytest.approx(1.0), f"{name}/{axis_name} mean"

            # The warped MR label must land on the exact expected CT voxel.
            transform = sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))
            warped = sitk.Resample(
                moving_label, fixed_label, transform,
                sitk.sitkNearestNeighbor, 0, moving_label.GetPixelID(),
            )
            warped_array = sitk.GetArrayViewFromImage(warped)
            for label, expected_index in _SYNTHETIC_ORGANS.items():
                found = np.argwhere(warped_array == label)
                assert len(found) == 1, f"{name}/{axis_name} label {label} count"
                actual = (int(found[0][2]), int(found[0][1]), int(found[0][0]))
                assert actual == expected_index, f"{name}/{axis_name} label {label}"

            # A pull along the wrong physical axis must not reproduce the anatomy.
            wrong = np.zeros(3)
            wrong[(component + 1) % 3] = spacing[(component + 1) % 3]
            wrong_field = _constant_displacement_field(fixed_label, wrong)
            wrong_transform = sitk.DisplacementFieldTransform(
                sitk.Cast(wrong_field, sitk.sitkVectorFloat64)
            )
            wrong_warped = sitk.Resample(
                moving_label, fixed_label, wrong_transform,
                sitk.sitkNearestNeighbor, 0, moving_label.GetPixelID(),
            )
            wrong_array = sitk.GetArrayViewFromImage(wrong_warped)
            wrong_found = np.argwhere(wrong_array == 1)
            wrong_index = (
                (int(wrong_found[0][2]), int(wrong_found[0][1]), int(wrong_found[0][0]))
                if len(wrong_found) else None
            )
            assert wrong_index != _SYNTHETIC_ORGANS[1], (
                f"{name}/{axis_name}: wrong axis reproduced the anatomy"
            )


def test_native_dsc_rejects_geometry_mismatch(tmp_path, monkeypatch):
    """A segmentation whose geometry differs from its image must be rejected."""
    monkeypatch.chdir(tmp_path)
    fixed = _label_image((5, 6, 7), (1.5, 2.0, 3.0), (11.0, -7.0, 42.0), _DIRECTION_IDENTITY, {(1, 1, 1): 1})
    moving = _label_image((5, 6, 7), (1.5, 2.0, 3.0), (11.0, -7.0, 42.0), _DIRECTION_IDENTITY, {(1, 1, 1): 1})
    sitk.WriteImage(fixed, "fixed-label.nii.gz")
    sitk.WriteImage(moving, "moving-label.nii.gz")
    field = _constant_displacement_field(fixed, (0.0, 0.0, 0.0))

    wrong_spacing = _label_image((5, 6, 7), (1.0, 2.0, 3.0), (11.0, -7.0, 42.0), _DIRECTION_IDENTITY, {(1, 1, 1): 1})
    sitk.WriteImage(wrong_spacing, "wrong-label.nii.gz")
    sample = {"case_id": "001", "fixed_sitk": fixed, "moving_sitk": moving}
    with pytest.raises(ValueError, match="geometry does not match"):
        _compute_dsc_for_sample(
            sample, field, "wrong-label.nii.gz", "moving-label.nii.gz",
            ["liver", "spleen", "kidney"],
        )

    with pytest.raises(ValueError, match="liver, spleen, and combined kidney"):
        _compute_dsc_for_sample(
            sample, field, "fixed-label.nii.gz", "moving-label.nii.gz",
            ["liver", "spleen"],
        )



def test_label_warp_identity_and_axis_translations():
    label = torch.arange(4 * 5 * 6, dtype=torch.float32).reshape(1, 1, 4, 5, 6)
    phi = torch.zeros(1, 3, 4, 5, 6)
    torch.testing.assert_close(_apply_deformation_to_label(label, phi), label)
    for component, axis in ((0, 4), (1, 3), (2, 2)):
        shifted = phi.clone()
        shifted[:, component] = 1
        expected = torch.zeros_like(label)
        dst = [slice(None)] * 5
        src = [slice(None)] * 5
        dst[axis] = slice(0, -1)
        src[axis] = slice(1, None)
        expected[tuple(dst)] = label[tuple(src)]
        torch.testing.assert_close(_apply_deformation_to_label(label, shifted), expected)


def test_native_dsc_uses_mr_labels_and_ct_to_mr_physical_field(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fixed_array = np.zeros((5, 6, 7), dtype=np.uint8)
    fixed_array[2, 3, 2] = 1
    fixed_array[2, 3, 3] = 2
    fixed_array[1, 1, 1] = 3
    fixed_array[1, 1, 4] = 4
    moving_array = np.zeros_like(fixed_array)
    moving_array[:, :, 1:] = fixed_array[:, :, :-1]
    fixed_label = sitk.GetImageFromArray(fixed_array)
    moving_label = sitk.GetImageFromArray(moving_array)
    for image in (fixed_label, moving_label):
        image.SetSpacing((1.5, 2.0, 3.0))
        image.SetOrigin((11.0, -7.0, 42.0))
    sitk.WriteImage(fixed_label, "fixed-label.nii.gz")
    sitk.WriteImage(moving_label, "moving-label.nii.gz")
    displacement = np.zeros((5, 6, 7, 3), dtype=np.float32)
    displacement[..., 0] = 1.5  # One MR voxel in the +x pull direction.
    field = sitk.GetImageFromArray(displacement, isVector=True)
    field.CopyInformation(fixed_label)
    sample = {"case_id": "001", "fixed_sitk": fixed_label, "moving_sitk": moving_label}
    result = _compute_dsc_for_sample(
        sample, field, "fixed-label.nii.gz", "moving-label.nii.gz",
        ["liver", "spleen", "kidney"],
    )
    assert result == {"liver": 1.0, "spleen": 1.0, "kidney": 1.0, "mean": 1.0}


def test_inference_refuses_overwrite_and_fake_checkpoint(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_competition_inference("unused", existing, max_iter=1, device="cpu")
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    with pytest.raises(ValueError, match="no learned checkpoint"):
        run_competition_inference("unused", tmp_path / "other", model_path="missing.pth", device="cpu")
    with pytest.raises(ValueError, match="positive integer"):
        run_competition_inference("unused", tmp_path / "other", max_iter=0, device="cpu")


def test_eight_synthetic_cases_infer_then_package(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = Path("data")
    for number in range(25, 33):
        case = data / "test" / f"{number:03d}"
        case.mkdir(parents=True)
        ct = sitk.GetImageFromArray(np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7))
        mr = sitk.GetImageFromArray(np.arange(6 * 7 * 8, dtype=np.float32).reshape(6, 7, 8))
        ct.SetSpacing((1.2, 1.5, 2.0))
        ct.SetOrigin((12.0, -8.0, 5.0))
        mr.SetSpacing((0.8, 1.1, 1.7))
        mr.SetOrigin((-6.0, 4.0, -3.0))
        sitk.WriteImage(ct, str(case / "ct.nii"))
        sitk.WriteImage(mr, str(case / "mr.nii"))
    summary = run_competition_inference(
        data, "inference", split="test", device="cpu", target_size=(6, 7, 8), max_iter=2,
    )
    assert summary["num_processed"] == 8 and summary["num_errors"] == 0
    assert len(summary["cases"]) == 8
    assert {case["case_id"] for case in summary["cases"]} == {f"{n:03d}" for n in range(25, 33)}
    report = generate_submission("inference", "delivery", data_root=data)
    assert report["status"] == "PASS" and report["num_cases"] == 8
    with zipfile.ZipFile("delivery/submission.zip") as archive:
        paths = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        assert paths == [f"submission/{n:03d}/displacement_field.nii.gz" for n in range(25, 33)]
        assert archive.testzip() is None
    assert json.loads(Path("delivery/format_check.json").read_text(encoding="utf-8"))["status"] == "PASS"
    # A field from an earlier run must never be paired with changed patient input.
    with (data / "test" / "025" / "mr.nii").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="mr_input_sha256"):
        generate_submission("inference", "mismatched", data_root=data)
    assert not Path("mismatched").exists()


def test_missing_test_split_cannot_fall_back_to_train(tmp_path):
    (tmp_path / "train").mkdir()
    with pytest.raises(FileNotFoundError, match="Missing test directory"):
        run_competition_inference(tmp_path, tmp_path / "out", device="cpu", max_iter=1)
    assert not (tmp_path / "out").exists()


def test_constant_mr_is_normalized_and_invalid_options_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    case = Path("data/test/025")
    case.mkdir(parents=True)
    ct = sitk.GetImageFromArray(np.zeros((3, 4, 5), dtype=np.float32))
    mr = sitk.GetImageFromArray(np.full((3, 4, 5), 42, dtype=np.float32))
    sitk.WriteImage(ct, str(case / "ct.nii"))
    sitk.WriteImage(mr, str(case / "mr.nii"))
    sample = CompetitionDataset("data", split="test")[0]
    assert torch.isfinite(sample["moving"]).all()
    assert torch.count_nonzero(sample["moving"]) == 0
    with pytest.raises(ValueError, match="ct_clip"):
        CompetitionDataset("data", split="test", ct_clip=(1, 1))
    with pytest.raises(ValueError, match="percentile"):
        CompetitionDataset("data", split="test", intensity_norm="unsupported")


def test_dataset_rejects_ambiguous_pairs_and_finds_gz_labels(tmp_path):
    case = tmp_path / "test" / "025"
    case.mkdir(parents=True)
    for name in ("ct.nii", "ct.nii.gz", "mr.nii"):
        (case / name).touch()
    with pytest.raises(ValueError, match="Multiple ct_image"):
        CompetitionDataset(tmp_path, split="test")
    unpaired = tmp_path / "extra" / "CT" / "ct"
    unpaired.mkdir(parents=True)
    (unpaired / "image.nii.gz").touch()
    (unpaired / "image-label.nii.gz").touch()
    dataset = CompetitionDataset(tmp_path, split="unpaired_ct")
    assert len(dataset) == 1
    assert Path(dataset.get_sample_info(0)["seg_path"]).name == "image-label.nii.gz"


def test_validation_end_to_end_uses_labels_and_records_one_case(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    case = Path("data/val/001")
    case.mkdir(parents=True)
    image = sitk.GetImageFromArray(np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7))
    labels = np.zeros((5, 6, 7), dtype=np.uint8)
    labels[1:3, 1:3, 1:3] = 1
    labels[1:3, 3:5, 1:3] = 2
    labels[1:3, 1:3, 3:5] = 3
    labels[1:3, 3:5, 3:5] = 4
    segmentation = sitk.GetImageFromArray(labels)
    segmentation.CopyInformation(image)
    for name, value in (("ct.nii", image), ("mr.nii", image),
                        ("ct-label.nii", segmentation), ("mr-label.nii", segmentation)):
        sitk.WriteImage(value, str(case / name))
    summary = run_competition_inference(
        "data", "validation", split="val", device="cpu", target_size=(6, 7, 8), max_iter=1,
    )
    assert summary["num_processed"] == 1 and summary["num_errors"] == 0
    assert len(summary["cases"]) == 1
    assert set(summary["cases"][0]["dsc"]) == {"liver", "spleen", "kidney", "mean"}
    assert 0 <= summary["cases"][0]["mean_dsc"] <= 1
    assert summary["evaluation"] is not None
    assert "composite_score" not in summary["evaluation"]


def test_changed_input_during_inference_is_reported_as_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    case = Path("data/test/025")
    case.mkdir(parents=True)
    image = sitk.GetImageFromArray(np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7))
    sitk.WriteImage(image, str(case / "ct.nii"))
    sitk.WriteImage(image, str(case / "mr.nii"))
    module = importlib.import_module("experiments.infer_competition")
    actual_hash = module._sha256_file
    calls = {"mr": 0}

    def fake_hash(path):
        if Path(path).name == "mr.nii":
            calls["mr"] += 1
            if calls["mr"] > 1:
                return "simulated-input-change"
        return actual_hash(path)

    monkeypatch.setattr(module, "_sha256_file", fake_hash)
    with pytest.warns(UserWarning, match="input changed"):
        summary = run_competition_inference(
            "data", "inference", split="test", device="cpu", target_size=(5, 6, 7), max_iter=1,
        )
    assert summary["num_processed"] == 0 and summary["num_errors"] == 1
    assert summary["cases"] == []
