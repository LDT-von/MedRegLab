"""Synthetic image and physical displacement I/O tests; no patient data."""

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from medreglab.utils.io_utils import (
    load_batch,
    load_deformation_field,
    load_image,
    normalize_image,
    resample_image,
    save_deformation_field,
    save_image,
)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_physical_vector_nifti_roundtrip_and_reference_geometry(monkeypatch, tmp_path, dimension, dtype):
    monkeypatch.chdir(tmp_path)  # SimpleITK on Windows cannot open some non-ASCII absolute paths.
    shape = (4, 5) if dimension == 2 else (3, 4, 5)
    values = (np.arange(dimension * np.prod(shape), dtype=dtype) / 10).reshape(dimension, *shape)
    tensor = torch.from_numpy(values.copy()).unsqueeze(0).requires_grad_()
    reference = sitk.Image(list(shape[::-1]), sitk.sitkFloat32)
    reference.SetSpacing((0.75, 1.25) if dimension == 2 else (0.75, 1.25, 3.5))
    reference.SetOrigin((10.0, -20.0) if dimension == 2 else (10.0, -20.0, 123.0))
    reference.SetDirection((-1.0, 0.0, 0.0, -1.0) if dimension == 2 else
                           (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
    save_deformation_field(tensor, "field.nii.gz", reference)
    loaded, image = load_deformation_field("field.nii.gz", return_sitk=True)
    assert loaded.shape == (1, dimension, *shape)
    assert loaded.numpy().dtype == dtype
    np.testing.assert_array_equal(loaded[0].numpy(), values)
    assert image.GetNumberOfComponentsPerPixel() == dimension
    assert image.GetSize() == reference.GetSize()
    np.testing.assert_allclose(image.GetSpacing(), reference.GetSpacing(), atol=1e-6)
    np.testing.assert_allclose(image.GetOrigin(), reference.GetOrigin(), atol=1e-6)
    np.testing.assert_allclose(image.GetDirection(), reference.GetDirection(), atol=1e-6)


def test_deformation_rejects_scalar_ambiguous_and_nonfinite_data(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    sitk.WriteImage(sitk.Image([5, 4, 3], sitk.sitkFloat32), "scalar.nii.gz")
    with pytest.raises(ValueError, match="vector image"):
        load_deformation_field("scalar.nii.gz")
    with pytest.raises(ValueError, match="channel-first"):
        save_deformation_field(np.zeros((4, 5, 3), dtype=np.float32), "ambiguous.nii.gz")
    bad = np.zeros((3, 3, 4, 5), dtype=np.float32)
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        save_deformation_field(bad, "nan.nii.gz")
    with pytest.raises(ValueError, match="reference spatial size"):
        save_deformation_field(np.zeros((3, 3, 4, 5), dtype=np.float32), "bad-ref.nii.gz",
                               sitk.Image([5, 4, 4], sitk.sitkFloat32))


def test_save_scalar_tensor_with_grad_and_singleton_depth(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tensor = torch.arange(20, dtype=torch.float32).reshape(1, 1, 1, 4, 5).requires_grad_()
    save_image(tensor, "scalar.mha", metadata={"kind": "synthetic"})
    loaded, image = load_image("scalar.mha", return_sitk=True)
    assert image.GetDimension() == 3
    assert image.GetSize() == (5, 4, 1)
    assert image.GetDirection() == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    torch.testing.assert_close(loaded, tensor.detach())
    assert (tmp_path / "scalar.mha.json").is_file()


def test_load_batch_does_not_add_duplicate_batch_dimension(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for i in range(2):
        save_image(np.full((4, 5), i, dtype=np.float32), f"image{i}.mha")
    batch, metadata = load_batch(["image0.mha", "image1.mha"], return_metadata=True)
    assert batch.shape == (2, 1, 4, 5)
    assert len(metadata) == 2
    assert torch.all(batch[0] == 0) and torch.all(batch[1] == 1)
    with pytest.raises(ValueError, match="at least one"):
        load_batch([])


def test_normalize_numpy_and_preserve_tensor_shape_and_gradient():
    array = np.arange(8, dtype=np.int16).reshape(2, 4)
    normalized = normalize_image(array)
    np.testing.assert_allclose(normalized, array / 7.0, atol=1e-6)
    tensor = torch.tensor(array, dtype=torch.float32, requires_grad=True)
    result = normalize_image(tensor)
    assert result.shape == tensor.shape and result.requires_grad
    result.sum().backward()
    assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_resample_outside_fov_uses_zero_and_rejects_invalid_spacing():
    image = sitk.GetImageFromArray(np.ones((2, 2), dtype=np.float32))
    result = resample_image(image, (0.2, 0.2))
    assert result.GetSize() == (10, 10)
    assert result[9, 9] == 0.0
    with pytest.raises(ValueError, match="target_spacing"):
        resample_image(image, (1.0, 0.0))
