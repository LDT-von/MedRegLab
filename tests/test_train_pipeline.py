"""Synthetic checks for the parameter-free registration diagnostic runner."""

import importlib
import pytest
import torch
import yaml

runner = importlib.import_module("experiments.train_competition")


def test_yaml_configuration_matches_effective_model_parameters():
    with open("configs/competition_config.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    settings = runner._configuration(config)
    assert settings["num_epochs"] == 1
    assert settings["max_iter"] == 100
    assert settings["lr"] == 0.1
    assert settings["tv_weight"] == 0.01
    assert settings["similarity"] == "mse"


@pytest.mark.parametrize("config", [
    {"num_epochs": 50}, {"batch_size": 2}, {"use_scheduler": True},
    {"random_flip": True}, {"max_iter": 0}, {"similarity": "invalid"},
])
def test_rejects_unsupported_or_ineffective_training_options(config):
    with pytest.raises(ValueError):
        runner._configuration(config)


def test_validation_requires_explicit_directory(tmp_path):
    (tmp_path / "train").mkdir()
    with pytest.raises(FileNotFoundError, match="validation directory"):
        runner._build_val_dataset(tmp_path, runner._configuration({}))


def test_training_and_validation_ids_must_not_overlap(tmp_path):
    class Dataset:
        def __init__(self, data):
            self.samples = data
    train = Dataset([{"case_id": "001", "fixed_path": str(tmp_path / "a"), "moving_path": str(tmp_path / "b")}])
    val = Dataset([{"case_id": "001", "fixed_path": str(tmp_path / "c"), "moving_path": str(tmp_path / "d")}])
    with pytest.raises(ValueError, match="case IDs overlap"):
        runner._check_disjoint(train, val)


def test_label_warp_identity_translation_and_outside_use_xyz_axes():
    label = torch.zeros(1, 1, 3, 4, 5)
    label[0, 0, 1, 2, 3] = 1
    phi = torch.zeros(1, 3, 3, 4, 5)
    assert torch.equal(runner._warp_label(label, phi), label)
    phi[:, 0] = 1
    moved = runner._warp_label(label, phi)
    assert moved.sum().item() == 1
    assert moved[0, 0, 1, 2, 2].item() == 1
    assert moved[0, 0, 1, 2, 4].item() == 0


def test_existing_output_and_checkpoint_are_rejected_before_any_processing(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "history.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner.train_competition({}, tmp_path / "missing_data", output_dir=existing)
    assert (existing / "history.txt").read_text(encoding="utf-8") == "keep"
    with pytest.raises(ValueError, match="no learned checkpoint"):
        runner.train_competition({}, tmp_path / "missing_data", resume_from="anything.pth")


def test_diagnostics_compare_fixed_ct_with_warped_mr_without_training(monkeypatch, tmp_path):
    shape = (3, 4, 5)
    ct_liver = torch.zeros((1, *shape))
    mr_liver = torch.zeros((1, *shape))
    ct_liver[0, 1, 1, 1] = 1
    mr_liver[0, 1, 1, 3] = 1
    zeros = torch.zeros((1, *shape))

    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, case):
            self.case = case
            self.samples = [{"case_id": case, "fixed_path": str(tmp_path / case / "ct"),
                             "moving_path": str(tmp_path / case / "mr")}]

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"case_id": self.case, "fixed": torch.zeros(1, 1, *shape),
                    "moving": torch.zeros(1, 1, *shape), "has_segmentations": True,
                    "seg_fixed": {"liver": ct_liver, "spleen": zeros, "kidney": zeros},
                    "seg_moving": {"liver": mr_liver, "spleen": zeros, "kidney": zeros},
                    "fixed_sitk": object(), "moving_sitk": object()}

    class SyntheticModel(torch.nn.Module):
        def forward(self, moving, fixed):
            return {"phi": torch.zeros(1, 3, *shape), "warped": moving,
                    "loss": 0.25, "loss_history": [0.25]}

    monkeypatch.setattr(runner, "_build_train_dataset", lambda *_: SyntheticDataset("001"))
    monkeypatch.setattr(runner, "_build_val_dataset", lambda *_: SyntheticDataset("025"))
    monkeypatch.setattr(runner, "load_model", lambda _: SyntheticModel())
    output = tmp_path / "diagnostic"
    report = runner.train_competition({"target_size": list(shape)}, tmp_path, tmp_path,
                                      output_dir=output, device="cpu")
    assert report["checkpoint_saved"] is False
    assert report["validation_dsc_scope"] == "resized_grid_diagnostic_not_official_native_space"
    dsc = report["validation_cases"][0]["resized_grid_diagnostic_dsc"]
    assert dsc["liver"] == 0.0
    assert dsc["spleen"] == dsc["kidney"] == 1.0
    assert dsc["mean"] == pytest.approx(2 / 3)
    assert (output / "logs" / "train_history.json").is_file()
    assert not list(output.rglob("*.pth"))
