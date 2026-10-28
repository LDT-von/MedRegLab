"""One-pass registration diagnostics for the parameter-free competition model.

The only registered model optimizes a displacement field for each input pair.
It does not learn network parameters and therefore cannot produce a checkpoint.
Validation scores here use resized tensors and are diagnostic, not the
competition's native-space evaluation.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from medreglab import load_model
from medreglab.data import CompetitionDataset

_SITK_KEYS = {"fixed_sitk", "moving_sitk", "sitk_img"}
_ORGANS = ("liver", "spleen", "kidney")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate_training(batch):
    """Collate one sample without passing SimpleITK objects to PyTorch."""
    if len(batch) != 1:
        raise ValueError("convex_3d diagnostics require batch_size=1")
    sample = {key: value for key, value in batch[0].items()
              if key not in _SITK_KEYS and value is not None}
    return torch.utils.data.dataloader.default_collate([sample])


def _configuration(config):
    """Normalize supported flat or YAML sections, rejecting ineffective options."""
    training = config.get("training") or {}
    registration = config.get("registration") or {}
    data = config.get("data") or {}
    augmentation = config.get("augmentation") or {}
    def pick(name, section, default):
        return config[name] if name in config else section.get(name, default)

    result = {
        "model": config.get("model", "convex_3d"),
        "num_epochs": pick("num_epochs", training, 1),
        "batch_size": pick("batch_size", training, 1),
        "num_workers": pick("num_workers", training, 0),
        "seed": pick("seed", training, 42),
        "target_size": pick("target_size", data, [128, 128, 128]),
        "max_iter": pick("max_iter", registration, 100),
        "lr": config.get("registration_lr", registration.get("lr", 0.1)),
        "tv_weight": config.get("registration_tv_weight", registration.get("tv_weight", 0.01)),
        "similarity": pick("similarity", registration, "mse"),
    }
    if result["model"] != "convex_3d":
        raise ValueError("Only convex_3d is available")
    if result["num_epochs"] != 1:
        raise ValueError("convex_3d learns no network parameters; use one diagnostics pass (--epochs 1)")
    if result["batch_size"] != 1:
        raise ValueError("convex_3d diagnostics currently require batch_size=1")
    if not isinstance(result["max_iter"], int) or isinstance(result["max_iter"], bool) or result["max_iter"] <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(result["num_workers"], int) or result["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if len(result["target_size"]) != 3 or any(not isinstance(n, int) or n < 2 for n in result["target_size"]):
        raise ValueError("target_size must contain three integers >= 2")
    if not math.isfinite(result["lr"]) or result["lr"] <= 0:
        raise ValueError("registration lr must be finite and positive")
    if not math.isfinite(result["tv_weight"]) or result["tv_weight"] < 0:
        raise ValueError("registration tv_weight must be finite and nonnegative")
    if result["similarity"] not in ("mse", "ncc"):
        raise ValueError("similarity must be mse or ncc")
    if pick("use_scheduler", training, False):
        raise ValueError("No scheduler is applicable: convex_3d has no trainable network parameters")
    for name in ("elastic_deform", "random_flip", "intensity_aug"):
        if config.get(name, augmentation.get(name, False)):
            raise ValueError(f"{name} is not supported by the one-pass diagnostic runner")
    return result


def _build_train_dataset(data_dir, config):
    if not (Path(data_dir) / "train").is_dir():
        raise FileNotFoundError(f"Training directory missing: {Path(data_dir) / 'train'}")
    dataset = CompetitionDataset(root=data_dir, split="train", transforms=None,
                                 target_size=tuple(config["target_size"]))
    if not len(dataset):
        raise ValueError("Training split contains no paired cases")
    return dataset


def _build_val_dataset(data_dir, config):
    if not (Path(data_dir) / "val").is_dir():
        raise FileNotFoundError(f"Explicit validation directory missing: {Path(data_dir) / 'val'}")
    dataset = CompetitionDataset(root=data_dir, split="val", transforms=None,
                                 target_size=tuple(config["target_size"]))
    if not len(dataset):
        raise ValueError("Validation split contains no paired cases")
    return dataset


def _check_disjoint(train_dataset, val_dataset):
    train_ids = {sample["case_id"] for sample in train_dataset.samples}
    val_ids = {sample["case_id"] for sample in val_dataset.samples}
    if train_ids & val_ids:
        raise ValueError(f"Training and validation case IDs overlap: {sorted(train_ids & val_ids)}")
    train_paths = {str(Path(sample[key]).resolve()) for sample in train_dataset.samples
                   for key in ("fixed_path", "moving_path")}
    val_paths = {str(Path(sample[key]).resolve()) for sample in val_dataset.samples
                 for key in ("fixed_path", "moving_path")}
    if train_paths & val_paths:
        raise ValueError("Training and validation reuse image files")


def _batch_image(batch, key, device):
    value = batch[key].to(device)
    if value.ndim == 6 and value.shape[1] == 1:
        value = value.squeeze(1)
    if value.ndim != 5:
        raise ValueError(f"Expected [B,C,D,H,W] after collation for {key}; got {tuple(value.shape)}")
    return value


def _warp_label(label, phi):
    """Nearest-neighbor pull in the model's x=W, y=H, z=D voxel convention."""
    if label.ndim != 5 or phi.ndim != 5 or label.shape[0] != phi.shape[0] or phi.shape[1] != 3:
        raise ValueError("Expected label [B,1,D,H,W] and phi [B,3,D,H,W]")
    batch, _, depth, height, width = label.shape
    if tuple(phi.shape[2:]) != (depth, height, width) or min(depth, height, width) < 2:
        raise ValueError("Label and field must have the same 3D grid with dimensions >= 2")
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, depth, device=phi.device, dtype=phi.dtype),
        torch.linspace(-1, 1, height, device=phi.device, dtype=phi.dtype),
        torch.linspace(-1, 1, width, device=phi.device, dtype=phi.dtype), indexing="ij",
    )
    base = torch.stack((x, y, z), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1, -1)
    scale = phi.new_tensor([2 / (width - 1), 2 / (height - 1), 2 / (depth - 1)])
    grid = base + phi.permute(0, 2, 3, 4, 1) * scale
    return F.grid_sample(label.to(device=phi.device, dtype=phi.dtype), grid,
                         mode="nearest", padding_mode="zeros", align_corners=True)


def _compute_batch_dsc(batch, result, device):
    """CT vs warped MR on the resized diagnostic grid, not official native DSC."""
    from medreglab.evaluation import competition_case_dice
    fixed_masks, moving_masks = batch.get("seg_fixed"), batch.get("seg_moving")
    if fixed_masks is None or moving_masks is None:
        return None
    if any(organ not in fixed_masks or organ not in moving_masks for organ in _ORGANS):
        raise ValueError("Expected liver, spleen and kidney masks for both CT and MR")
    phi = result["phi"].to(device)
    warped = {organ: _warp_label(moving_masks[organ].to(device).float(), phi)[0, 0]
              for organ in _ORGANS}
    fixed = {organ: fixed_masks[organ].to(device)[0, 0] for organ in _ORGANS}
    return competition_case_dice(fixed, warped)


def train_competition(config, train_data_dir, val_data_dir=None,
                      output_dir="competition_outputs", device=None, resume_from=None):
    """Run one diagnostic pass; no learned weights/checkpoints are produced."""
    settings = _configuration(config)
    if resume_from is not None:
        raise ValueError("convex_3d has no learned checkpoint; omit --resume")
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output already contains data; choose a new directory: {output}")
    train_dataset = _build_train_dataset(train_data_dir, settings)
    val_dataset = _build_val_dataset(val_data_dir, settings) if val_data_dir is not None else None
    if val_dataset is not None:
        _check_disjoint(train_dataset, val_dataset)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(settings["model"]).to(device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("This diagnostic runner supports parameter-free convex_3d only")
    model.max_iter = settings["max_iter"]
    model.lr = settings["lr"]
    model.tv_weight = settings["tv_weight"]
    model.similarity = settings["similarity"]
    set_seed(settings["seed"])
    loader_options = {"batch_size": 1, "shuffle": False, "num_workers": settings["num_workers"],
                      "collate_fn": _collate_training}
    train_loader = DataLoader(train_dataset, **loader_options)
    val_loader = DataLoader(val_dataset, **loader_options) if val_dataset is not None else None
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    (output / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"[MedRegLab] One-pass per-pair registration diagnostics on {device}; no learned checkpoint")
    records = []
    with torch.enable_grad():
        for batch in train_loader:
            moving, fixed = _batch_image(batch, "moving", device), _batch_image(batch, "fixed", device)
            result = model(moving, fixed)
            records.append({"case_id": batch["case_id"][0], "objective": float(result["loss"]),
                            "iterations": len(result["loss_history"])})
        validation = []
        if val_loader is not None:
            for batch in val_loader:
                moving, fixed = _batch_image(batch, "moving", device), _batch_image(batch, "fixed", device)
                result = model(moving, fixed)
                validation.append({"case_id": batch["case_id"][0], "objective": float(result["loss"]),
                                   "resized_grid_diagnostic_dsc": _compute_batch_dsc(batch, result, device)})
    summary = {
        "mode": "one_pass_per_pair_registration_diagnostics",
        "model": settings["model"],
        "learned_network_parameters": False,
        "checkpoint_saved": False,
        "training_cases": records,
        "validation_cases": validation,
        "validation_dsc_scope": "resized_grid_diagnostic_not_official_native_space",
    }
    (output / "logs" / "train_history.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[MedRegLab] Diagnostics finished: {len(records)} train, {len(validation)} validation cases")
    return summary


def quick_train(train_data_dir, val_data_dir=None, output_dir="competition_outputs",
                model_name="convex_3d", num_epochs=1, batch_size=1,
                target_size=None, learning_rate=0.1):
    """Compatibility wrapper; learning_rate controls per-pair displacement Adam."""
    return train_competition({"model": model_name, "num_epochs": num_epochs, "batch_size": batch_size,
                              "target_size": target_size or [128, 128, 128],
                              "registration_lr": learning_rate},
                             train_data_dir, val_data_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset root containing train/")
    parser.add_argument("--val", help="Dataset root containing a separate val/ split")
    parser.add_argument("--output", default="competition_outputs", help="New/empty output directory")
    parser.add_argument("--config", help="Optional configs/competition_config.yaml")
    parser.add_argument("--model", help="Only convex_3d is available")
    parser.add_argument("--epochs", type=int, help="Must be 1: this model has no learnable parameters")
    parser.add_argument("--batch_size", type=int, help="Must be 1")
    parser.add_argument("--lr", type=float, help="Per-pair displacement Adam learning rate")
    parser.add_argument("--tv_weight", type=float, help="Per-pair displacement TV weight")
    parser.add_argument("--similarity", choices=["mse", "ncc"])
    parser.add_argument("--max_iter", type=int, help="Maximum per-pair optimizer iterations")
    parser.add_argument("--target_size", type=int, nargs=3, metavar=("D", "H", "W"))
    parser.add_argument("--resume", help="Unsupported: convex_3d has no checkpoint")
    args = parser.parse_args()
    config = {}
    if args.config:
        import yaml
        with open(args.config, encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        if not isinstance(config, dict):
            raise ValueError("Config must contain a mapping")
    overrides = {"model": args.model, "num_epochs": args.epochs, "batch_size": args.batch_size,
                 "registration_lr": args.lr, "registration_tv_weight": args.tv_weight,
                 "similarity": args.similarity, "max_iter": args.max_iter, "target_size": args.target_size}
    config.update({key: value for key, value in overrides.items() if value is not None})
    train_competition(config, args.data, args.val, args.output, resume_from=args.resume)


if __name__ == "__main__":
    main()
