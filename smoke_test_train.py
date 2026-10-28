"""Synthetic smoke check for the parameter-free per-pair registration model.

This script does not read competition data or train network weights.
"""

import argparse
import torch

from medreglab import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--size", type=int, default=8)
    args = parser.parse_args()
    if args.iterations < 1 or args.size < 2:
        parser.error("--iterations must be positive and --size must be at least 2")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    moving = torch.rand(1, 1, args.size, args.size, args.size, device=args.device)
    fixed = torch.rand_like(moving)
    model = load_model("convex_3d").to(args.device)
    model.max_iter = args.iterations
    with torch.enable_grad():
        result = model(moving, fixed)
    assert result["phi"].shape == (1, 3, args.size, args.size, args.size)
    assert result["warped"].shape == fixed.shape
    assert torch.isfinite(result["phi"]).all() and torch.isfinite(result["warped"]).all()
    assert not any(parameter.requires_grad for parameter in model.parameters())
    print(f"Synthetic per-pair registration passed on {args.device}; no learned checkpoint")


if __name__ == "__main__":
    main()
