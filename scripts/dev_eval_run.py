"""Compare fixed registration configurations on a development set.

This writes to a fresh directory and refuses to overwrite an existing one.
Every configuration is evaluated on exactly the same cases, and the report
records the effective parameters, per-case DSC, per-case pairing against
baseline A, input/label hashes and the git commit.

The results are DEVELOPMENT-SET results. The hidden test labels are never
used, and selecting a configuration on these cases is not an independent
validation.

Example:
    python scripts/dev_eval_run.py --data Datasets --output outputs/dev_eval_20261028
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medreglab.evaluation.dev_eval import (  # noqa: E402
    DEFAULT_CONFIGS,
    EvaluationError,
    config_by_name,
    run_configurations,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset root")
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument("--split", default="train", help="Split to evaluate")
    parser.add_argument(
        "--target_size", type=int, nargs="+", default=[128, 128, 128],
        help="Model grid size (D H W)",
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Configuration names to run (default: all fixed configurations)",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu")
    args = parser.parse_args()

    if args.split == "test":
        print(
            "refusing to run on the test split: the hidden test labels are not "
            "available and the test set must not drive configuration selection",
            file=sys.stderr,
        )
        return 2

    configs = DEFAULT_CONFIGS
    if args.configs:
        try:
            configs = tuple(config_by_name(name) for name in args.configs)
        except EvaluationError as error:
            print(f"configuration error: {error}", file=sys.stderr)
            return 2

    try:
        comparison = run_configurations(
            configs=configs,
            data_root=Path(args.data),
            output_dir=Path(args.output),
            target_size=tuple(args.target_size),
            split=args.split,
            device=args.device,
        )
    except EvaluationError as error:
        print(f"development evaluation failed: {error}", file=sys.stderr)
        return 1

    print()
    print("=== development-set comparison (NOT independent validation) ===")
    for name, entry in comparison["configs"].items():
        summary = entry["summary"]
        if summary["num_cases_scored"] == 0:
            print(f"{name}: no cases scored")
            continue
        print(
            f"{name}: mean DSC={summary['mean_dsc']:.4f} "
            f"DSC30={summary['dsc30']:.4f} (n={summary['dsc30_num_worst']}) "
            f"fold_ratio={summary['mean_fold_ratio']}"
        )
    print()
    print(f"report: {Path(args.output) / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
