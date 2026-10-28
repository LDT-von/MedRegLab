"""Read-only preflight check for a development split.

Prints the case inventory, geometry and label histogram without running any
registration, so that a broken or ambiguous dataset is caught before the
expensive part of the evaluation.

Example:
    python scripts/dev_eval_preflight.py --data Datasets --split train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medreglab.evaluation.dev_eval import EvaluationError, preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset root")
    parser.add_argument("--split", default="train", help="Split to inspect")
    parser.add_argument("--output", help="Optional path for the JSON report")
    args = parser.parse_args()

    try:
        report = preflight(Path(args.data), split=args.split)
    except EvaluationError as error:
        print(f"preflight failed: {error}", file=sys.stderr)
        return 1

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)

    differing = [case["case_id"] for case in report["cases"] if not case["same_geometry"]]
    print()
    print(f"cases: {report['num_cases']}")
    print(f"cases whose CT and MR geometry differ: {len(differing)} {differing}")
    print(
        "NOTE: baseline A (zero physical field) and baseline B (model identity "
        "converted to the CT grid) differ for these cases."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
