"""Run the development-set evaluation for a subset of cases (timing probe).

This is a thin wrapper around medreglab.evaluation.dev_eval that restricts the
case list so a single case can be timed before committing to the full run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from medreglab.evaluation.dev_eval import (  # noqa: E402
    DEFAULT_CONFIGS,
    EvaluationError,
    build_comparison,
    collect_development_cases,
    config_by_name,
    git_commit,
    git_dirty,
    run_configuration,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--target_size", type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--configs", nargs="+", default=None)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    configs = DEFAULT_CONFIGS
    if args.configs:
        configs = tuple(config_by_name(n) for n in args.configs)

    cases = collect_development_cases(Path(args.data), split=args.split)
    if args.cases:
        wanted = set(args.cases)
        cases = [c for c in cases if c.case_id in wanted]
        if len(cases) != len(wanted):
            have = {c.case_id for c in cases}
            raise EvaluationError(f"Requested cases not found: {sorted(wanted - have)}")
    case_ids = [c.case_id for c in cases]

    from medreglab.data import CompetitionDataset
    dataset = CompetitionDataset(
        root=args.data, split=args.split, target_size=tuple(args.target_size)
    )

    log = []
    results = []
    for config in configs:
        t0 = time.perf_counter()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        result = run_configuration(
            config, cases, Path(args.data), output,
            tuple(args.target_size), args.device, dataset=dataset,
        )
        elapsed = time.perf_counter() - t0
        peak = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if args.device == "cuda" else None
        )
        results.append(result)
        (output / config.name / "per_case.json").write_text(
            json.dumps(
                {
                    "summary": result.summary(),
                    "per_case": result.per_case,
                    "errors": result.errors,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        entry = {
            "config": config.name,
            "elapsed_seconds": elapsed,
            "peak_gpu_gib": peak,
            "summary": result.summary(),
        }
        log.append(entry)
        print(
            f"[probe] {config.name}: {elapsed:.1f}s "
            f"peak_gpu={peak if peak is None else round(peak, 2)} GiB "
            f"mean_dsc={result.summary()['mean_dsc']}",
            flush=True,
        )

    comparison = build_comparison(results, case_ids)
    (output / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "timing_log.json").write_text(
        json.dumps(
            {
                "git_commit": git_commit(REPO_ROOT),
                "git_dirty": git_dirty(REPO_ROOT),
                "device": args.device,
                "target_size_dhw": list(args.target_size),
                "case_ids": case_ids,
                "runs": log,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
