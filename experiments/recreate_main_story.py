from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.exp01_iq_vs_fft.run import run_experiment as run_exp01
from experiments.exp02_gated_multimodal.run import run_experiment as run_exp02
from experiments.exp03_frozen_expert_residual.run import run_experiment as run_exp03
from experiments.shared.repro import save_json
from experiments.shared.story_datasets import missing_story_experiment3_dependencies


RESULTS_PATH = Path(__file__).resolve().parent / "main_story_results.json"


def format_missing_paths(paths: list[Path]) -> str:
    return "\n".join(f"- {path}" for path in paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate the cleaned experiment story.")
    parser.add_argument(
        "--mode",
        choices=["auto", "synthetic-only", "full"],
        default="auto",
        help="`synthetic-only` runs Experiments 1 and 2 only. `auto` also runs Experiment 3 when its datasets are present. `full` requires all Experiment 3 datasets.",
    )
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    start = time.perf_counter()
    exp01 = run_exp01()
    exp02 = run_exp02()
    missing_for_exp03 = missing_story_experiment3_dependencies()
    exp03 = None
    skipped = []

    if args.mode == "synthetic-only":
        skipped.append(
            {
                "experiment": "exp03_frozen_expert_residual",
                "reason": "Skipped by --mode synthetic-only.",
            }
        )
    elif missing_for_exp03:
        if args.mode == "full":
            raise FileNotFoundError(
                "Experiment 3 requires local datasets that are not present in this clone.\n"
                "Missing paths:\n"
                f"{format_missing_paths(missing_for_exp03)}\n\n"
                "Run `./.venv/bin/python experiments/check_data.py` for a full audit and see docs/datasets.md for placement details."
            )
        skipped.append(
            {
                "experiment": "exp03_frozen_expert_residual",
                "reason": "Required local datasets were not present, so Experiment 3 was skipped.",
                "missing_paths": [str(path) for path in missing_for_exp03],
            }
        )
    else:
        exp03 = run_exp03()

    summary = {
        "story": [
            "Experiment 1 shows IQ and FFT are each strong on different synthetic tasks.",
            "Experiment 2 shows gated multimodal fusion helps, but not with strict dominance over the best expert.",
            "Experiment 3 shows frozen-expert residual fusion matches or exceeds the best expert across the comparable benchmark family.",
        ],
        "experiments": {
            "exp01_iq_vs_fft": exp01,
            "exp02_gated_multimodal": exp02,
        },
        "skipped": skipped,
        "mode": args.mode,
        "runtime_seconds": time.perf_counter() - start,
    }
    if exp03 is not None:
        summary["experiments"]["exp03_frozen_expert_residual"] = exp03
    save_json(args.results_path, summary)
    print(f"Saved main story results to {args.results_path}")


if __name__ == "__main__":
    main()
