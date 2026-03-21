from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.exp03_frozen_expert_residual.run import DEFAULT_DATASET_ORDER, run_experiment
from experiments.shared.repro import save_json


HERE = Path(__file__).resolve().parent
DEFAULT_SWEEP_DIR = HERE / "artifacts" / "seed_sweeps" / "real_suite"
REAL_SUITE_DATASETS = [
    "subghz_real_512",
    "subghz_real_augmented_512",
    "orbit_rf",
    "captured_npy_real_128",
]
METRIC_FIELDS = [
    "test_acc",
    "macro_f1",
    "weighted_f1",
    "delta_vs_best_single",
]
MODEL_FIELDS = [
    "iq_cnn",
    "fft_cnn",
    "frozen_expert_residual_fusion",
]


def seed_results_path(output_dir: Path, seed: int) -> Path:
    return output_dir / f"seed_{seed:02d}.json"


def is_complete_seed_result(path: Path, dataset_names: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        data = __import__("json").loads(path.read_text())
    except Exception:
        return False
    datasets = data.get("datasets", {})
    return all(name in datasets for name in dataset_names)


def summarize_values(values: list[float]) -> dict[str, float]:
    count = len(values)
    mean = sum(values) / count
    if count == 1:
        std = 0.0
        ci95_half_width = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        std = math.sqrt(variance)
        ci95_half_width = 1.96 * std / math.sqrt(count)
    return {
        "count": count,
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "ci95_half_width": ci95_half_width,
    }


def build_aggregate(output_dir: Path, dataset_names: list[str], seeds: list[int]) -> dict:
    import json

    aggregate = {
        "experiment": "exp03_frozen_expert_residual_seed_sweep",
        "datasets": dataset_names,
        "seeds_requested": seeds,
        "completed_seeds": [],
        "seed_results": {},
        "summary": {},
    }

    for seed in seeds:
        path = seed_results_path(output_dir, seed)
        if not is_complete_seed_result(path, dataset_names):
            continue
        data = json.loads(path.read_text())
        aggregate["completed_seeds"].append(seed)
        aggregate["seed_results"][str(seed)] = str(path)

        for dataset_name in dataset_names:
            dataset_summary = aggregate["summary"].setdefault(dataset_name, {})
            dataset_results = data["datasets"][dataset_name]
            for model_name in MODEL_FIELDS:
                model_summary = dataset_summary.setdefault(model_name, {})
                for metric_name in ("test_acc", "macro_f1", "weighted_f1"):
                    model_summary.setdefault(metric_name, []).append(float(dataset_results[model_name][metric_name]))
            dataset_summary.setdefault("best_single_test_acc", []).append(float(dataset_results["best_single_test_acc"]))
            dataset_summary.setdefault("delta_vs_best_single", []).append(float(dataset_results["delta_vs_best_single"]))

    for dataset_name, dataset_summary in aggregate["summary"].items():
        for model_name in MODEL_FIELDS:
            for metric_name in ("test_acc", "macro_f1", "weighted_f1"):
                values = dataset_summary[model_name][metric_name]
                dataset_summary[model_name][metric_name] = summarize_values(values)
        dataset_summary["best_single_test_acc"] = summarize_values(dataset_summary["best_single_test_acc"])
        dataset_summary["delta_vs_best_single"] = summarize_values(dataset_summary["delta_vs_best_single"])

    aggregate["num_completed_seeds"] = len(aggregate["completed_seeds"])
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable seed sweep for Experiment 3.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=REAL_SUITE_DATASETS,
        choices=DEFAULT_DATASET_ORDER,
        help="Datasets to include in the sweep.",
    )
    parser.add_argument("--orbit-max-packets-per-node", type=int, default=256)
    parser.add_argument("--force", action="store_true", help="Re-run seeds even if a complete result file already exists.")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))

    for seed in seeds:
        result_path = seed_results_path(output_dir, seed)
        if not args.force and is_complete_seed_result(result_path, args.datasets):
            print(f"Skipping completed seed {seed}: {result_path}")
            aggregate = build_aggregate(output_dir, args.datasets, seeds)
            save_json(output_dir / "aggregate.json", aggregate)
            continue

        print(f"Running seed {seed} -> {result_path}")
        run_experiment(
            results_path=result_path,
            orbit_max_packets_per_node=args.orbit_max_packets_per_node,
            seed=seed,
            dataset_names=args.datasets,
        )
        aggregate = build_aggregate(output_dir, args.datasets, seeds)
        save_json(output_dir / "aggregate.json", aggregate)
        print(f"Updated aggregate summary: {output_dir / 'aggregate.json'}")


if __name__ == "__main__":
    main()
