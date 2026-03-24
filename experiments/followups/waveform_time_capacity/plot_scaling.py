from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
RESULTS_PATH = ARTIFACTS_DIR / "results.json"
FULL_SWEEP_RESULTS_PATH = ARTIFACTS_DIR / "results_full_sweep_before_gated_capacity_match.json"
PLOT_PATH = ARTIFACTS_DIR / "param_count_vs_test_accuracy.png"

FAMILY_COLORS = {
    "baseline": "#4c78a8",
    "fusion": "#f58518",
    "multiscale_time": "#54a24b",
    "large_kernel_time": "#e45756",
    "filterbank_time": "#72b7b2",
}


def _extract_points(payload: dict) -> list[dict]:
    points = []
    for model_name, result in payload["results"].items():
        points.append(
            {
                "model_name": model_name,
                "family": result["family"],
                "scale": result["scale"],
                "params": int(result["num_trainable_parameters"]),
                "test_acc": float(result["test_acc"]),
            }
        )
    return points


def load_points() -> list[dict]:
    merged: dict[str, dict] = {}
    if FULL_SWEEP_RESULTS_PATH.exists():
        payload = json.loads(FULL_SWEEP_RESULTS_PATH.read_text())
        for point in _extract_points(payload):
            merged[point["model_name"]] = point
    payload = json.loads(RESULTS_PATH.read_text())
    for point in _extract_points(payload):
        merged[point["model_name"]] = point
    return sorted(merged.values(), key=lambda item: item["params"])


def label_for(point: dict) -> str:
    scale = point["scale"]
    if scale in {"baseline", "reference"}:
        return point["model_name"]
    return f"{point['model_name']} ({scale})"


def main() -> None:
    points = load_points()

    fig, ax = plt.subplots(figsize=(10, 6))
    seen_families: set[str] = set()

    for index, point in enumerate(points):
        family = point["family"]
        color = FAMILY_COLORS.get(family, "#333333")
        label = family.replace("_", " ") if family not in seen_families else None
        ax.scatter(point["params"], point["test_acc"], s=80, color=color, edgecolor="black", linewidth=0.4, label=label)
        seen_families.add(family)

        y_offset = 10 if index % 2 == 0 else -14
        ax.annotate(
            label_for(point),
            (point["params"], point["test_acc"]),
            textcoords="offset points",
            xytext=(6, y_offset),
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (log scale)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Waveform-Family Scaling: Parameter Count vs Test Accuracy")
    ax.set_ylim(0.55, 0.92)
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)
    print(f"Saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
