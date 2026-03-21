from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from experiments.run_experiment11_frozen_expert_residual_multidataset import (  # noqa: E402
    AUG_REAL_TEST_PATH,
    AUG_REAL_TRAIN_PATH,
    AUG_REAL_VAL_PATH,
    CAPTURED_TEST_PATH,
    CAPTURED_TRAIN_PATH,
    CAPTURED_VAL_PATH,
    ExpertCNN,
    FrozenExpertResidualFusion,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    REAL_TEST_PATH,
    REAL_TRAIN_PATH,
    REAL_VAL_PATH,
    SEED,
    WEIGHT_DECAY,
    build_h5_dataset,
    compute_residual_loss,
    make_loader,
    make_pair_loader,
)
from experiments.run_experiment9_wavelet_multidataset import (  # noqa: E402
    load_modulation_dataset,
    load_orbit_dataset,
    load_waveform_dataset,
    to_fft,
)


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results.json"
PLOTS_DIR = HERE / "plots"

PATTERN_LABELS = {
    "000": "None",
    "100": "IQ only",
    "010": "FFT only",
    "001": "Fusion only",
    "110": "IQ+FFT",
    "101": "IQ+Fusion",
    "011": "FFT+Fusion",
    "111": "All three",
}
PATTERN_ORDER = ["000", "100", "010", "001", "110", "101", "011", "111"]


def train_single_expert_with_preds(
    train_loader,
    val_loader,
    test_loader,
    y_test: np.ndarray,
    class_names: list[str],
    epochs: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model = ExpertCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=use_cuda)

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    for epoch in range(1, epochs + 1):
        model.train(True)
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                logits = model(xb)
                loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=use_cuda)
                yb = yb.to(device, non_blocking=use_cuda)
                with autocast(device_type=device.type, enabled=use_cuda):
                    logits = model(xb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / max(total, 1)
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    return model.cpu(), {
        "best_epoch": int(best_epoch),
        "val_acc": float(best_val_acc),
        "test_acc": float((y_pred == y_test).mean()),
        "predictions": y_pred,
    }


def train_fusion_with_preds(
    iq_expert: ExpertCNN,
    fft_expert: ExpertCNN,
    train_loader,
    val_loader,
    test_loader,
    y_test: np.ndarray,
    class_names: list[str],
    epochs: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model = FrozenExpertResidualFusion(iq_expert, fft_expert, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=use_cuda)

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    for epoch in range(1, epochs + 1):
        model.train(True)
        for iqb, fftb, yb in train_loader:
            iqb = iqb.to(device, non_blocking=use_cuda)
            fftb = fftb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iqb, fftb)
                loss = compute_residual_loss(outputs, yb, criterion)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for iqb, fftb, yb in val_loader:
                iqb = iqb.to(device, non_blocking=use_cuda)
                fftb = fftb.to(device, non_blocking=use_cuda)
                yb = yb.to(device, non_blocking=use_cuda)
                with autocast(device_type=device.type, enabled=use_cuda):
                    outputs = model(iqb, fftb)
                correct += (outputs["final_logits"].argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / max(total, 1)
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for iqb, fftb, _ in test_loader:
            iqb = iqb.to(device, non_blocking=use_cuda)
            fftb = fftb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iqb, fftb)
            preds.append(outputs["final_logits"].argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    return {
        "best_epoch": int(best_epoch),
        "val_acc": float(best_val_acc),
        "test_acc": float((y_pred == y_test).mean()),
        "predictions": y_pred,
    }


def build_datasets():
    modulation = load_modulation_dataset()
    modulation["epochs"] = 40

    waveform = load_waveform_dataset()
    waveform["epochs"] = 40

    datasets = [
        modulation,
        waveform,
        build_h5_dataset("subghz_real_128", REAL_TRAIN_PATH, REAL_VAL_PATH, REAL_TEST_PATH, max_windows_per_file=128, batch_size=256, epochs=20),
        build_h5_dataset("subghz_real_512", REAL_TRAIN_PATH, REAL_VAL_PATH, REAL_TEST_PATH, max_windows_per_file=512, batch_size=256, epochs=20),
        build_h5_dataset("subghz_real_1024_40ep", REAL_TRAIN_PATH, REAL_VAL_PATH, REAL_TEST_PATH, max_windows_per_file=1024, batch_size=256, epochs=40),
        build_h5_dataset("subghz_real_augmented_512", AUG_REAL_TRAIN_PATH, AUG_REAL_VAL_PATH, AUG_REAL_TEST_PATH, max_windows_per_file=512, batch_size=256, epochs=20),
        load_orbit_dataset(max_packets_per_node=256),
        build_h5_dataset("captured_npy_real_128", CAPTURED_TRAIN_PATH, CAPTURED_VAL_PATH, CAPTURED_TEST_PATH, max_windows_per_file=128, batch_size=256, epochs=20),
    ]
    datasets[6]["epochs"] = 20
    datasets[6]["batch_size"] = 256
    return datasets


def compute_overlap_summary(y_true: np.ndarray, iq_pred: np.ndarray, fft_pred: np.ndarray, fusion_pred: np.ndarray):
    iq_correct = iq_pred == y_true
    fft_correct = fft_pred == y_true
    fusion_correct = fusion_pred == y_true

    codes = (
        iq_correct.astype(np.int8).astype(str)
        + fft_correct.astype(np.int8).astype(str)
        + fusion_correct.astype(np.int8).astype(str)
    )
    pattern_counts = {code: int((codes == code).sum()) for code in PATTERN_ORDER}

    expert_union = iq_correct | fft_correct
    fusion_covers_union = fusion_correct & expert_union
    fusion_extra = fusion_correct & ~expert_union
    fusion_misses_union = expert_union & ~fusion_correct

    return {
        "num_test_examples": int(len(y_true)),
        "iq_correct": int(iq_correct.sum()),
        "fft_correct": int(fft_correct.sum()),
        "fusion_correct": int(fusion_correct.sum()),
        "expert_union_correct": int(expert_union.sum()),
        "expert_intersection_correct": int((iq_correct & fft_correct).sum()),
        "fusion_covers_expert_union": int(fusion_covers_union.sum()),
        "fusion_misses_from_expert_union": int(fusion_misses_union.sum()),
        "fusion_extra_beyond_expert_union": int(fusion_extra.sum()),
        "pattern_counts": pattern_counts,
    }


def plot_overlap(dataset_name: str, summary: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pattern_values = [summary["pattern_counts"][code] for code in PATTERN_ORDER]
    pattern_labels = [PATTERN_LABELS[code] for code in PATTERN_ORDER]
    bars0 = axes[0].bar(pattern_labels, pattern_values, color="#4C78A8")
    axes[0].set_title(f"{dataset_name}: Correctness Overlap")
    axes[0].set_ylabel("Test samples")
    axes[0].set_yscale("log")
    axes[0].tick_params(axis="x", labelrotation=45)
    for label in axes[0].get_xticklabels():
        label.set_horizontalalignment("right")

    summary_labels = [
        "IQ correct",
        "FFT correct",
        "Fusion correct",
        "Expert union",
        "Fusion covers union",
        "Fusion extra",
        "Fusion misses union",
    ]
    summary_values = [
        summary["iq_correct"],
        summary["fft_correct"],
        summary["fusion_correct"],
        summary["expert_union_correct"],
        summary["fusion_covers_expert_union"],
        summary["fusion_extra_beyond_expert_union"],
        summary["fusion_misses_from_expert_union"],
    ]
    summary_colors = ["#72B7B2", "#F58518", "#54A24B", "#B279A2", "#54A24B", "#E45756", "#9D755D"]
    bars1 = axes[1].bar(summary_labels, summary_values, color=summary_colors)
    axes[1].set_title(f"{dataset_name}: Union Coverage Summary")
    axes[1].set_ylabel("Test samples")
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", labelrotation=45)
    for label in axes[1].get_xticklabels():
        label.set_horizontalalignment("right")

    def annotate_bars(ax, bars, values):
        for bar, value in zip(bars, values):
            if value <= 0:
                y = 1.05
                text = "0"
            else:
                y = value * 1.08
                text = str(int(value))
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                text,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    annotate_bars(axes[0], bars0, pattern_values)
    annotate_bars(axes[1], bars1, summary_values)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def regenerate_plots_from_results(results_path: Path):
    data = json.loads(results_path.read_text())
    for dataset_name, dataset_data in data["datasets"].items():
        plot_path = Path(dataset_data["plot_path"])
        plot_overlap(dataset_name, dataset_data["overlap"], plot_path)
        print(f"Updated {plot_path}")


def run_dataset(dataset: dict):
    name = dataset["name"]
    class_names = dataset["class_names"]
    iq_train = dataset["iq_train"]
    iq_val = dataset["iq_val"]
    iq_test = dataset["iq_test"]
    y_train = dataset["y_train"]
    y_val = dataset["y_val"]
    y_test = dataset["y_test"]
    batch_size = dataset["batch_size"]
    epochs = dataset["epochs"]

    print(f"\n=== Dataset: {name} ===")
    fft_train = to_fft(iq_train)
    fft_val = to_fft(iq_val)
    fft_test = to_fft(iq_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    iq_train_loader = make_loader(iq_train, y_train, True, use_cuda, batch_size)
    iq_val_loader = make_loader(iq_val, y_val, False, use_cuda, batch_size)
    iq_test_loader = make_loader(iq_test, y_test, False, use_cuda, batch_size)
    fft_train_loader = make_loader(fft_train, y_train, True, use_cuda, batch_size)
    fft_val_loader = make_loader(fft_val, y_val, False, use_cuda, batch_size)
    fft_test_loader = make_loader(fft_test, y_test, False, use_cuda, batch_size)
    pair_train_loader = make_pair_loader(iq_train, fft_train, y_train, True, use_cuda, batch_size)
    pair_val_loader = make_pair_loader(iq_val, fft_val, y_val, False, use_cuda, batch_size)
    pair_test_loader = make_pair_loader(iq_test, fft_test, y_test, False, use_cuda, batch_size)

    iq_expert, iq_result = train_single_expert_with_preds(iq_train_loader, iq_val_loader, iq_test_loader, y_test, class_names, epochs)
    fft_expert, fft_result = train_single_expert_with_preds(fft_train_loader, fft_val_loader, fft_test_loader, y_test, class_names, epochs)
    fusion_result = train_fusion_with_preds(iq_expert, fft_expert, pair_train_loader, pair_val_loader, pair_test_loader, y_test, class_names, epochs)

    overlap = compute_overlap_summary(y_test, iq_result["predictions"], fft_result["predictions"], fusion_result["predictions"])
    plot_path = PLOTS_DIR / f"{name}_overlap.png"
    plot_overlap(name, overlap, plot_path)

    return {
        "train_examples": int(iq_train.shape[0]),
        "val_examples": int(iq_val.shape[0]),
        "test_examples": int(iq_test.shape[0]),
        "signal_length": int(iq_train.shape[-1]),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "iq_expert": {k: v for k, v in iq_result.items() if k != "predictions"},
        "fft_expert": {k: v for k, v in fft_result.items() if k != "predictions"},
        "fusion_model": {k: v for k, v in fusion_result.items() if k != "predictions"},
        "overlap": overlap,
        "plot_path": str(plot_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze correctness-set overlap for the frozen-expert residual fusion experiment.")
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--plots-only", action="store_true", help="Regenerate plots from an existing results JSON without rerunning models.")
    args = parser.parse_args()

    if args.plots_only:
        regenerate_plots_from_results(args.results_path)
        return

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    start = time.perf_counter()

    results = {
        "experiment": "experiment11_correct_set_overlap_analysis",
        "datasets": {},
    }
    for dataset in build_datasets():
        dataset_start = time.perf_counter()
        dataset_results = run_dataset(dataset)
        dataset_results["runtime_seconds"] = time.perf_counter() - dataset_start
        results["datasets"][dataset["name"]] = dataset_results

    results["runtime_seconds"] = time.perf_counter() - start
    args.results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {args.results_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
