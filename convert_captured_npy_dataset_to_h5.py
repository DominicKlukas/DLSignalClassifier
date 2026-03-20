from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


DEFAULT_INPUT_DIR = "CapturedData/dataset"
DEFAULT_OUTPUT_DIR = "data/captured_npy_dataset_experiment5"
DEFAULT_WINDOW_LENGTH = 1024
DEFAULT_MAX_WINDOWS_PER_FILE = 128
DEFAULT_RANDOM_SEED = 0


@dataclass(frozen=True)
class SourceFile:
    path: Path
    protocol: str
    center_freq_hz: float
    sample_rate_hz: int
    gain_db: float
    clip_duration_s: float
    mean_power_db: float
    timestamp: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert captured complex .npy recordings into Experiment 5-style train/val/test HDF5 splits."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Root directory containing the captured dataset.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory where train/val/test HDF5 files will be written.")
    parser.add_argument("--window-length", type=int, default=DEFAULT_WINDOW_LENGTH, help="Slice length for each IQ example.")
    parser.add_argument(
        "--max-windows-per-file",
        type=int,
        default=DEFAULT_MAX_WINDOWS_PER_FILE,
        help="Maximum evenly spaced non-overlapping windows to keep from each source file.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for file-level split assignment.")
    return parser.parse_args()


def normalize_relative_path(raw_path: str) -> Path:
    return Path(raw_path.replace("\\", "/"))


def discover_files(root: Path) -> list[SourceFile]:
    metadata_path = root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.csv under {root}")

    files: list[SourceFile] = []
    with metadata_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rel_path = normalize_relative_path(row["path"])
            path = root / rel_path
            if not path.exists():
                raise FileNotFoundError(f"Metadata entry points to a missing file: {path}")
            files.append(
                SourceFile(
                    path=path,
                    protocol=row["label"],
                    center_freq_hz=float(row["center_freq_hz"]),
                    sample_rate_hz=int(float(row["sample_rate_hz"])),
                    gain_db=float(row["gain_db"]),
                    clip_duration_s=float(row["clip_duration_s"]),
                    mean_power_db=float(row["mean_power_db"]),
                    timestamp=row["timestamp"],
                )
            )

    if not files:
        raise ValueError(f"No recordings found in {metadata_path}")
    return files


def assign_splits(files: list[SourceFile], seed: int) -> dict[str, list[SourceFile]]:
    grouped: dict[str, list[SourceFile]] = defaultdict(list)
    for item in files:
        grouped[item.protocol].append(item)

    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}

    for protocol, items in sorted(grouped.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_val = max(1, n_total // 5)
        n_test = max(1, n_total // 5)
        if n_val + n_test >= n_total:
            if n_total < 3:
                raise ValueError(f"Protocol {protocol} has too few files for a 60/20/20 split")
            n_val = 1
            n_test = 1
        n_train = n_total - n_val - n_test

        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train:n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val:])

    return splits


def select_window_starts(num_samples: int, window_length: int, max_windows_per_file: int) -> np.ndarray:
    if num_samples < window_length:
        return np.asarray([], dtype=np.int64)
    num_windows = num_samples // window_length
    if num_windows <= 0:
        return np.asarray([], dtype=np.int64)
    starts = np.arange(num_windows, dtype=np.int64) * window_length
    if len(starts) <= max_windows_per_file:
        return starts
    positions = np.linspace(0, len(starts) - 1, num=max_windows_per_file, dtype=np.int64)
    return starts[positions]


def normalize_window(window: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(np.abs(window) ** 2) + 1e-8)
    return window / rms


def encode_strings(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=h5py.string_dtype(encoding="utf-8"))


def write_split(
    output_path: Path,
    files: list[SourceFile],
    class_names: list[str],
    class_to_index: dict[str, int],
    window_length: int,
    max_windows_per_file: int,
) -> tuple[int, Counter]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_counter: Counter[str] = Counter()

    with h5py.File(output_path, "w") as h5_file:
        signals = h5_file.create_dataset(
            "signals",
            shape=(0, 2, window_length),
            maxshape=(None, 2, window_length),
            dtype=np.float32,
            chunks=(256, 2, window_length),
        )
        labels = h5_file.create_dataset(
            "labels",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True,
        )
        metadata_group = h5_file.create_group("metadata")
        file_paths = metadata_group.create_dataset(
            "source_file",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=True,
        )
        protocols = metadata_group.create_dataset(
            "protocol",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=True,
        )
        center_freqs = metadata_group.create_dataset(
            "center_freq_hz",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=True,
        )
        sample_rates = metadata_group.create_dataset(
            "sample_rate_hz",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True,
        )
        gains = metadata_group.create_dataset(
            "gain_db",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=True,
        )
        mean_powers = metadata_group.create_dataset(
            "mean_power_db",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=True,
        )
        timestamps = metadata_group.create_dataset(
            "timestamp",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=True,
        )
        window_starts = metadata_group.create_dataset(
            "window_start",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True,
        )

        write_index = 0
        for item in files:
            iq = np.load(item.path)
            if not np.iscomplexobj(iq):
                raise ValueError(f"Expected complex-valued .npy file: {item.path}")
            iq = np.asarray(iq, dtype=np.complex64).reshape(-1)
            starts = select_window_starts(
                num_samples=iq.shape[0],
                window_length=window_length,
                max_windows_per_file=max_windows_per_file,
            )
            if starts.size == 0:
                continue

            num_windows = int(starts.size)
            end_index = write_index + num_windows

            signals.resize((end_index, 2, window_length))
            labels.resize((end_index,))
            file_paths.resize((end_index,))
            protocols.resize((end_index,))
            center_freqs.resize((end_index,))
            sample_rates.resize((end_index,))
            gains.resize((end_index,))
            mean_powers.resize((end_index,))
            timestamps.resize((end_index,))
            window_starts.resize((end_index,))

            split_windows = np.empty((num_windows, 2, window_length), dtype=np.float32)
            for local_idx, start in enumerate(starts):
                normalized = normalize_window(iq[start:start + window_length])
                split_windows[local_idx, 0, :] = normalized.real
                split_windows[local_idx, 1, :] = normalized.imag

            label_index = class_to_index[item.protocol]
            signals[write_index:end_index] = split_windows
            labels[write_index:end_index] = label_index
            file_paths[write_index:end_index] = [str(item.path)] * num_windows
            protocols[write_index:end_index] = [item.protocol] * num_windows
            center_freqs[write_index:end_index] = item.center_freq_hz
            sample_rates[write_index:end_index] = item.sample_rate_hz
            gains[write_index:end_index] = item.gain_db
            mean_powers[write_index:end_index] = item.mean_power_db
            timestamps[write_index:end_index] = [item.timestamp] * num_windows
            window_starts[write_index:end_index] = starts
            write_index = end_index
            protocol_counter[item.protocol] += num_windows

        h5_file.attrs["class_names"] = encode_strings(class_names)
        h5_file.attrs["signal_channels"] = encode_strings(["I", "Q"])
        h5_file.attrs["signal_layout"] = "NCH"
        h5_file.attrs["window_length"] = int(window_length)
        h5_file.attrs["max_windows_per_file"] = int(max_windows_per_file)
        h5_file.attrs["normalization"] = "per-window complex RMS; shared scalar across I/Q"
        h5_file.attrs["split_basis"] = "file-level stratified by protocol"
        h5_file.attrs["window_selection"] = "non-overlapping windows, evenly spaced when capped"

    return write_index, protocol_counter


def summarize_split(name: str, files: list[SourceFile], protocol_windows: Counter[str]) -> str:
    file_counts = Counter(item.protocol for item in files)
    file_text = ", ".join(f"{protocol}={count}" for protocol, count in sorted(file_counts.items()))
    window_text = ", ".join(f"{protocol}={count}" for protocol, count in sorted(protocol_windows.items()))
    return f"{name}: files[{file_text}] windows[{window_text}]"


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.window_length <= 0 or args.max_windows_per_file <= 0:
        raise SystemExit("window-length and max-windows-per-file must be positive")

    files = discover_files(input_dir)
    splits = assign_splits(files, seed=args.seed)
    class_names = sorted({item.protocol for item in files})
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    print(f"Discovered {len(files)} files across classes: {class_names}")
    for split_name in ("train", "val", "test"):
        output_path = output_dir / f"{split_name}.h5"
        num_windows, protocol_windows = write_split(
            output_path=output_path,
            files=splits[split_name],
            class_names=class_names,
            class_to_index=class_to_index,
            window_length=args.window_length,
            max_windows_per_file=args.max_windows_per_file,
        )
        print(f"Wrote {output_path} with {num_windows} windows")
        print(summarize_split(split_name, splits[split_name], protocol_windows))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
