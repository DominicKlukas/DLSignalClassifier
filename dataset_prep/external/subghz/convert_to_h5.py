from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy.io import loadmat, whosmat


DEFAULT_WINDOW_LENGTH = 1024
DEFAULT_RANDOM_SEED = 0
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SourceFile:
    path: Path
    band: str
    protocol: str
    variant: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MATLAB IQ capture files into train/val/test HDF5 datasets."
    )
    parser.add_argument(
        "--input-dir",
        default=str(REPO_ROOT / "external" / "subghz_raw" / "Dataset"),
        help="Root directory containing the .mat dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "real_mat_dataset"),
        help="Directory where train/val/test HDF5 files will be written.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=DEFAULT_WINDOW_LENGTH,
        help="Slice length for each IQ example.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_WINDOW_LENGTH,
        help="Stride between successive windows. Default is non-overlapping windows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for file-level split assignment.",
    )
    return parser.parse_args()


def discover_files(root: Path) -> list[SourceFile]:
    files: list[SourceFile] = []
    for path in sorted(root.rglob("*.mat")):
        rel = path.relative_to(root)
        if len(rel.parts) < 3:
            continue
        band = rel.parts[0]
        protocol = rel.parts[1]
        variant = path.stem.rsplit("_", 1)[0]
        files.append(SourceFile(path=path, band=band, protocol=protocol, variant=variant))
    if not files:
        raise ValueError(f"No .mat files found under {root}")
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
                raise ValueError(f"Protocol {protocol} has too few files for 60/20/20 split")
            n_val = 1
            n_test = 1
        n_train = n_total - n_val - n_test

        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train:n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val:])

    return splits


def load_iq_samples(path: Path) -> np.ndarray:
    mat = loadmat(path, variable_names=["IQ_samples"])
    if "IQ_samples" not in mat:
        raise ValueError(f"{path} does not contain IQ_samples")
    iq = np.asarray(mat["IQ_samples"]).reshape(-1)
    if not np.iscomplexobj(iq):
        iq = iq.astype(np.float32) + 0j
    return iq.astype(np.complex64, copy=False)


def iter_windows(iq: np.ndarray, window_length: int, stride: int) -> Iterable[tuple[int, np.ndarray]]:
    max_start = iq.shape[0] - window_length
    for start in range(0, max_start + 1, stride):
        yield start, iq[start:start + window_length]


def normalize_window(window: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(np.abs(window) ** 2) + 1e-8)
    return window / rms


def encode_strings(values: list[str]) -> np.ndarray:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    return np.asarray(values, dtype=string_dtype)


def write_split(
    output_path: Path,
    files: list[SourceFile],
    class_names: list[str],
    class_to_index: dict[str, int],
    window_length: int,
    stride: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        bands = metadata_group.create_dataset(
            "band",
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
        variants = metadata_group.create_dataset(
            "variant",
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
        signal_length = 0
        for item in files:
            iq = load_iq_samples(item.path)
            signal_length = int(iq.shape[0])
            label = class_to_index[item.protocol]
            if iq.shape[0] < window_length:
                continue

            num_windows = 1 + (iq.shape[0] - window_length) // stride
            starts = np.arange(num_windows, dtype=np.int64) * stride
            end_index = write_index + num_windows

            signals.resize((end_index, 2, window_length))
            labels.resize((end_index,))
            file_paths.resize((end_index,))
            bands.resize((end_index,))
            protocols.resize((end_index,))
            variants.resize((end_index,))
            window_starts.resize((end_index,))

            split_windows = np.empty((num_windows, 2, window_length), dtype=np.float32)
            for local_idx, start in enumerate(starts):
                normalized = normalize_window(iq[start:start + window_length])
                split_windows[local_idx, 0, :] = np.real(normalized)
                split_windows[local_idx, 1, :] = np.imag(normalized)

            signals[write_index:end_index] = split_windows
            labels[write_index:end_index] = label
            file_paths[write_index:end_index] = [str(item.path)] * num_windows
            bands[write_index:end_index] = [item.band] * num_windows
            protocols[write_index:end_index] = [item.protocol] * num_windows
            variants[write_index:end_index] = [item.variant] * num_windows
            window_starts[write_index:end_index] = starts
            write_index = end_index

        h5_file.attrs["class_names"] = encode_strings(class_names)
        h5_file.attrs["signal_channels"] = encode_strings(["I", "Q"])
        h5_file.attrs["signal_layout"] = "NCH"
        h5_file.attrs["window_length"] = int(window_length)
        h5_file.attrs["window_stride"] = int(stride)
        h5_file.attrs["source_signal_length"] = int(signal_length)
        h5_file.attrs["normalization"] = "per-window complex RMS; shared scalar across I/Q"
        h5_file.attrs["split_basis"] = "file-level stratified by protocol"


def summarize_split(name: str, files: list[SourceFile], window_length: int, stride: int) -> str:
    file_counts = Counter(item.protocol for item in files)
    window_counts: Counter[str] = Counter()
    for item in files:
        signal_length = int(whosmat(item.path)[0][1][1])  # IQ_samples is stored as (1, N)
        window_counts[item.protocol] += 1 + (signal_length - window_length) // stride
    file_text = ", ".join(f"{protocol}={count}" for protocol, count in sorted(file_counts.items()))
    window_text = ", ".join(f"{protocol}={count}" for protocol, count in sorted(window_counts.items()))
    return f"{name}: files[{file_text}] windows[{window_text}]"


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.window_length <= 0 or args.stride <= 0:
        raise SystemExit("window-length and stride must be positive")

    files = discover_files(input_dir)
    splits = assign_splits(files, seed=args.seed)
    class_names = sorted({item.protocol for item in files})
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    print(f"Discovered {len(files)} source files under {input_dir}")
    print(f"Classes: {', '.join(class_names)}")
    for split_name in ("train", "val", "test"):
        print(summarize_split(split_name, splits[split_name], args.window_length, args.stride))

    for split_name in ("train", "val", "test"):
        output_path = output_dir / f"{split_name}.h5"
        print(f"Writing {split_name} to {output_path}")
        write_split(
            output_path=output_path,
            files=splits[split_name],
            class_names=class_names,
            class_to_index=class_to_index,
            window_length=args.window_length,
            stride=args.stride,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
