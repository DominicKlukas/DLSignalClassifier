from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "real_mat_dataset"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "real_mat_dataset_augmented"
DEFAULT_MAX_WINDOWS_PER_FILE = 512
SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an augmented real-data HDF5 dataset from the base Sub-GHz splits.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-windows-per-file", type=int, default=DEFAULT_MAX_WINDOWS_PER_FILE)
    return parser.parse_args()


def selected_relative_indices(length: int, max_windows_per_file: int) -> np.ndarray:
    if length <= max_windows_per_file:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, num=max_windows_per_file, dtype=np.int64)


def normalize_complex(signal: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(np.abs(signal) ** 2) + 1e-8)
    return (signal / rms).astype(np.complex64)


def batched_linear_resample(signals: np.ndarray, scales: np.ndarray) -> np.ndarray:
    batch, length = signals.shape
    center = 0.5 * (length - 1)
    coords = np.arange(length, dtype=np.float32)[None, :]
    positions = center + scales[:, None] * (coords - center)
    positions = np.clip(positions, 0.0, length - 1.001)

    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, length - 1)
    frac = (positions - left).astype(np.float32)
    batch_idx = np.arange(batch)[:, None]

    real = signals.real
    imag = signals.imag
    real_interp = (1.0 - frac) * real[batch_idx, left] + frac * real[batch_idx, right]
    imag_interp = (1.0 - frac) * imag[batch_idx, left] + frac * imag[batch_idx, right]
    return (real_interp + 1j * imag_interp).astype(np.complex64)


def augment_signals(signal_2ch: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    complex_signals = signal_2ch[:, 0, :].astype(np.float32) + 1j * signal_2ch[:, 1, :].astype(np.float32)
    batch, length = complex_signals.shape

    omega = rng.uniform(-0.18, 0.18, size=(batch,)).astype(np.float32)
    scale = rng.uniform(0.90, 1.10, size=(batch,)).astype(np.float32)
    snr_db = rng.uniform(8.0, 24.0, size=(batch,)).astype(np.float32)

    phase = np.exp(1j * omega[:, None] * np.arange(length, dtype=np.float32)[None, :]).astype(np.complex64)
    shifted = complex_signals * phase
    resampled = batched_linear_resample(shifted, scales=scale)

    signal_power = np.mean(np.abs(resampled) ** 2, axis=1).astype(np.float32)
    snr_linear = np.power(10.0, snr_db / 10.0, dtype=np.float32)
    sigma = np.sqrt(signal_power / np.maximum(snr_linear, 1e-8) / 2.0, dtype=np.float32)
    noise = (
        rng.normal(0.0, 1.0, size=resampled.shape).astype(np.float32)
        + 1j * rng.normal(0.0, 1.0, size=resampled.shape).astype(np.float32)
    ) * sigma[:, None]
    augmented = (resampled + noise).astype(np.complex64)

    rms = np.sqrt(np.mean(np.abs(augmented) ** 2, axis=1, keepdims=True) + 1e-8).astype(np.float32)
    augmented = (augmented / rms).astype(np.complex64)
    stacked = np.stack((augmented.real, augmented.imag), axis=1).astype(np.float32)
    return stacked, omega, scale, snr_db


def encode_strings(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=h5py.string_dtype(encoding="utf-8"))


def build_split(input_path: Path, output_path: Path, max_windows_per_file: int, seed_offset: int) -> dict[str, int]:
    rng = np.random.default_rng(SEED + seed_offset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as src:
        source_files = src["metadata"]["source_file"][:]
        class_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in src.attrs["class_names"]
        ]
        decoded_files = np.asarray(
            [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in source_files],
            dtype=object,
        )

        starts = [0]
        for idx in range(1, len(decoded_files)):
            if decoded_files[idx] != decoded_files[idx - 1]:
                starts.append(idx)
        starts.append(len(decoded_files))

        signal_parts = []
        label_parts = []
        band_parts = []
        protocol_parts = []
        variant_parts = []
        source_file_parts = []
        window_start_parts = []

        for block_idx in range(len(starts) - 1):
            start = starts[block_idx]
            end = starts[block_idx + 1]
            rel_idx = selected_relative_indices(end - start, max_windows_per_file=max_windows_per_file)

            signal_parts.append(src["signals"][start:end][rel_idx].astype(np.float32))
            label_parts.append(src["labels"][start:end][rel_idx].astype(np.int64))
            band_parts.append(src["metadata"]["band"][start:end][rel_idx])
            protocol_parts.append(src["metadata"]["protocol"][start:end][rel_idx])
            variant_parts.append(src["metadata"]["variant"][start:end][rel_idx])
            source_file_parts.append(src["metadata"]["source_file"][start:end][rel_idx])
            window_start_parts.append(src["metadata"]["window_start"][start:end][rel_idx])

        signals = np.concatenate(signal_parts, axis=0)
        labels = np.concatenate(label_parts, axis=0)
        band = np.concatenate(band_parts, axis=0)
        protocol = np.concatenate(protocol_parts, axis=0)
        variant = np.concatenate(variant_parts, axis=0)
        source_file = np.concatenate(source_file_parts, axis=0)
        window_start = np.concatenate(window_start_parts, axis=0)

    augmented, freq_shift, sample_rate_scale, noise_snr_db = augment_signals(signals, rng=rng)

    with h5py.File(output_path, "w") as dst:
        dst.create_dataset("signals", data=augmented, chunks=True)
        dst.create_dataset("labels", data=labels, chunks=True)
        metadata = dst.create_group("metadata")
        metadata.create_dataset("band", data=band, chunks=True)
        metadata.create_dataset("protocol", data=protocol, chunks=True)
        metadata.create_dataset("variant", data=variant, chunks=True)
        metadata.create_dataset("source_file", data=source_file, chunks=True)
        metadata.create_dataset("window_start", data=window_start, chunks=True)
        metadata.create_dataset("frequency_shift_rad_per_sample", data=freq_shift, chunks=True)
        metadata.create_dataset("sample_rate_scale", data=sample_rate_scale, chunks=True)
        metadata.create_dataset("noise_snr_db", data=noise_snr_db, chunks=True)

        dst.attrs["signal_layout"] = "NCH"
        dst.attrs["signal_channels"] = encode_strings(["I", "Q"])
        dst.attrs["class_names"] = encode_strings(class_names)
        dst.attrs["augmentation"] = "frequency_shift + sample_rate_interpolation + awgn"
        dst.attrs["max_windows_per_file"] = int(max_windows_per_file)
        dst.attrs["normalization"] = "post-augmentation complex RMS; shared scalar across I/Q"

    counts = np.bincount(labels, minlength=len(class_names))
    return {class_names[idx]: int(counts[idx]) for idx in range(len(class_names))}


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    summary = {}
    for split_name, seed_offset in (("train", 1), ("val", 2), ("test", 3)):
        counts = build_split(
            input_path=input_dir / f"{split_name}.h5",
            output_path=output_dir / f"{split_name}.h5",
            max_windows_per_file=args.max_windows_per_file,
            seed_offset=seed_offset,
        )
        summary[split_name] = counts
        print(f"{split_name}: {counts}", flush=True)

    print(f"Saved augmented dataset to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
