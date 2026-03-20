from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
from scipy.io import whosmat


@dataclass
class EntrySummary:
    name: str
    kind: str
    shape: tuple[int, ...] | None
    dtype: str | None


@dataclass
class MatFileSummary:
    path: Path
    size_bytes: int
    format_name: str
    entries: list[EntrySummary]
    notes: list[str]
    inference: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect MATLAB .mat files and summarize what they contain."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="File or directory to inspect. Defaults to the current directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories when the target is a directory.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=25,
        help="Maximum number of variables or HDF5 objects to print per file.",
    )
    return parser.parse_args()


def iter_mat_files(target: Path, recursive: bool) -> Iterable[Path]:
    if target.is_file():
        if target.suffix.lower() == ".mat":
            yield target
        return

    pattern = "**/*.mat" if recursive else "*.mat"
    for path in sorted(target.glob(pattern)):
        if path.is_file():
            yield path


def format_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


def describe_classic_mat(path: Path) -> MatFileSummary:
    variables = whosmat(str(path))
    entries = [
        EntrySummary(name=name, shape=tuple(shape), dtype=class_name, kind="variable")
        for name, shape, class_name in variables
    ]
    return MatFileSummary(
        path=path,
        size_bytes=path.stat().st_size,
        format_name="MATLAB Level 4/5/7.2",
        entries=entries,
        notes=[],
        inference=infer_file_purpose(entries),
    )


def describe_hdf5_mat(path: Path) -> MatFileSummary:
    entries: list[EntrySummary] = []
    notes: list[str] = []

    with h5py.File(path, "r") as handle:
        matlab_class = handle.attrs.get("MATLAB_class")
        if matlab_class is not None:
            notes.append(f"Root MATLAB_class={decode_attr(matlab_class)}")

        def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if isinstance(obj, h5py.Dataset):
                entries.append(
                    EntrySummary(
                        name=name,
                        kind="dataset",
                        shape=tuple(int(dim) for dim in obj.shape),
                        dtype=str(obj.dtype),
                    )
                )
            else:
                entries.append(
                    EntrySummary(
                        name=name,
                        kind="group",
                        shape=None,
                        dtype=None,
                    )
                )

        handle.visititems(visitor)

    return MatFileSummary(
        path=path,
        size_bytes=path.stat().st_size,
        format_name="MATLAB v7.3 (HDF5)",
        entries=entries,
        notes=notes,
        inference=infer_file_purpose(entries),
    )


def decode_attr(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return str(converted)
    return str(value)


def infer_file_purpose(entries: list[EntrySummary]) -> str:
    if not entries:
        return "empty or unreadable"

    visible_entries = [entry for entry in entries if not entry.name.startswith("__")]
    function_like = [entry for entry in visible_entries if entry.dtype == "function"]
    dataset_like = [
        entry
        for entry in visible_entries
        if entry.shape
        and len(entry.shape) >= 2
        and any(dim >= 32 for dim in entry.shape)
        and entry.kind in {"variable", "dataset"}
    ]
    struct_like = [entry for entry in visible_entries if entry.dtype in {"struct", "cell"}]
    numeric_like = [
        entry
        for entry in visible_entries
        if entry.dtype
        and any(token in entry.dtype.lower() for token in ("int", "float", "double", "single", "uint"))
    ]

    if function_like:
        return "contains MATLAB function objects or serialized function state"
    if dataset_like and len(dataset_like) == 1 and len(visible_entries) <= 4:
        main = dataset_like[0]
        return f"likely a single primary array dataset centered on `{main.name}`"
    if dataset_like and len(dataset_like) > 1:
        return "likely a multi-array dataset or feature bundle"
    if struct_like:
        return "likely MATLAB structs or cell containers with nested metadata"
    if numeric_like:
        return "mostly numeric variables; possibly lookup tables or small arrays"
    return "mixed contents; inspect variable names for domain meaning"


def inspect_mat_file(path: Path) -> MatFileSummary:
    if h5py.is_hdf5(path):
        return describe_hdf5_mat(path)
    return describe_classic_mat(path)


def print_summary(summary: MatFileSummary, max_entries: int) -> None:
    print(f"File: {summary.path}")
    print(f"Size: {format_size(summary.size_bytes)}")
    print(f"Format: {summary.format_name}")
    print(f"Inference: {summary.inference}")
    if summary.notes:
        for note in summary.notes:
            print(f"Note: {note}")
    if not summary.entries:
        print("Entries: none")
        print()
        return

    print(f"Entries ({len(summary.entries)} total):")
    for entry in summary.entries[:max_entries]:
        shape_text = str(entry.shape) if entry.shape is not None else "-"
        dtype_text = entry.dtype or "-"
        print(f"  - {entry.name}: kind={entry.kind}, shape={shape_text}, dtype={dtype_text}")
    if len(summary.entries) > max_entries:
        remaining = len(summary.entries) - max_entries
        print(f"  ... {remaining} more entries not shown")
    print()


def main() -> int:
    args = parse_args()
    target = Path(args.path).expanduser().resolve()
    if not target.exists():
        raise SystemExit(f"Path does not exist: {target}")

    mat_files = list(iter_mat_files(target, recursive=args.recursive))
    if not mat_files:
        print(f"No .mat files found under {target}")
        return 0

    failures: list[tuple[Path, Exception]] = []
    for mat_file in mat_files:
        try:
            summary = inspect_mat_file(mat_file)
            print_summary(summary, max_entries=args.max_entries)
        except Exception as exc:
            failures.append((mat_file, exc))

    if failures:
        print("Unreadable files:")
        for path, exc in failures:
            print(f"  - {path}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
