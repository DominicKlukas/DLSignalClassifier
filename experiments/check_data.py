from __future__ import annotations

from experiments.shared.story_datasets import story_dependency_report


def main() -> None:
    report = story_dependency_report()

    print("Dataset dependency audit")
    print("")
    for name, info in report.items():
        status = "OK" if info["available"] else "MISSING"
        print(f"[{status}] {name}")
        print(f"  {info['description']}")
        for path in info["paths"]:
            marker = "present" if path.exists() else "missing"
            print(f"  - {path} ({marker})")
        print("")

    print("Notes")
    print("  - Experiment 1 can run from a clean clone because it generates its synthetic datasets.")
    print("  - Experiment 2 requires the Experiment 1 synthetic artifacts.")
    print("  - Experiment 3 requires the Sub-GHz HDF5 splits, ORBIT day files, and captured RTL-SDR HDF5 splits.")
    print("  - See docs/datasets.md for dataset provenance and placement details.")


if __name__ == "__main__":
    main()
