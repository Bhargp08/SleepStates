"""Compute per-epoch spectral features for every cached subject (SS-02)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleepstates import data, features


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch-dir", default="data/epochs", help="cached subject .npz files")
    parser.add_argument("--out", default="data/features.npz", help="output feature dataset")
    args = parser.parse_args(argv)

    records = data.load_epoch_dir(args.epoch_dir)
    if not records:
        print(f"no epoch files in {args.epoch_dir}; run scripts/download.py first")
        return 1

    dataset = features.build_feature_dataset(records)
    path = features.save_feature_dataset(args.out, dataset)

    print(f"subjects: {len(records)}  epochs: {dataset['X'].shape[0]}")
    print("features: " + ", ".join(dataset["feature_names"]))
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
