"""Sweep the number of HMM states and save BIC / held-out likelihood (SS-08)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleepstates import analysis, features, plots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/features.npz", help="feature dataset")
    parser.add_argument("--out-dir", default="outputs", help="where tables and figures go")
    parser.add_argument("--k-min", type=int, default=3, help="smallest K")
    parser.add_argument("--k-max", type=int, default=8, help="largest K")
    parser.add_argument("--folds", type=int, default=4, help="subject folds for held-out scoring")
    parser.add_argument("--restarts", type=int, default=3, help="GMM restarts per fit")
    parser.add_argument("--components", type=int, default=5, help="PCA dimensions")
    parser.add_argument("--max-iter", type=int, default=100, help="Baum-Welch iterations")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args(argv)

    dataset = features.load_feature_dataset(args.features)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = analysis.sweep_k(
        dataset["X"],
        dataset["lengths"],
        k_values=range(args.k_min, args.k_max + 1),
        n_folds=args.folds,
        restarts=args.restarts,
        max_iter=args.max_iter,
        pca_components=args.components,
        seed=args.seed,
    )

    csv_path = out_dir / "model_selection.csv"
    table.to_csv(csv_path, index=False)
    plots.plot_model_selection(table, out_dir / "model_selection.png")

    best = {
        "k_by_bic": analysis.best_k_by_bic(table),
        "k_by_heldout": analysis.best_k_by_heldout(table),
    }
    (out_dir / "best_k.json").write_text(json.dumps(best, indent=2) + "\n")

    print(table.to_string(index=False))
    print(f"best K by BIC: {best['k_by_bic']}, by held-out likelihood: {best['k_by_heldout']}")
    print(f"saved {csv_path}, {out_dir / 'model_selection.png'}, {out_dir / 'best_k.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
