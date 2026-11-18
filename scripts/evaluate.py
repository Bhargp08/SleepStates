"""Match HMM states to expert stages and write agreement metrics (SS-09)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleepstates import evaluate, plots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", default="data/states.npz", help="states from fit_models.py")
    parser.add_argument("--out-dir", default="outputs", help="where metrics and figures go")
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "hungarian", "majority"],
        help="state matching (auto: hungarian for K<=5, majority above)",
    )
    args = parser.parse_args(argv)

    with np.load(args.states, allow_pickle=False) as npz:
        states = npz["states"].astype(int)
        labels = np.array([str(label) for label in npz["labels"]], dtype=object)

    method = args.method
    if method == "auto":
        method = "hungarian" if states.max() < len(evaluate.STAGES) else "majority"

    result = evaluate.evaluate(states, labels, method=method)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "method": result["method"],
        "mapping": {int(state): stage for state, stage in result["mapping"].items()},
        "accuracy": result["accuracy"],
        "kappa": result["kappa"],
        "adjusted_rand": result["adjusted_rand"],
        "n_epochs": result["n_epochs"],
    }
    (out_dir / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")

    result["per_stage"].to_csv(out_dir / "per_stage.csv", index=False)
    pd.DataFrame(
        result["confusion"], index=[f"state {i}" for i in range(len(result["confusion"]))],
        columns=list(evaluate.STAGES),
    ).to_csv(out_dir / "confusion_matrix.csv")
    plots.plot_confusion_matrix(result["confusion"], path=out_dir / "confusion_matrix.png")

    print(f"matching: {result['method']}, epochs: {result['n_epochs']}")
    print(json.dumps(summary["mapping"], indent=2))
    print(f"accuracy {result['accuracy']:.3f}  kappa {result['kappa']:.3f}  ARI {result['adjusted_rand']:.3f}")
    print(result["per_stage"].to_string(index=False))
    print(f"saved metrics and figures in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
