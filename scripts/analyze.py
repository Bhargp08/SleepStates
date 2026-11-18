"""Transition matrix, dwell times and age comparison (SS-10)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleepstates import analysis, plots


def _write_matrix(matrix: np.ndarray, path: Path) -> None:
    labels = [f"state {index}" for index in range(len(matrix))]
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", default="data/states.npz", help="states from fit_models.py")
    parser.add_argument("--model", default="data/model.npz", help="fitted parameters")
    parser.add_argument("--out-dir", default="outputs", help="where tables and figures go")
    args = parser.parse_args(argv)

    with np.load(args.states, allow_pickle=False) as npz:
        states = npz["states"].astype(int)
        subjects = npz["subjects"].astype(int)
        ages = npz["ages"].astype(float)
    with np.load(args.model, allow_pickle=False) as npz:
        transmat = npz["transmat"].astype(float)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stationary = analysis.stationary_distribution(transmat)
    empirical = analysis.transition_matrix_from_states(states, n_states=len(transmat))
    dwell = analysis.dwell_table(transmat, states)
    age_result = analysis.age_comparison(states, subjects, ages)

    _write_matrix(transmat, out_dir / "transitions.csv")
    _write_matrix(empirical, out_dir / "transitions_empirical.csv")
    plots.plot_transition_matrix(transmat, path=out_dir / "transition_matrix.png")

    dwell.to_csv(out_dir / "dwell_times.csv", index=False)
    plots.plot_dwell_times(dwell, path=out_dir / "dwell_times.png")

    age_result["subjects"].to_csv(out_dir / "age_comparison.csv", index=False)
    age_result["summary"].to_csv(out_dir / "age_summary.csv", index=False)
    if not age_result["subjects"].empty:
        plots.plot_age_comparison(age_result, path=out_dir / "age_comparison.png")

    summary = {
        "stationary_distribution": [float(value) for value in stationary],
        "p_transition_rate": age_result["p_transition_rate"],
        "p_mean_dwell": age_result["p_mean_dwell"],
    }
    (out_dir / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("stationary distribution: " + ", ".join(f"{value:.3f}" for value in stationary))
    print(dwell.to_string(index=False))
    if age_result["subjects"].empty:
        print("no age metadata; skipped the age comparison")
    else:
        print(age_result["summary"].to_string(index=False))
        print(
            f"younger vs older: p(transition rate)={summary['p_transition_rate']:.3f}, "
            f"p(mean dwell)={summary['p_mean_dwell']:.3f}"
        )
    print(f"saved analysis tables and figures in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
