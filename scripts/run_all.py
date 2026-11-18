"""Run the full SleepStates pipeline (SS-01 to SS-12) with one command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent


def build_plan(
    subjects: str = "0-19",
    night: int = 1,
    best_k: int | None = None,
    data_dir: str = "data/raw",
    out_dir: str = "outputs",
    include_baseline: bool = True,
) -> list[tuple[str, list[str]]]:
    """Ordered (step name, command) pairs for the whole pipeline."""
    epochs = "data/epochs"
    features = "data/features.npz"
    model = "data/model.npz"
    states = "data/states.npz"

    plan: list[tuple[str, list[str]]] = [
        (
            "download and epoch Sleep-EDF (SS-01)",
            [
                "download.py",
                "--subjects",
                subjects,
                "--night",
                str(night),
                "--data-dir",
                data_dir,
                "--out-dir",
                epochs,
            ],
        ),
        (
            "spectral features (SS-02)",
            ["extract_features.py", "--epoch-dir", epochs, "--out", features],
        ),
        (
            "K sweep: BIC + held-out likelihood (SS-08)",
            ["select_k.py", "--features", features, "--out-dir", out_dir],
        ),
    ]

    fit_command = [
        "fit_models.py",
        "--features",
        features,
        "--model-out",
        model,
        "--states-out",
        states,
    ]
    if best_k is not None:
        fit_command += ["--n-states", str(best_k)]
    plan.append(("PCA, GMM and Baum-Welch (SS-03 to SS-07)", fit_command))

    plan.append(
        (
            "match states to expert stages (SS-09)",
            ["evaluate.py", "--states", states, "--out-dir", out_dir],
        )
    )
    plan.append(
        (
            "transitions, dwell times and age (SS-10)",
            ["analyze.py", "--states", states, "--model", model, "--out-dir", out_dir],
        )
    )
    if include_baseline:
        plan.append(
            (
                "YASA supervised baseline (SS-11)",
                ["baseline.py", "--epoch-dir", epochs, "--out-dir", out_dir],
            )
        )
    plan.append(
        (
            "hypnogram viewer (SS-12)",
            ["viewer.py", "--states", states, "--out", f"{out_dir}/hypnogram.html"],
        )
    )

    return [(name, [str(SCRIPTS / command[0]), *command[1:]]) for name, command in plan]


def read_best_k(path) -> int | None:
    """Read the BIC-selected K, or None when the sweep has not run."""
    path = Path(path)
    if not path.exists():
        return None
    return int(json.loads(path.read_text())["k_by_bic"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default="0-19", help="subject ids, e.g. 0-19")
    parser.add_argument("--night", type=int, default=1, help="recording night")
    parser.add_argument("--data-dir", default="data/raw", help="where mne caches EDF files")
    parser.add_argument("--out-dir", default="outputs", help="figure and table directory")
    parser.add_argument("--best-k", type=int, default=None, help="states (default: from the sweep)")
    parser.add_argument("--skip-baseline", action="store_true", help="skip the YASA baseline")
    args = parser.parse_args(argv)

    plan = build_plan(
        subjects=args.subjects,
        night=args.night,
        best_k=args.best_k,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        include_baseline=not args.skip_baseline,
    )

    for name, command in plan:
        # The K sweep runs before the fit step, so pick its answer up on the way.
        if Path(command[0]).name == "fit_models.py" and "--n-states" not in command:
            chosen = read_best_k(Path(args.out_dir) / "best_k.json")
            if chosen is not None:
                command = [*command, "--n-states", str(chosen)]
                print(f"using K={chosen} from the sweep")

        print(f"\n== {name}")
        subprocess.run([sys.executable, *command], check=True, cwd=ROOT)

    print("\npipeline finished; see outputs/ for figures and tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
