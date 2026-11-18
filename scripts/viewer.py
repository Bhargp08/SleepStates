"""Write outputs/hypnogram.html: expert vs machine hypnograms with a subject dropdown (SS-12)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleepstates import plots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", default="data/states.npz", help="states from fit_models.py")
    parser.add_argument("--out", default="outputs/hypnogram.html", help="output HTML file")
    args = parser.parse_args(argv)

    with np.load(args.states, allow_pickle=False) as npz:
        states = npz["states"].astype(int)
        labels = np.array([str(label) for label in npz["labels"]], dtype=object)
        subjects = npz["subjects"].astype(int)

    figure = plots.build_hypnogram_figure(states, labels, subjects)
    path = plots.write_hypnogram_html(figure, args.out)

    print(f"subjects: {len(np.unique(subjects))}  epochs: {len(states)}")
    print(f"saved {path} (open it in a browser and use the dropdown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
