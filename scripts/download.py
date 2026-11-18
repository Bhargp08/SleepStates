"""Download Sleep-EDF Expanded nights and cache 30 s epochs as .npz files (SS-01)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `uv run python scripts/download.py` without installing the project first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleepstates import data

DEFAULT_SUBJECTS = 20


def parse_subjects(spec: str) -> list[int]:
    """Parse "0-19", "0,3,5" or "0-5,9" into a sorted list of subject ids."""
    subjects: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            first, last = chunk.split("-", 1)
            subjects.update(range(int(first), int(last) + 1))
        else:
            subjects.add(int(chunk))
    return sorted(subjects)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        default=f"0-{DEFAULT_SUBJECTS - 1}",
        help=f"subject ids to download, e.g. 0-19 (default: 0-{DEFAULT_SUBJECTS - 1})",
    )
    parser.add_argument("--night", type=int, default=1, help="recording night (default: 1)")
    parser.add_argument("--data-dir", default="data/raw", help="where mne caches downloads")
    parser.add_argument("--out-dir", default="data/epochs", help="where epoch .npz files go")
    parser.add_argument("--force", action="store_true", help="re-epoch nights already saved")
    args = parser.parse_args(argv)

    subjects = parse_subjects(args.subjects)
    out_dir = Path(args.out_dir)

    # Ages come from the dataset's own subject table; the analysis degrades
    # gracefully when that table cannot be downloaded.
    ages: dict[tuple[int, int], float] = {}
    try:
        ages = data.subject_ages(data.load_subject_table(data.fetch_subject_table(args.data_dir)))
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"warning: no age metadata ({exc})")

    for subject in subjects:
        path = out_dir / f"sub-{subject:02d}_night{args.night}.npz"
        if path.exists() and not args.force:
            print(f"skip {path} (already downloaded)")
            continue
        print(f"fetching subject {subject} night {args.night} ...", flush=True)
        record = data.build_subject(
            subject,
            night=args.night,
            data_dir=args.data_dir,
            age=ages.get((subject, args.night)),
        )
        data.save_subject(path, record)
        counts = {stage: int((record["labels"] == stage).sum()) for stage in data.STAGES}
        print(f"  saved {path} ({len(record['labels'])} epochs) {counts}")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
