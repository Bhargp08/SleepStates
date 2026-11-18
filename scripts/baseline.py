"""Run the YASA supervised stager on the same recordings and score it (SS-11)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleepstates import baseline, data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch-dir", default="data/epochs", help="cached subject .npz files")
    parser.add_argument("--data-dir", default="data/raw", help="where mne caches EDF files")
    parser.add_argument("--out-dir", default="outputs", help="where tables go")
    parser.add_argument("--eeg-name", default="Fpz-Cz", help="channel name passed to YASA")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, pooled_predicted, pooled_labels = [], [], []
    for path in sorted(Path(args.epoch_dir).glob("*.npz")):
        record = data.load_subject(path)
        subject, night = record["subject"], record["night"]
        psg_path, hyp_path = data.recording_paths(subject, night, args.data_dir)

        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
        annotations = mne.read_annotations(hyp_path)
        channel = next(
            name for name in raw.ch_names if data.EEG_CHANNEL.lower() in name.lower()
        )
        raw.pick([channel], verbose="ERROR")
        # YASA looks channels up by short names such as Fpz-Cz.
        raw.rename_channels({channel: args.eeg_name})

        n_epochs = raw.n_times // round(data.EPOCH_SECONDS * raw.info["sfreq"])
        labels = data.epoch_labels(annotations, n_epochs)
        predicted = baseline.run_yasa(raw, eeg_name=args.eeg_name, verbose=False)

        length = min(len(predicted), n_epochs)
        labels, predicted = labels[:length], predicted[:length]
        result = baseline.stage_agreement(predicted, labels)

        rows.append(
            {
                "subject": subject,
                "night": night,
                "n_epochs": result["n_epochs"],
                "accuracy": result["accuracy"],
                "kappa": result["kappa"],
                "adjusted_rand": result["adjusted_rand"],
            }
        )
        pooled_predicted.append(np.asarray(predicted, dtype=str))
        pooled_labels.append(np.asarray(labels, dtype=str))
        print(
            f"subject {subject:02d} night {night}: kappa {result['kappa']:.3f} "
            f"accuracy {result['accuracy']:.3f} ({result['n_epochs']} epochs)"
        )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "baseline.csv", index=False)

    pooled = baseline.stage_agreement(
        np.concatenate(pooled_predicted), np.concatenate(pooled_labels)
    )
    pd.DataFrame(
        pooled["confusion"], index=list(baseline.STAGES), columns=list(baseline.STAGES)
    ).to_csv(out_dir / "baseline_confusion.csv")

    summary = {
        "pooled_kappa": pooled["kappa"],
        "pooled_accuracy": pooled["accuracy"],
        "pooled_adjusted_rand": pooled["adjusted_rand"],
        "mean_subject_kappa": float(table["kappa"].mean()) if len(table) else float("nan"),
        "n_subjects": len(table),
    }
    (out_dir / "baseline.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"pooled kappa {pooled['kappa']:.3f}, mean per-subject kappa "
        f"{summary['mean_subject_kappa']:.3f} over {len(table)} nights"
    )
    print(f"saved {out_dir / 'baseline.csv'}, baseline_confusion.csv, baseline.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
