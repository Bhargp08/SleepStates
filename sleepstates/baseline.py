"""YASA supervised staging baseline as the upper bound (SS-11)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score

from sleepstates.evaluate import STAGES


def run_yasa(raw, eeg_name: str = "C4-M1", staging_factory=None, **kwargs) -> np.ndarray:
    """Run yasa.SleepStaging on a raw recording and return its stage labels."""
    if staging_factory is None:
        from yasa import SleepStaging as staging_factory

    model = staging_factory(raw, eeg_name=eeg_name, **kwargs)
    return np.asarray(model.predict(), dtype=object)


def agreement_confusion(predicted, labels, stages=STAGES) -> np.ndarray:
    """Counts with expert stages as rows and YASA stages as columns."""
    table = np.zeros((len(stages), len(stages)), dtype=int)
    for label, prediction in zip(np.asarray(labels, dtype=str), np.asarray(predicted, dtype=str), strict=True):
        if label in stages and prediction in stages:
            table[stages.index(label), stages.index(prediction)] += 1
    return table


def stage_agreement(predicted, labels, stages=STAGES) -> dict:
    """Accuracy, Cohen's kappa and ARI against expert labels (no matching needed)."""
    predicted = np.asarray(predicted, dtype=str)
    labels = np.asarray(labels, dtype=str)
    keep = np.isin(predicted, stages) & np.isin(labels, stages)
    predicted, labels = predicted[keep], labels[keep]

    if not len(labels):
        return {
            "accuracy": float("nan"),
            "kappa": float("nan"),
            "adjusted_rand": float("nan"),
            "n_epochs": 0,
            "confusion": agreement_confusion(predicted, labels, stages),
        }

    return {
        "accuracy": float((predicted == labels).mean()),
        "kappa": float(cohen_kappa_score(labels, predicted)),
        "adjusted_rand": float(adjusted_rand_score(labels, predicted)),
        "n_epochs": len(labels),
        "confusion": agreement_confusion(predicted, labels, stages),
    }
