"""Tests for sleepstates.baseline (SS-11). YASA is stubbed, never trained on data."""

from typing import ClassVar

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from sleepstates import baseline

STAGES = baseline.STAGES


class FakeSleepStaging:
    """Minimal stand-in for yasa.SleepStaging."""

    calls: ClassVar[list[dict]] = []
    stages: ClassVar[list[str]] = ["W", "N1", "N2", "N3", "REM"]

    def __init__(self, raw, eeg_name="C4-M1", **kwargs):
        FakeSleepStaging.calls.append({"raw": raw, "eeg_name": eeg_name, "kwargs": kwargs})
        self.raw = raw
        self.eeg_name = eeg_name

    def predict(self):
        return np.array(FakeSleepStaging.stages, dtype=object)


def test_run_yasa_returns_stage_array():
    FakeSleepStaging.calls.clear()

    stages = baseline.run_yasa("raw-object", eeg_name="Fpz-Cz", staging_factory=FakeSleepStaging)

    assert list(stages) == FakeSleepStaging.stages
    assert FakeSleepStaging.calls[0]["raw"] == "raw-object"
    assert FakeSleepStaging.calls[0]["eeg_name"] == "Fpz-Cz"


def test_run_yasa_passes_extra_keywords_through():
    FakeSleepStaging.calls.clear()

    baseline.run_yasa("raw-object", staging_factory=FakeSleepStaging, verbose=False)

    assert FakeSleepStaging.calls[0]["kwargs"] == {"verbose": False}


def test_stage_agreement_perfect_match():
    labels = np.repeat(np.array(STAGES), 10)

    result = baseline.stage_agreement(labels.copy(), labels)

    assert result["accuracy"] == pytest.approx(1.0)
    assert result["kappa"] == pytest.approx(1.0)
    assert result["adjusted_rand"] == pytest.approx(1.0)
    assert result["n_epochs"] == len(labels)
    assert result["confusion"].shape == (len(STAGES), len(STAGES))


def test_stage_agreement_matches_sklearn_kappa():
    rng = np.random.default_rng(0)
    labels = rng.choice(STAGES, size=300, p=[0.2, 0.2, 0.35, 0.15, 0.1])
    predicted = rng.choice(STAGES, size=300, p=[0.15, 0.2, 0.4, 0.15, 0.1])

    result = baseline.stage_agreement(predicted, labels)

    assert result["kappa"] == pytest.approx(cohen_kappa_score(labels, predicted))


def test_stage_agreement_drops_unscored_epochs():
    labels = np.array(["W", "N2", "Movement time", "N3"])
    predicted = np.array(["W", "N2", "N2", "ART"])

    result = baseline.stage_agreement(predicted, labels)

    assert result["n_epochs"] == 2
    assert result["accuracy"] == pytest.approx(1.0)


def test_stage_agreement_without_overlap_is_nan():
    labels = np.array(["Movement time", "Unscored"])
    predicted = np.array(["ART", "UNS"])

    result = baseline.stage_agreement(predicted, labels)

    assert np.isnan(result["accuracy"])
    assert np.isnan(result["kappa"])
    assert result["n_epochs"] == 0


def test_majority_agreement_uses_string_labels():
    labels = np.array(["W", "W", "N2", "N2"])
    predicted = np.array(["W", "W", "N2", "N3"])

    result = baseline.stage_agreement(predicted, labels)

    assert result["accuracy"] == pytest.approx(0.75)
