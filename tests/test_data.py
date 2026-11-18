"""Tests for sleepstates.data (SS-01). Every recording here is synthetic."""

import numpy as np
import pandas as pd
import pytest
from mne import Annotations, create_info
from mne.io import RawArray

from sleepstates import data

SFREQ = 100.0


def _raw(seconds: float = 600.0, sfreq: float = SFREQ):
    """A one-channel noise recording, standing in for Fpz-Cz."""
    rng = np.random.default_rng(0)
    n_samples = int(seconds * sfreq)
    info = create_info([data.EEG_CHANNEL], sfreq, ch_types="eeg")
    signal = rng.standard_normal((1, n_samples)) * 1e-5
    return RawArray(signal, info, verbose="ERROR")


def _annotations(onsets, durations, descriptions):
    return Annotations(onsets, durations, descriptions, orig_time=None)


def test_map_stage_collapses_rk_to_aasm():
    assert data.map_stage("Sleep stage W") == "W"
    assert data.map_stage("Sleep stage 1") == "N1"
    assert data.map_stage("Sleep stage 2") == "N2"
    # R&K stages 3 and 4 both become N3.
    assert data.map_stage("Sleep stage 3") == "N3"
    assert data.map_stage("Sleep stage 4") == "N3"
    assert data.map_stage("Sleep stage R") == "REM"
    # Short forms show up in written annotation files.
    assert data.map_stage("W") == "W"
    assert data.map_stage("4") == "N3"


@pytest.mark.parametrize(
    "label",
    ["Movement time", "Sleep stage ?", "Unscored", "", None],
)
def test_map_stage_drops_non_sleep_labels(label):
    assert data.map_stage(label) is None


def test_epoch_labels_uses_epoch_midpoints():
    ann = _annotations([0.0, 30.0, 90.0], [30.0, 60.0, 30.0], ["Sleep stage W", "Sleep stage 2", "Sleep stage R"])
    labels = data.epoch_labels(ann, n_epochs=4)
    assert list(labels) == ["W", "N2", "N2", "REM"]


def test_epoch_labels_marks_movement_as_missing():
    ann = _annotations([0.0, 30.0], [30.0, 60.0], ["Sleep stage W", "Movement time"])
    labels = data.epoch_labels(ann, n_epochs=3)
    assert labels[0] == "W"
    assert labels[1] is None
    assert labels[2] is None


def test_trim_wake_keeps_only_thirty_minutes_at_each_end():
    labels = np.array(["W"] * 100 + ["N2"] * 40 + ["W"] * 60, dtype=object)
    start, stop = data.trim_wake(labels)
    assert start == 40  # 60 wake epochs kept before the first sleep epoch
    assert stop == 200  # trailing wake fits inside the 30 min allowance


def test_trim_wake_does_not_trim_short_wake():
    labels = np.array(["W"] * 10 + ["N2"] * 5 + ["W"] * 3, dtype=object)
    assert data.trim_wake(labels) == (0, 18)


def test_epoch_recording_shapes_and_labels():
    raw = _raw()
    ann = _annotations([0.0, 60.0, 300.0], [60.0, 240.0, 300.0], ["Sleep stage W", "Sleep stage 2", "Sleep stage R"])
    record = data.epoch_recording(raw, ann)

    assert record["sfreq"] == SFREQ
    assert record["epochs"].shape == (20, 3000)  # 20 epochs of 30 s at 100 Hz
    assert record["epochs"].dtype == np.float32
    assert list(record["labels"]) == ["W", "W"] + ["N2"] * 8 + ["REM"] * 10


def test_epoch_recording_drops_movement_and_unknown_epochs():
    raw = _raw(seconds=300.0)
    ann = _annotations([0.0, 30.0], [30.0, 270.0], ["Movement time", "Sleep stage 2"])
    record = data.epoch_recording(raw, ann)

    assert record["epochs"].shape == (9, 3000)
    assert set(record["labels"]) == {"N2"}


def test_save_and_load_subject_roundtrip(tmp_path):
    record = {
        "subject": 3,
        "night": 1,
        "age": 41.0,
        "sfreq": SFREQ,
        "start_epoch": 40,
        "epochs": np.zeros((2, 3000), dtype=np.float32),
        "labels": np.array(["W", "N2"], dtype=object),
    }
    path = tmp_path / "sub-03_night1.npz"
    data.save_subject(path, record)
    loaded = data.load_subject(path)

    assert loaded["subject"] == 3
    assert loaded["night"] == 1
    assert loaded["age"] == pytest.approx(41.0)
    assert loaded["sfreq"] == SFREQ
    assert loaded["start_epoch"] == 40
    assert loaded["epochs"].shape == (2, 3000)
    assert list(loaded["labels"]) == ["W", "N2"]


def test_load_subject_table_reads_csv(tmp_path):
    path = tmp_path / "SC-subjects.csv"
    pd.DataFrame(
        {
            "subject": [0, 0, 1],
            "night": [1, 2, 1],
            "age": [33, 33, 25],
            "sex": [1, 1, 2],
        }
    ).to_csv(path, index=False)

    table = data.load_subject_table(path)

    assert list(table.columns[:3]) == ["subject", "night", "age"]
    assert data.subject_ages(table)[(1, 1)] == pytest.approx(25.0)


def test_load_subject_table_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_subject_table(tmp_path / "nope.csv")


def test_recording_paths_delegates_to_mne(monkeypatch, tmp_path):
    calls = {}

    def fake_fetch_data(subjects, recording, path, **kwargs):
        calls["subjects"] = list(subjects)
        calls["recording"] = list(recording)
        return [(tmp_path / "sub-00_psg.edf", tmp_path / "sub-00_hyp.edf")]

    monkeypatch.setattr(data, "fetch_data", fake_fetch_data)

    paths = data.recording_paths(0, night=1, data_dir=tmp_path)

    assert calls["subjects"] == [0]
    assert calls["recording"] == [1]
    assert paths == [tmp_path / "sub-00_psg.edf", tmp_path / "sub-00_hyp.edf"]
