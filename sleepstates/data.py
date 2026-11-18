"""Sleep-EDF Expanded download, 30 s epoching and per-subject caching (SS-01)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.datasets.sleep_physionet.age import fetch_data

EPOCH_SECONDS = 30.0
EEG_CHANNEL = "EEG Fpz-Cz"
STAGES = ("W", "N1", "N2", "N3", "REM")
# Wake is kept for at most 30 min before the first and after the last sleep epoch.
MAX_WAKE_EPOCHS = int(30 * 60 / EPOCH_SECONDS)

# Sleep-EDF is scored with R&K rules; AASM merges stages 3 and 4 into N3.
STAGE_ALIASES = {
    "sleep stage w": "W",
    "sleep stage 1": "N1",
    "sleep stage 2": "N2",
    "sleep stage 3": "N3",
    "sleep stage 4": "N3",
    "sleep stage r": "REM",
    "w": "W",
    "1": "N1",
    "2": "N2",
    "3": "N3",
    "4": "N3",
    "r": "REM",
}

# SC-subjects.xls carries the age column used by the SS-10 age comparison.
SUBJECT_TABLE_URLS = (
    "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC-subjects.xls",
    "https://physionet.org/files/sleep-edfx/1.0.0/SC-subjects.xls",
)


def map_stage(label: object) -> str | None:
    """Map one hypnogram label to W/N1/N2/N3/REM, or None to drop the epoch."""
    if label is None:
        return None
    return STAGE_ALIASES.get(str(label).strip().lower())


def annotation_arrays(annotations: object) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (onset, duration, description) arrays from mne.Annotations or a frame."""
    if hasattr(annotations, "onset") and hasattr(annotations, "description"):
        onset = np.asarray(annotations.onset, dtype=float)
        duration = np.asarray(annotations.duration, dtype=float)
        description = [str(d) for d in annotations.description]
    else:
        frame = pd.DataFrame(annotations)
        onset = frame["onset"].to_numpy(dtype=float)
        duration = frame["duration"].to_numpy(dtype=float)
        description = [str(d) for d in frame["description"]]
    # Zero/unknown durations are treated as instantaneous annotations.
    duration = np.nan_to_num(duration, nan=0.0, posinf=0.0, neginf=0.0)
    return onset, duration, description


def epoch_labels(annotations: object, n_epochs: int, epoch_seconds: float = EPOCH_SECONDS):
    """Label each 30 s epoch by the hypnogram annotation covering its midpoint."""
    onset, duration, description = annotation_arrays(annotations)
    midpoints = (np.arange(n_epochs) + 0.5) * epoch_seconds
    labels = np.empty(n_epochs, dtype=object)
    labels[:] = None
    for start, length, desc in zip(onset, duration, description):
        inside = (midpoints >= start) & (midpoints < start + length)
        if inside.any():
            labels[inside] = map_stage(desc)
    return labels


def trim_wake(labels, max_wake_epochs: int = MAX_WAKE_EPOCHS) -> tuple[int, int]:
    """Return the (start, stop) epoch slice that keeps sleep plus bounded wake."""
    sleep_mask = np.array(
        [label in STAGES and label != "W" for label in labels], dtype=bool
    )
    if not sleep_mask.any():
        return 0, len(labels)
    indices = np.flatnonzero(sleep_mask)
    start = max(0, int(indices[0]) - max_wake_epochs)
    stop = min(len(labels), int(indices[-1]) + 1 + max_wake_epochs)
    return start, stop


def epoch_recording(
    raw,
    annotations: object,
    epoch_seconds: float = EPOCH_SECONDS,
    max_wake_epochs: int = MAX_WAKE_EPOCHS,
) -> dict:
    """Cut a raw recording into labelled 30 s epochs, dropping unscored epochs."""
    pick = [name for name in raw.ch_names if EEG_CHANNEL.lower() in name.lower()]
    if not pick:
        raise ValueError(f"{EEG_CHANNEL} not found in channels: {raw.ch_names}")
    raw = raw.copy().pick(pick, verbose="ERROR")

    sfreq = float(raw.info["sfreq"])
    samples_per_epoch = round(epoch_seconds * sfreq)
    n_epochs = raw.n_times // samples_per_epoch
    if n_epochs == 0:
        raise ValueError("recording is shorter than one epoch")

    labels = epoch_labels(annotations, n_epochs, epoch_seconds)
    start, stop = trim_wake(labels, max_wake_epochs=wake_limit(epoch_seconds, max_wake_epochs))

    signal = raw.get_data(start=start * samples_per_epoch, stop=stop * samples_per_epoch)
    epochs = signal[0].reshape(-1, samples_per_epoch).astype(np.float32)
    kept = np.array([label is not None for label in labels[start:stop]], dtype=bool)

    return {
        "sfreq": sfreq,
        "start_epoch": start,
        "epochs": epochs[kept],
        "labels": np.array([str(label) for label in labels[start:stop][kept]], dtype="<U4"),
    }


def wake_limit(epoch_seconds: float, max_wake_epochs: int) -> int:
    """Scale the wake allowance from 30 s epochs to a custom epoch length."""
    if epoch_seconds == EPOCH_SECONDS:
        return max_wake_epochs
    return round(max_wake_epochs * EPOCH_SECONDS / epoch_seconds)


def recording_paths(subject: int, night: int = 1, data_dir=None) -> list[Path]:
    """Fetch (or reuse) the PSG and hypnogram paths for one subject/night."""
    paths = fetch_data(
        subjects=[int(subject)],
        recording=[int(night)],
        path=None if data_dir is None else str(data_dir),
        on_missing="raise",
        verbose="ERROR",
    )
    psg_path, hyp_path = paths[0]
    return [Path(psg_path), Path(hyp_path)]


def build_subject(subject: int, night: int = 1, data_dir=None, age: float | None = None) -> dict:
    """Load one Sleep-EDF night and return its labelled epochs."""
    psg_path, hyp_path = recording_paths(subject, night, data_dir)
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
    annotations = mne.read_annotations(hyp_path)
    record = epoch_recording(raw, annotations)
    record.update(
        subject=int(subject),
        night=int(night),
        age=float("nan") if age is None else float(age),
    )
    return record


def save_subject(path, record: dict) -> Path:
    """Write one subject's epochs and labels to a compressed .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        subject=np.int64(record["subject"]),
        night=np.int64(record["night"]),
        age=np.float64(record.get("age", np.nan)),
        sfreq=np.float64(record["sfreq"]),
        start_epoch=np.int64(record.get("start_epoch", 0)),
        epochs=np.asarray(record["epochs"], dtype=np.float32),
        labels=np.asarray(record["labels"], dtype=str),
    )
    return path


def load_subject(path) -> dict:
    """Read a .npz written by save_subject back into a plain dict."""
    with np.load(Path(path), allow_pickle=False) as npz:
        record = {name: npz[name] for name in npz.files}
    return {
        "subject": int(record["subject"]),
        "night": int(record["night"]),
        "age": float(record["age"]),
        "sfreq": float(record["sfreq"]),
        "start_epoch": int(record["start_epoch"]),
        "epochs": np.asarray(record["epochs"], dtype=np.float32),
        "labels": np.array([str(label) for label in record["labels"]], dtype=object),
    }


def load_epoch_dir(directory) -> list[dict]:
    """Load every cached subject .npz in a directory, sorted by filename."""
    return [load_subject(path) for path in sorted(Path(directory).glob("*.npz"))]


def load_subject_table(path) -> pd.DataFrame:
    """Load the Sleep-EDF subject table (SC-subjects.xls or a csv copy)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no subject table at {path}")
    if path.suffix.lower() in {".csv", ".txt"}:
        table = pd.read_csv(path)
    else:
        table = pd.read_excel(path)
    table.columns = [str(column).strip().lower() for column in table.columns]
    # The xls export starts with an unnamed id column that duplicates `subject`.
    if "subject" not in table.columns and table.columns[0].startswith("unnamed"):
        table = table.rename(columns={table.columns[0]: "subject"})
    missing = {"subject", "night", "age"} - set(table.columns)
    if missing:
        raise ValueError(f"subject table is missing columns: {sorted(missing)}")
    table = table[["subject", "night", "age"]].copy()
    table["subject"] = table["subject"].astype(int)
    table["night"] = table["night"].astype(int)
    table["age"] = pd.to_numeric(table["age"])
    return table.reset_index(drop=True)


def subject_ages(table: pd.DataFrame) -> dict[tuple[int, int], float]:
    """Turn a subject table into a {(subject, night): age} mapping."""
    return {
        (int(row.subject), int(row.night)): float(row.age)
        for row in table.itertuples(index=False)
    }


def fetch_subject_table(data_dir, timeout: float = 60.0) -> Path:
    """Download SC-subjects.xls (age metadata) once; the download script calls this."""
    path = Path(data_dir) / "SC-subjects.xls"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for url in SUBJECT_TABLE_URLS:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                path.write_bytes(response.read())
            return path
        except OSError:
            continue
    raise RuntimeError(
        "could not download SC-subjects.xls; age-based analysis will be skipped"
    )
