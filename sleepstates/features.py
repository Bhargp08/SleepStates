"""Per-epoch spectral features from multitaper PSDs (SS-02)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mne.time_frequency import psd_array_multitaper

FMIN, FMAX = 0.5, 30.0

# Canonical sleep EEG bands (AASM), in Hz.
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 15.0),
    "beta": (15.0, 30.0),
}
BAND_NAMES = tuple(BANDS)
FEATURE_NAMES = BAND_NAMES + ("delta_theta", "theta_alpha", "sigma_total")


def multitaper_psd(epochs, sfreq: float, fmin: float = FMIN, fmax: float = FMAX):
    """Multitaper PSD for a (n_epochs, n_samples) array; returns (psds, freqs)."""
    psds, freqs = psd_array_multitaper(
        np.asarray(epochs, dtype=float),
        float(sfreq),
        fmin=fmin,
        fmax=fmax,
        normalization="full",
        verbose="ERROR",
    )
    return psds, freqs


def band_powers(epochs, sfreq: float) -> np.ndarray:
    """Absolute power in each band, integrated over frequency: (n_epochs, 5)."""
    psds, freqs = multitaper_psd(epochs, sfreq)
    powers = np.empty((psds.shape[0], len(BANDS)), dtype=float)
    for column, (low, high) in enumerate(BANDS.values()):
        band = (freqs >= low) & (freqs < high)
        powers[:, column] = np.trapezoid(psds[:, band], freqs[band], axis=1)
    return powers


def spectral_features(epochs, sfreq: float) -> np.ndarray:
    """Log band powers plus three log ratios: (n_epochs, 8)."""
    powers = band_powers(epochs, sfreq)
    delta, theta, sigma = powers[:, 0], powers[:, 1], powers[:, 3]
    total = np.maximum(powers.sum(axis=1), 1e-12)
    ratios = np.column_stack(
        [
            delta / np.maximum(theta, 1e-12),
            theta / np.maximum(powers[:, 2], 1e-12),
            sigma / total,
        ]
    )
    # Log features are far closer to Gaussian, which the GMM/HMM assume.
    return np.column_stack([np.log10(np.maximum(powers, 1e-12)), np.log10(ratios)])


def zscore_per_subject(data, subjects=None) -> np.ndarray:
    """Standardise each feature column within each subject (or the whole array)."""
    values = np.array(data, dtype=float, copy=True)
    if subjects is None:
        groups = [np.ones(len(values), dtype=bool)]
    else:
        subject_ids = np.asarray(subjects)
        groups = [subject_ids == subject for subject in np.unique(subject_ids)]
    for mask in groups:
        block = values[mask]
        spread = block.std(axis=0)
        spread = np.where(spread < 1e-12, 1.0, spread)
        values[mask] = (block - block.mean(axis=0)) / spread
    return values


def build_feature_dataset(records: list[dict]) -> dict:
    """Feature matrix for a list of epoch records, standardised per subject."""
    chunks, labels, subjects, ages, lengths = [], [], [], [], []
    for record in records:
        feats = spectral_features(record["epochs"], record["sfreq"])
        n_epochs = len(feats)
        chunks.append(feats)
        labels.extend(str(label) for label in record["labels"])
        subjects.extend([int(record["subject"])] * n_epochs)
        ages.extend([float(record.get("age", np.nan))] * n_epochs)
        lengths.append(n_epochs)

    matrix = zscore_per_subject(np.vstack(chunks), np.asarray(subjects))
    return {
        "X": matrix.astype(np.float32),
        "labels": np.asarray(labels),
        "subjects": np.asarray(subjects, dtype=int),
        "ages": np.asarray(ages, dtype=float),
        "lengths": np.asarray(lengths, dtype=int),
        "feature_names": np.asarray(FEATURE_NAMES),
    }


def save_feature_dataset(path, dataset: dict) -> Path:
    """Persist a feature dataset as a single compressed .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=np.asarray(dataset["X"], dtype=np.float32),
        labels=np.asarray(dataset["labels"], dtype=str),
        subjects=np.asarray(dataset["subjects"], dtype=int),
        ages=np.asarray(dataset["ages"], dtype=float),
        lengths=np.asarray(dataset["lengths"], dtype=int),
        feature_names=np.asarray(dataset["feature_names"], dtype=str),
    )
    return path


def load_feature_dataset(path) -> dict:
    """Reload a feature dataset written by save_feature_dataset."""
    with np.load(Path(path), allow_pickle=False) as npz:
        return {
            "X": np.asarray(npz["X"], dtype=np.float32),
            "labels": np.array([str(label) for label in npz["labels"]], dtype=object),
            "subjects": np.asarray(npz["subjects"], dtype=int),
            "ages": np.asarray(npz["ages"], dtype=float),
            "lengths": np.asarray(npz["lengths"], dtype=int),
            "feature_names": np.asarray([str(name) for name in npz["feature_names"]]),
        }
