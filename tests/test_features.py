"""Tests for sleepstates.features (SS-02) on synthetic tones and noise."""

import numpy as np
import pytest

from sleepstates import features

SFREQ = 100.0
EPOCH_SECONDS = 30.0


def _epochs(freq: float, n_epochs: int = 4, amplitude: float = 20e-6, seed: int = 0) -> np.ndarray:
    """Pure sinusoids plus a little noise, shaped like real epoched EEG."""
    rng = np.random.default_rng(seed)
    n_samples = int(EPOCH_SECONDS * SFREQ)
    t = np.arange(n_samples) / SFREQ
    tone = amplitude * np.sin(2 * np.pi * freq * t)
    return (tone[None, :] + rng.standard_normal((n_epochs, n_samples)) * amplitude / 20).astype(
        np.float64
    )


def test_feature_names_are_stable():
    assert features.FEATURE_NAMES == (
        "delta",
        "theta",
        "alpha",
        "sigma",
        "beta",
        "delta_theta",
        "theta_alpha",
        "sigma_total",
    )


def test_band_powers_peak_in_the_expected_band():
    powers = features.band_powers(_epochs(2.0), SFREQ)
    assert powers.shape == (4, 5)
    assert int(np.argmax(powers.mean(axis=0))) == 0  # delta

    powers = features.band_powers(_epochs(10.0), SFREQ)
    assert int(np.argmax(powers.mean(axis=0))) == 2  # alpha


def test_band_powers_scale_with_squared_amplitude():
    quiet = features.band_powers(_epochs(6.0, amplitude=10e-6), SFREQ).mean(axis=0)
    loud = features.band_powers(_epochs(6.0, amplitude=20e-6), SFREQ).mean(axis=0)
    ratio = loud / quiet
    # Power is quadratic in amplitude, and the noise term is small but nonzero.
    assert np.all((ratio > 3.0) & (ratio < 5.0))


def test_spectral_features_shape_and_ratios():
    epochs = _epochs(10.0)
    feats = features.spectral_features(epochs, SFREQ)
    assert feats.shape == (len(epochs), len(features.FEATURE_NAMES))
    assert np.isfinite(feats).all()

    powers = features.band_powers(epochs, SFREQ)
    total = powers.sum(axis=1)
    expected_sigma_total = np.log10(powers[:, 3] / total)
    np.testing.assert_allclose(feats[:, -1], expected_sigma_total, rtol=1e-6)
    # Alpha dominates, so its log power exceeds delta's for a 10 Hz tone.
    assert feats[:, 2].mean() > feats[:, 0].mean()


def test_zscore_per_subject_centres_each_subject():
    rng = np.random.default_rng(1)
    subjects = np.array([0] * 20 + [1] * 30)
    data = np.vstack(
        [rng.normal(0.0, 2.0, (20, 3)) + 5.0, rng.normal(0.0, 0.5, (30, 3)) - 3.0]
    )

    scaled = features.zscore_per_subject(data, subjects)

    for subject in (0, 1):
        block = scaled[subjects == subject]
        np.testing.assert_allclose(block.mean(axis=0), 0.0, atol=1e-9)
        np.testing.assert_allclose(block.std(axis=0), 1.0, atol=1e-9)
    # Different offsets disappear once each subject is standardised.
    np.testing.assert_allclose(scaled[subjects == 0].mean(axis=0), scaled[subjects == 1].mean(0))


def test_zscore_per_subject_tolerates_constant_columns():
    subjects = np.array([0, 0, 0, 1, 1, 1])
    data = np.column_stack([np.ones(6), np.arange(6, dtype=float)])

    scaled = features.zscore_per_subject(data, subjects)

    assert np.isfinite(scaled).all()
    np.testing.assert_allclose(scaled[:, 0], 0.0)


def test_zscore_without_subject_labels_uses_one_group():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaled = features.zscore_per_subject(data)
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0)


def test_build_feature_dataset_stacks_and_standardises():
    records = [
        {"subject": 0, "age": 30.0, "sfreq": SFREQ, "epochs": _epochs(2.0, n_epochs=5, seed=2),
         "labels": np.array(["N2"] * 5, dtype=object)},
        {"subject": 1, "age": 55.0, "sfreq": SFREQ, "epochs": _epochs(10.0, n_epochs=3, seed=3),
         "labels": np.array(["W"] * 3, dtype=object)},
    ]

    dataset = features.build_feature_dataset(records)

    assert dataset["X"].shape == (8, len(features.FEATURE_NAMES))
    assert dataset["X"].dtype == np.float32
    assert list(dataset["lengths"]) == [5, 3]
    assert list(dataset["subjects"]) == [0] * 5 + [1] * 3
    assert list(dataset["ages"]) == pytest.approx([30.0] * 5 + [55.0] * 3)
    assert list(dataset["labels"]) == ["N2"] * 5 + ["W"] * 3
    # Standardisation is per subject, so each subject block has zero mean.
    for subject in (0, 1):
        block = dataset["X"][dataset["subjects"] == subject]
        np.testing.assert_allclose(block.mean(axis=0), 0.0, atol=1e-6)


def test_save_and_load_feature_dataset(tmp_path):
    records = [
        {"subject": 0, "age": 30.0, "sfreq": SFREQ, "epochs": _epochs(2.0, n_epochs=3),
         "labels": np.array(["N2"] * 3, dtype=object)},
    ]
    dataset = features.build_feature_dataset(records)
    path = tmp_path / "features.npz"

    features.save_feature_dataset(path, dataset)
    loaded = features.load_feature_dataset(path)

    np.testing.assert_allclose(loaded["X"], dataset["X"])
    assert list(loaded["labels"]) == ["N2"] * 3
    assert list(loaded["lengths"]) == [3]
