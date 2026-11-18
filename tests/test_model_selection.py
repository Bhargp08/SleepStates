"""Tests for model selection (SS-08): parameter counts, BIC, K sweep, plot."""

import numpy as np
import pytest

from sleepstates import analysis, plots

STAGES = 3


def _synthetic_epochs(n_subjects: int = 4, n_epochs: int = 60, n_features: int = 3, seed: int = 0):
    """Feature matrix with a subject-specific offset, plus per-subject lengths."""
    rng = np.random.default_rng(seed)
    chunks = [rng.normal(loc=subject, size=(n_epochs, n_features)) for subject in range(n_subjects)]
    return np.vstack(chunks), np.array([n_epochs] * n_subjects)


def test_hmm_parameter_count_counts_transitions_and_covariances():
    # start probs (K-1), transitions (K*(K-1)), means (K*D), full covariances (K*D(D+1)/2).
    assert analysis.hmm_n_parameters(2, 2) == 1 + 2 + 4 + 6
    assert analysis.hmm_n_parameters(3, 5) == 2 + 6 + 15 + 45


def test_bic_hmm_applies_the_parameter_penalty():
    n_samples = 500
    expected = -2 * (-400.0) + analysis.hmm_n_parameters(3, 5) * np.log(n_samples)

    assert analysis.bic_hmm(-400.0, 3, 5, n_samples) == pytest.approx(expected)


def test_sweep_k_reports_one_row_per_k():
    features, lengths = _synthetic_epochs()

    table = analysis.sweep_k(
        features,
        lengths,
        k_values=(2, 3),
        n_folds=2,
        restarts=1,
        max_iter=5,
        pca_components=2,
        seed=0,
    )

    assert list(table["k"]) == [2, 3]
    for column in (
        "k",
        "n_parameters",
        "train_loglik",
        "bic",
        "heldout_loglik_per_epoch",
        "heldout_std",
    ):
        assert column in table.columns
    assert np.isfinite(table["train_loglik"]).all()
    assert np.isfinite(table["heldout_loglik_per_epoch"]).all()
    # More states can only mean more free parameters.
    assert table["n_parameters"].is_monotonic_increasing


def test_sweep_k_heldout_scores_differ_from_training_scores():
    features, lengths = _synthetic_epochs(n_subjects=5, n_epochs=40)

    table = analysis.sweep_k(
        features,
        lengths,
        k_values=(2,),
        n_folds=2,
        restarts=1,
        max_iter=3,
        pca_components=2,
        seed=1,
    )

    row = table.iloc[0]
    # Held-out likelihood is per epoch and must not simply repeat the training figure.
    assert row["heldout_loglik_per_epoch"] != pytest.approx(row["train_loglik"] / (5 * 40))


def test_best_k_selection_helpers():
    table = analysis.sweep_k(
        *_synthetic_epochs(n_subjects=4, n_epochs=30),
        k_values=(2, 3),
        n_folds=2,
        restarts=1,
        max_iter=3,
        pca_components=2,
        seed=2,
    )

    assert analysis.best_k_by_bic(table) in (2, 3)
    assert analysis.best_k_by_heldout(table) in (2, 3)


def test_plot_model_selection_writes_a_png(tmp_path):
    table = analysis.sweep_k(
        *_synthetic_epochs(n_subjects=3, n_epochs=30),
        k_values=(2, 3),
        n_folds=2,
        restarts=1,
        max_iter=3,
        pca_components=2,
        seed=3,
    )
    path = tmp_path / "model_selection.png"

    plots.plot_model_selection(table, path)

    assert path.exists()
    assert path.stat().st_size > 1000
