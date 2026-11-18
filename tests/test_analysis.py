"""Tests for sleepstates.analysis (SS-10): transitions, dwell times, age groups."""

import numpy as np
import pytest

from sleepstates import analysis, plots


def test_transition_counts_and_matrix():
    states = np.array([0, 0, 1, 1, 1, 0])

    counts = analysis.transition_counts(states)

    assert counts.shape == (2, 2)
    assert counts.sum() == len(states) - 1
    np.testing.assert_array_equal(counts, [[1, 1], [1, 2]])

    matrix = analysis.transition_matrix_from_states(states)
    np.testing.assert_allclose(matrix, [[0.5, 0.5], [1 / 3, 2 / 3]], atol=1e-12)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)


def test_stationary_distribution_is_the_left_eigenvector():
    transmat = np.array([[0.9, 0.1], [0.2, 0.8]])

    stationary = analysis.stationary_distribution(transmat)

    np.testing.assert_allclose(stationary, [2 / 3, 1 / 3], atol=1e-10)
    np.testing.assert_allclose(stationary @ transmat, stationary, atol=1e-10)
    assert stationary.sum() == pytest.approx(1.0)


def test_expected_dwell_is_the_geometric_mean():
    transmat = np.array([[0.9, 0.05, 0.05], [0.125, 0.75, 0.125], [0.25, 0.25, 0.5]])

    assert list(analysis.expected_dwell(transmat)) == pytest.approx([10.0, 4.0, 2.0], rel=1e-9)


def test_empirical_dwell_measures_run_lengths():
    states = np.array([0, 0, 0, 1, 1, 2, 0, 0])

    dwell = analysis.empirical_dwell(states)

    np.testing.assert_allclose(dwell, [2.5, 2.0, 1.0])


def test_dwell_table_compares_model_with_data():
    transmat = np.array([[0.8, 0.2], [0.5, 0.5]])
    states = np.array([0, 0, 0, 0, 1, 1])

    table = analysis.dwell_table(transmat, states)

    assert list(table.columns) == ["state", "expected", "empirical"]
    assert list(table["expected"]) == pytest.approx([5.0, 2.0])
    assert list(table["empirical"]) == pytest.approx([4.0, 2.0])


def test_subject_metrics_counts_transitions_per_epoch():
    subjects = np.array([0] * 4 + [1] * 4)
    states = np.array([0, 0, 0, 0, 0, 1, 0, 1])

    frame = analysis.subject_metrics(states, subjects)

    assert list(frame["subject"]) == [0, 1]
    assert list(frame["n_epochs"]) == [4, 4]
    assert list(frame["transitions"]) == [0, 3]
    assert list(frame["transition_rate"]) == pytest.approx([0.0, 1.0])
    assert list(frame["mean_dwell"]) == pytest.approx([4.0, 1.0])


def test_age_comparison_splits_subjects_at_the_median_age():
    # Three young subjects with sticky sequences, three older ones alternating.
    subjects = np.repeat(np.arange(6), 20)
    sticky = np.repeat([0], 20)
    alternating = np.tile([0, 1], 10)
    states = np.concatenate([sticky, sticky, sticky, alternating, alternating, alternating])
    ages = np.repeat([20.0, 22.0, 24.0, 60.0, 62.0, 64.0], 20)

    result = analysis.age_comparison(states, subjects, ages)

    summary = result["summary"].set_index("group")
    assert set(summary.index) == {"younger", "older"}
    assert summary.loc["younger", "n"] == 3
    assert summary.loc["older", "n"] == 3
    assert summary.loc["older", "transition_rate"] > summary.loc["younger", "transition_rate"]
    assert 0.0 <= result["p_transition_rate"] <= 1.0
    assert 0.0 <= result["p_mean_dwell"] <= 1.0


def test_age_comparison_drops_missing_ages():
    subjects = np.array([0] * 5 + [1] * 5)
    states = np.array([0] * 5 + [0, 0, 0, 0, 0])
    ages = np.array([30.0] * 5 + [np.nan] * 5)

    result = analysis.age_comparison(states, subjects, ages)

    assert len(result["subjects"]) == 1


def test_plots_write_pngs(tmp_path):
    transmat = np.array([[0.8, 0.2], [0.3, 0.7]])
    table = analysis.dwell_table(transmat, np.array([0, 0, 1, 1, 1]))
    subjects = np.repeat(np.arange(4), 10)
    states = np.concatenate([np.repeat(0, 10), np.repeat(0, 10), np.tile([0, 1], 5), np.tile([1, 0], 5)])
    ages = np.repeat([20.0, 21.0, 60.0, 65.0], 10)
    age_result = analysis.age_comparison(states, subjects, ages)

    paths = [
        tmp_path / "transitions.png",
        tmp_path / "dwell.png",
        tmp_path / "age.png",
    ]
    plots.plot_transition_matrix(transmat, path=paths[0])
    plots.plot_dwell_times(table, path=paths[1])
    plots.plot_age_comparison(age_result, path=paths[2])

    for path in paths:
        assert path.exists() and path.stat().st_size > 1000
