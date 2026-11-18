"""Tests for sleepstates.evaluate (SS-09): matching, kappa, ARI, per-stage scores."""

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from sleepstates import evaluate, plots

STAGES = evaluate.STAGES


def test_confusion_counts_shape_and_total():
    labels = np.array(["W", "W", "N2", "REM"])
    states = np.array([0, 0, 1, 2])

    counts = evaluate.confusion_counts(states, labels)

    assert counts.shape == (3, len(STAGES))
    assert counts.sum() == len(labels)
    assert counts[0, STAGES.index("W")] == 2
    assert counts[2, STAGES.index("REM")] == 1


def test_hungarian_mapping_recovers_a_permutation():
    # Expert stages in the order N2, REM, W, N1, N3 map to states 0..4 shuffled.
    labels = np.repeat(np.array(["W", "N1", "N2", "N3", "REM"]), 4)
    states = np.repeat(np.array([2, 3, 0, 4, 1]), 4)

    mapping = evaluate.hungarian_mapping(evaluate.confusion_counts(states, labels))

    assert mapping == {0: "N2", 1: "REM", 2: "W", 3: "N1", 4: "N3"}


def test_majority_mapping_allows_many_to_one():
    # Two machine states both spend most of their time in N2.
    labels = np.array(["N2"] * 8 + ["N2"] * 3 + ["W"] * 5)
    states = np.array([0] * 6 + [1] * 2 + [1] * 3 + [2] * 5)

    mapping = evaluate.majority_mapping(evaluate.confusion_counts(states, labels))

    assert mapping[0] == "N2"
    assert mapping[1] == "N2"
    assert mapping[2] == "W"


def test_evaluate_perfect_agreement():
    labels = np.repeat(np.array(STAGES), 10)
    states = np.repeat(np.arange(len(STAGES)), 10)

    result = evaluate.evaluate(states, labels)

    assert result["accuracy"] == pytest.approx(1.0)
    assert result["kappa"] == pytest.approx(1.0)
    assert result["adjusted_rand"] == pytest.approx(1.0)
    assert np.allclose(result["per_stage"]["recall"], 1.0)


def test_evaluate_kappa_matches_sklearn():
    rng = np.random.default_rng(0)
    labels = rng.choice(STAGES, size=200, p=[0.2, 0.2, 0.4, 0.1, 0.1])
    states = rng.integers(0, 5, size=200)

    result = evaluate.evaluate(states, labels)
    predicted = evaluate.apply_mapping(states, result["mapping"])

    assert result["kappa"] == pytest.approx(cohen_kappa_score(labels, predicted))
    assert 0.0 <= result["accuracy"] <= 1.0


def test_per_stage_recall_and_precision():
    labels = np.array(["W", "W", "N2", "N2", "N2", "N3"])
    states = np.array([0, 0, 1, 1, 1, 1])

    result = evaluate.evaluate(states, labels)
    per_stage = result["per_stage"].set_index("stage")

    # State 1 holds three N2 epochs and one N3 epoch, so it becomes N2.
    assert result["mapping"][1] == "N2"
    assert per_stage.loc["N2", "recall"] == pytest.approx(1.0)
    assert per_stage.loc["N3", "recall"] == pytest.approx(0.0)
    assert per_stage.loc["W", "support"] == 2


def test_evaluate_with_fewer_states_than_expert_stages():
    labels = np.array(["W"] * 10 + ["N2"] * 10 + ["REM"] * 10)
    states = np.repeat([0, 1, 2], 10)

    result = evaluate.evaluate(states, labels)

    assert len(result["mapping"]) == 3
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["confusion"].shape == (3, len(STAGES))


def test_many_to_one_mode_still_scores_accuracy():
    labels = np.repeat(np.array(STAGES), 5)
    states = np.repeat(np.arange(len(STAGES)), 5)
    states[0] = 5  # an extra state that also lands in W

    result = evaluate.evaluate(states, labels, method="majority")

    assert result["mapping"][5] == "W"
    assert result["accuracy"] > 0.9


def test_plot_confusion_matrix_writes_a_png(tmp_path):
    labels = np.repeat(np.array(STAGES), 4)
    states = np.repeat(np.arange(len(STAGES)), 4)
    result = evaluate.evaluate(states, labels)
    path = tmp_path / "confusion.png"

    plots.plot_confusion_matrix(result["confusion"], path=path)

    assert path.exists() and path.stat().st_size > 1000
