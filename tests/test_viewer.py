"""Tests for the hypnogram viewer (SS-12): figure structure and HTML output."""

import numpy as np

from sleepstates import plots

STAGES = ("W", "N1", "N2", "N3", "REM")


def _viewer_data(n_subjects: int = 3, n_epochs: int = 60):
    rng = np.random.default_rng(0)
    labels = rng.choice(STAGES, size=n_subjects * n_epochs)
    states = rng.integers(0, len(STAGES), size=n_subjects * n_epochs)
    subjects = np.repeat(np.arange(n_subjects), n_epochs)
    return states, labels, subjects


def test_figure_has_two_traces_per_subject():
    states, labels, subjects = _viewer_data()

    figure = plots.build_hypnogram_figure(states, labels, subjects)

    assert len(figure.data) == 2 * len(np.unique(subjects))
    assert "expert" in figure.data[0].name
    assert "machine" in figure.data[1].name


def test_dropdown_lists_every_subject_and_only_the_first_is_visible():
    states, labels, subjects = _viewer_data(n_subjects=4)

    figure = plots.build_hypnogram_figure(states, labels, subjects)

    buttons = figure.layout.updatemenus[0].buttons
    assert len(buttons) == 4
    assert "subject 00" in buttons[0].label

    visible = list(buttons[0].args[0]["visible"])
    assert visible == [True, True] + [False] * 6

    second = list(buttons[1].args[0]["visible"])
    assert second == [False, False, True, True] + [False] * 4


def test_machine_trace_uses_matched_stage_labels():
    # One expert stage per state, so the mapping is unambiguous.
    labels = np.repeat(np.array(STAGES), 5)
    states = np.repeat(np.arange(len(STAGES)), 5)
    subjects = np.zeros(len(labels), dtype=int)

    figure = plots.build_hypnogram_figure(states, labels, subjects)

    ranks = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
    expected = [ranks[stage] for stage in labels]
    assert list(figure.data[1].y) == expected
    assert list(figure.data[0].y) == expected


def test_write_hypnogram_html(tmp_path):
    states, labels, subjects = _viewer_data()

    figure = plots.build_hypnogram_figure(states, labels, subjects)
    path = tmp_path / "hypnogram.html"
    plots.write_hypnogram_html(figure, path)

    text = path.read_text()
    assert "<html" in text.lower()
    assert "plotly" in text.lower()
    assert "updatemenu" in text.lower()


def test_viewer_script_writes_html(tmp_path):
    from scripts.viewer import main

    states, labels, subjects = _viewer_data(n_subjects=2)
    states_path = tmp_path / "states.npz"
    np.savez_compressed(
        states_path,
        states=states,
        labels=labels,
        subjects=subjects,
        ages=np.full(len(labels), 40.0),
        lengths=np.full(2, len(labels) // 2),
    )

    exit_code = main(
        [
            "--states",
            str(states_path),
            "--out",
            str(tmp_path / "hypnogram.html"),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "hypnogram.html").exists()
