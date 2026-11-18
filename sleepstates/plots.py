"""Figures for model selection, transitions and hypnograms (SS-08, SS-10, SS-12)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: CI writes files, never windows

import matplotlib.pyplot as plt


def _save(figure, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
    return figure


def plot_model_selection(table, path=None):
    """BIC and held-out likelihood as a function of the number of states."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(table["k"], table["bic"], marker="o")
    axes[0].set_xlabel("states (K)")
    axes[0].set_ylabel("BIC")
    axes[0].set_title("BIC (lower is better)")

    axes[1].errorbar(
        table["k"],
        table["heldout_loglik_per_epoch"],
        yerr=table["heldout_std"],
        marker="o",
        capsize=3,
    )
    axes[1].set_xlabel("states (K)")
    axes[1].set_ylabel("held-out log-likelihood / epoch")
    axes[1].set_title("Leave-subjects-out likelihood (higher is better)")

    figure.tight_layout()
    return _save(figure, path)


def plot_confusion_matrix(confusion, stages=("W", "N1", "N2", "N3", "REM"), path=None):
    """Machine states (rows) against expert stages (columns), row-normalised."""
    matrix = np.asarray(confusion, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    figure, axis = plt.subplots(figsize=(6, 0.6 * len(matrix) + 2))
    image = axis.imshow(normalised, cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(stages)), stages)
    axis.set_yticks(range(len(matrix)), [f"state {index}" for index in range(len(matrix))])
    axis.set_xlabel("expert stage")
    axis.set_ylabel("machine state")
    axis.set_title("State-to-stage confusion (row-normalised)")
    figure.colorbar(image, ax=axis, label="fraction of state")

    for row in range(len(matrix)):
        for column in range(len(stages)):
            axis.text(
                column,
                row,
                f"{normalised[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if normalised[row, column] < 0.6 else "black",
            )

    figure.tight_layout()
    return _save(figure, path)


def plot_transition_matrix(transmat, state_labels=None, path=None):
    """Heatmap of the state-to-state transition matrix."""
    matrix = np.asarray(transmat, dtype=float)
    labels = state_labels or [f"state {index}" for index in range(len(matrix))]

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="magma", vmin=0.0, vmax=max(matrix.max(), 1e-9))
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("to")
    axis.set_ylabel("from")
    axis.set_title("State transition probabilities")
    figure.colorbar(image, ax=axis, label="p(transition)")

    peak = max(matrix.max(), 1e-9)
    for row in range(len(matrix)):
        for column in range(len(matrix)):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] < 0.5 * peak else "black",
            )

    figure.tight_layout()
    return _save(figure, path)


def plot_dwell_times(table, path=None):
    """Expected versus empirical dwell time for each state."""
    positions = np.arange(len(table))
    width = 0.4

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(positions - width / 2, table["expected"], width, label="expected 1/(1-a_ii)")
    axis.bar(positions + width / 2, table["empirical"], width, label="empirical (Viterbi)")
    axis.set_xticks(positions, [f"state {state}" for state in table["state"]])
    axis.set_ylabel("epochs per run")
    axis.set_title("Dwell times: model versus data")
    axis.legend()

    figure.tight_layout()
    return _save(figure, path)


def build_hypnogram_figure(states, labels, subjects, stages=("W", "N1", "N2", "N3", "REM"), mapping=None):
    """Interactive Plotly figure: expert and machine hypnograms, one dropdown entry per subject."""
    from plotly.graph_objects import Scatter
    from plotly.subplots import make_subplots

    from sleepstates.evaluate import apply_mapping, confusion_counts, majority_mapping

    states = np.asarray(states, dtype=int)
    labels = np.asarray(labels, dtype=str)
    subjects = np.asarray(subjects)

    if mapping is None:
        mapping = majority_mapping(confusion_counts(states, labels, stages), stages)
    machine = apply_mapping(states, mapping)
    ranks = {stage: index for index, stage in enumerate(stages)}

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("expert (AASM)", "machine (HMM Viterbi)"),
    )

    unique_subjects = np.unique(subjects)
    total_traces = 2 * len(unique_subjects)
    buttons = []

    for index, subject in enumerate(unique_subjects):
        mask = subjects == subject
        hours = np.arange(int(mask.sum())) * 30.0 / 3600.0
        expert_ranks = [ranks.get(stage, np.nan) for stage in labels[mask]]
        machine_ranks = [ranks.get(stage, np.nan) for stage in machine[mask]]
        show = index == 0

        figure.add_trace(
            Scatter(
                x=hours,
                y=expert_ranks,
                mode="lines",
                name=f"expert · subject {int(subject):02d}",
                visible=show,
                line={"color": "#1f77b4", "width": 1.5},
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            Scatter(
                x=hours,
                y=machine_ranks,
                mode="lines",
                name=f"machine · subject {int(subject):02d}",
                visible=show,
                line={"color": "#d62728", "width": 1.5},
            ),
            row=2,
            col=1,
        )

        visibility = [False] * total_traces
        visibility[2 * index] = True
        visibility[2 * index + 1] = True
        buttons.append(
            {
                "label": f"subject {int(subject):02d}",
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {"title.text": f"subject {int(subject):02d}"},
                ],
            }
        )

    figure.update_layout(
        title={"text": f"subject {int(unique_subjects[0]):02d}" if len(unique_subjects) else ""},
        height=620,
        template="plotly_white",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 1.02,
                "y": 1.15,
                "showactive": True,
            }
        ],
    )
    for row in (1, 2):
        figure.update_yaxes(
            tickvals=list(range(len(stages))),
            ticktext=list(stages),
            range=[len(stages) - 0.5, -0.5],
            title_text="stage",
            row=row,
            col=1,
        )
    figure.update_xaxes(title_text="hours from recording start", row=2, col=1)
    return figure


def write_hypnogram_html(figure, path) -> Path:
    """Write the interactive figure to a self-contained HTML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(path), include_plotlyjs=True)
    return path


def plot_age_comparison(result, path=None):
    """Per-subject transition rate and mean dwell, split by age group."""
    frame = result["subjects"]

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, column, label in (
        (axes[0], "transition_rate", "transitions / epoch"),
        (axes[1], "mean_dwell", "mean dwell (epochs)"),
    ):
        groups = [
            frame.loc[frame["group"] == group, column].to_numpy(dtype=float)
            for group in ("younger", "older")
        ]
        axis.boxplot(groups, widths=0.5)
        axis.set_xticks([1, 2], ["younger", "older"])
        axis.set_ylabel(label)
        axis.set_title(f"{label} by age group")

    figure.tight_layout()
    return _save(figure, path)
