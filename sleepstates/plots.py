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
