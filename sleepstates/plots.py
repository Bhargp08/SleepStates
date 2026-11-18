"""Figures for model selection, transitions and hypnograms (SS-08, SS-10, SS-12)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

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
