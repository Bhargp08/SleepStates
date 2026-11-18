"""Matching HMM states to expert stages and scoring agreement (SS-09)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score

STAGES = ("W", "N1", "N2", "N3", "REM")


def confusion_counts(states, labels, stages=STAGES) -> np.ndarray:
    """Counts with machine states as rows and expert stages as columns."""
    states = np.asarray(states, dtype=int)
    labels = np.asarray(labels, dtype=str)
    n_states = int(states.max()) + 1 if states.size else 0
    table = np.zeros((n_states, len(stages)), dtype=int)
    for state, label in zip(states, labels, strict=True):
        column = stages.index(label) if label in stages else None
        if column is not None:
            table[state, column] += 1
    return table


def majority_mapping(table: np.ndarray, stages=STAGES) -> dict[int, str]:
    """Map every state to its most frequent expert stage (many-to-one allowed)."""
    mapping = {}
    for state, row in enumerate(table):
        if row.sum() > 0:
            mapping[int(state)] = stages[int(row.argmax())]
    return mapping


def hungarian_mapping(table: np.ndarray, stages=STAGES) -> dict[int, str]:
    """One-to-one assignment maximising agreement; leftovers use majority vote."""
    if table.size == 0:
        return {}
    rows, columns = linear_sum_assignment(-table.astype(float))
    mapping = {int(row): stages[int(column)] for row, column in zip(rows, columns, strict=True)}

    # With K > 5 some states cannot get a unique stage, so fall back per state.
    fallback = majority_mapping(table, stages)
    for state in range(table.shape[0]):
        mapping.setdefault(state, fallback.get(state, stages[0]))
    return mapping


def apply_mapping(states, mapping: dict[int, str]) -> np.ndarray:
    """Translate state ids into expert stage names."""
    return np.array([mapping[int(state)] for state in np.asarray(states, dtype=int)], dtype=object)


def per_stage_scores(labels, predicted, stages=STAGES) -> pd.DataFrame:
    """Recall and precision for each expert stage, with epoch counts."""
    rows = []
    for stage in stages:
        true_mask = labels == stage
        pred_mask = predicted == stage
        rows.append(
            {
                "stage": stage,
                "support": int(true_mask.sum()),
                "recall": float((predicted[true_mask] == stage).mean())
                if true_mask.any()
                else float("nan"),
                "precision": float((labels[pred_mask] == stage).mean())
                if pred_mask.any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def evaluate(states, labels, stages=STAGES, method: str = "hungarian") -> dict:
    """Score a machine hypnogram against expert labels after state matching."""
    if method not in {"hungarian", "majority"}:
        raise ValueError(f"unknown matching method: {method}")
    states = np.asarray(states, dtype=int)
    labels = np.asarray(labels, dtype=str)

    known = np.isin(labels, stages)
    states, labels = states[known], labels[known]
    table = confusion_counts(states, labels, stages)
    mapping = (
        hungarian_mapping(table, stages)
        if method == "hungarian"
        else majority_mapping(table, stages)
    )
    predicted = apply_mapping(states, mapping)

    return {
        "mapping": mapping,
        "accuracy": float((predicted == labels).mean()) if len(labels) else float("nan"),
        "kappa": float(cohen_kappa_score(labels, predicted)) if len(labels) else float("nan"),
        "adjusted_rand": float(adjusted_rand_score(labels, states)) if len(labels) else float("nan"),
        "confusion": table,
        "per_stage": per_stage_scores(labels, predicted, stages),
        "n_epochs": len(labels),
        "method": method,
    }
