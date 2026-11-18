"""Model selection (SS-08) and stage-transition analysis (SS-10)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from sleepstates import gmm, hmm


def split_sequences(lengths, matrix: np.ndarray) -> list[np.ndarray]:
    """Split a stacked (T, D) matrix into one sequence per subject."""
    boundaries = np.cumsum(np.asarray(lengths, dtype=int))[:-1]
    return [chunk for chunk in np.split(np.asarray(matrix), boundaries) if len(chunk)]


def subject_blocks(lengths, matrix: np.ndarray) -> list[np.ndarray]:
    """Split a stacked matrix into per-subject blocks (empty blocks dropped)."""
    boundaries = np.concatenate([[0], np.cumsum(np.asarray(lengths, dtype=int))])
    blocks = [
        np.asarray(matrix)[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)
    ]
    return [block for block in blocks if len(block)]


def hmm_n_parameters(n_components: int, n_features: int) -> int:
    """Free HMM parameters: start probs, transitions, means, full covariances."""
    return (
        (n_components - 1)
        + n_components * (n_components - 1)
        + n_components * n_features
        + n_components * n_features * (n_features + 1) // 2
    )


def bic_hmm(log_likelihood: float, n_components: int, n_features: int, n_samples: int) -> float:
    """Bayesian information criterion for a trained HMM; lower is better."""
    return -2.0 * log_likelihood + hmm_n_parameters(n_components, n_features) * np.log(n_samples)


def fit_pipeline(
    features: np.ndarray,
    lengths,
    n_states: int,
    pca_components: int = 5,
    restarts: int = 3,
    max_iter: int = 100,
    seed: int = 0,
):
    """PCA -> GMM initialisation -> Baum-Welch. Returns (model, projection, pca)."""
    features = np.asarray(features, dtype=float)
    n_components = min(pca_components, features.shape[1])
    projection, pca = gmm.fit_pca(features, n_components=n_components, random_state=seed)
    mixture = gmm.GMM(n_components=n_states, n_init=restarts, random_state=seed).fit(projection)
    sequences = split_sequences(lengths, projection)
    model = hmm.GaussianHMM.from_gmm(mixture, max_iter=max_iter, random_state=seed).fit(sequences)
    return model, projection, pca


def sweep_k(
    features: np.ndarray,
    lengths,
    k_values=(3, 4, 5, 6, 7, 8),
    n_folds: int = 4,
    restarts: int = 3,
    max_iter: int = 100,
    pca_components: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Sweep K, scoring each model by BIC and by held-out (unseen subject) likelihood.

    PCA is refitted inside every fold on training subjects only, so held-out
    subjects never influence the projection they are scored in.
    """
    features = np.asarray(features, dtype=float)
    lengths = np.asarray(lengths, dtype=int)
    blocks = subject_blocks(lengths, features)
    n_features = min(pca_components, features.shape[1])

    folds = [fold for fold in np.array_split(np.arange(len(blocks)), min(n_folds, len(blocks))) if len(fold)]

    rows = []
    for n_states in k_values:
        model, _, _ = fit_pipeline(
            features,
            lengths,
            n_states,
            pca_components=pca_components,
            restarts=restarts,
            max_iter=max_iter,
            seed=seed,
        )

        heldout_scores = []
        for fold in folds:
            held = {int(index) for index in fold}
            train_blocks = [block for index, block in enumerate(blocks) if index not in held]
            test_blocks = [block for index, block in enumerate(blocks) if index in held]
            if not train_blocks or not test_blocks:
                continue

            fold_model, _, fold_pca = fit_pipeline(
                np.vstack(train_blocks),
                [len(block) for block in train_blocks],
                n_states,
                pca_components=pca_components,
                restarts=restarts,
                max_iter=max_iter,
                seed=seed,
            )
            projected_test = [fold_pca.transform(block) for block in test_blocks]
            n_test_epochs = sum(len(block) for block in test_blocks)
            heldout_scores.append(fold_model.total_log_likelihood(projected_test) / n_test_epochs)

        rows.append(
            {
                "k": int(n_states),
                "n_parameters": hmm_n_parameters(n_states, n_features),
                "train_loglik": float(model.log_likelihood_),
                "bic": float(bic_hmm(model.log_likelihood_, n_states, n_features, len(features))),
                "heldout_loglik_per_epoch": float(np.mean(heldout_scores))
                if heldout_scores
                else float("nan"),
                "heldout_std": float(np.std(heldout_scores)) if heldout_scores else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def best_k_by_bic(table: pd.DataFrame) -> int:
    """K with the lowest BIC."""
    return int(table.loc[table["bic"].idxmin(), "k"])


def best_k_by_heldout(table: pd.DataFrame) -> int:
    """K with the highest held-out log likelihood per epoch."""
    return int(table.loc[table["heldout_loglik_per_epoch"].idxmax(), "k"])


# -- transition structure and dwell times (SS-10) ------------------------
def transition_counts(states, n_states: int | None = None) -> np.ndarray:
    """Count state-to-state transitions within one hypnogram."""
    states = np.asarray(states, dtype=int)
    n_states = n_states or (int(states.max()) + 1 if states.size else 0)
    counts = np.zeros((n_states, n_states), dtype=int)
    for previous, current in itertools.pairwise(states):
        counts[previous, current] += 1
    return counts


def transition_matrix_from_states(states, n_states: int | None = None) -> np.ndarray:
    """Row-normalised empirical transition matrix (empty rows become uniform)."""
    counts = transition_counts(states, n_states).astype(float)
    row_sums = counts.sum(axis=1, keepdims=True)
    return np.divide(
        counts,
        row_sums,
        out=np.full_like(counts, 1.0 / counts.shape[0]),
        where=row_sums > 0,
    )


def stationary_distribution(transmat) -> np.ndarray:
    """Left eigenvector of the transition matrix for eigenvalue 1."""
    transmat = np.asarray(transmat, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    stationary = np.abs(np.real(eigenvectors[:, index]))
    return stationary / stationary.sum()


def expected_dwell(transmat) -> np.ndarray:
    """Expected run length per state: 1 / (1 - a_ii)."""
    diagonal = np.diag(np.asarray(transmat, dtype=float))
    return 1.0 / np.maximum(1.0 - diagonal, 1e-12)


def run_lengths(states) -> tuple[np.ndarray, np.ndarray]:
    """Return (state id, run length) for every run of identical states."""
    states = np.asarray(states, dtype=int)
    if states.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    changes = np.flatnonzero(np.diff(states)) + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(states)]])
    return states[starts], ends - starts


def empirical_dwell(states, n_states: int | None = None) -> np.ndarray:
    """Mean observed run length per state (nan for states that never occur)."""
    states = np.asarray(states, dtype=int)
    n_states = n_states or (int(states.max()) + 1 if states.size else 0)
    ids, lengths = run_lengths(states)
    dwell = np.full(n_states, np.nan)
    for state in range(n_states):
        mask = ids == state
        if mask.any():
            dwell[state] = lengths[mask].mean()
    return dwell


def dwell_table(transmat, states) -> pd.DataFrame:
    """Expected (from the model) versus empirical (from Viterbi) dwell times."""
    return pd.DataFrame(
        {
            "state": np.arange(len(transmat)),
            "expected": expected_dwell(transmat),
            "empirical": empirical_dwell(states, n_states=len(transmat)),
        }
    )


# -- per-subject summaries and age comparison ----------------------------
def subject_metrics(states, subjects) -> pd.DataFrame:
    """Per-subject epoch count, transitions, transition rate and mean dwell."""
    states = np.asarray(states, dtype=int)
    subjects = np.asarray(subjects)
    rows = []
    for subject in np.unique(subjects):
        block = states[subjects == subject]
        transitions = int((np.diff(block) != 0).sum())
        rows.append(
            {
                "subject": int(subject),
                "n_epochs": len(block),
                "transitions": transitions,
                "transition_rate": transitions / max(len(block) - 1, 1),
                "mean_dwell": len(block) / (transitions + 1),
            }
        )
    return pd.DataFrame(rows)


def per_subject_ages(subjects, ages) -> dict[int, float]:
    """Map subject id to age, accepting per-epoch or per-subject age arrays."""
    subjects = np.asarray(subjects)
    ages = np.asarray(ages, dtype=float)
    unique = np.unique(subjects)
    per_epoch = len(ages) == len(subjects)
    mapping = {}
    for index, subject in enumerate(unique):
        values = ages[subjects == subject] if per_epoch else ages[index : index + 1]
        values = values[~np.isnan(values)]
        mapping[int(subject)] = float(values.mean()) if values.size else float("nan")
    return mapping


def _p_value(frame: pd.DataFrame, column: str) -> float:
    younger = frame.loc[frame["group"] == "younger", column].to_numpy(dtype=float)
    older = frame.loc[frame["group"] == "older", column].to_numpy(dtype=float)
    if younger.size == 0 or older.size == 0:
        return float("nan")
    return float(mannwhitneyu(younger, older, alternative="two-sided").pvalue)


def age_comparison(states, subjects, ages) -> dict:
    """Split subjects at the median age and compare transition/dwell behaviour."""
    frame = subject_metrics(states, subjects)
    frame["age"] = frame["subject"].map(per_subject_ages(subjects, ages))
    frame = frame.dropna(subset=["age"]).reset_index(drop=True)

    summary = pd.DataFrame(columns=["group", "n", "transition_rate", "mean_dwell"])
    if frame.empty:
        return {
            "subjects": frame,
            "summary": summary,
            "p_transition_rate": float("nan"),
            "p_mean_dwell": float("nan"),
        }

    cutoff = frame["age"].median()
    frame["group"] = np.where(frame["age"] <= cutoff, "younger", "older")
    summary = (
        frame.groupby("group")
        .agg(
            n=("subject", "size"),
            transition_rate=("transition_rate", "median"),
            mean_dwell=("mean_dwell", "median"),
        )
        .reset_index()
    )
    return {
        "subjects": frame,
        "summary": summary,
        "p_transition_rate": _p_value(frame, "transition_rate"),
        "p_mean_dwell": _p_value(frame, "mean_dwell"),
    }
