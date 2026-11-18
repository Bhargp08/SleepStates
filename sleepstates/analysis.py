"""Model selection (SS-08) and stage-transition analysis (SS-10)."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
