"""Reduce features with PCA, fit the GMM, then fit the HMM with Baum-Welch (SS-03, SS-06)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleepstates import features, gmm, hmm


def split_sequences(lengths, matrix: np.ndarray) -> list[np.ndarray]:
    """Split a stacked (T, D) matrix into one sequence per subject."""
    boundaries = np.cumsum(lengths)[:-1]
    return [chunk for chunk in np.split(matrix, boundaries) if len(chunk)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/features.npz", help="feature dataset")
    parser.add_argument("--model-out", default="data/model.npz", help="fitted parameters")
    parser.add_argument("--states-out", default="data/states.npz", help="per-epoch states")
    parser.add_argument("--components", type=int, default=5, help="PCA dimensions")
    parser.add_argument("--n-states", type=int, default=5, help="GMM components / HMM states")
    parser.add_argument("--restarts", type=int, default=3, help="GMM restarts")
    parser.add_argument("--hmm-iter", type=int, default=100, help="Baum-Welch iterations")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args(argv)

    dataset = features.load_feature_dataset(args.features)
    projection, pca = gmm.fit_pca(
        dataset["X"], n_components=args.components, random_state=args.seed
    )

    mixture = gmm.GMM(
        n_components=args.n_states, n_init=args.restarts, random_state=args.seed
    ).fit(projection)

    sequences = split_sequences(dataset["lengths"], projection)
    model = hmm.GaussianHMM.from_gmm(
        mixture, max_iter=args.hmm_iter, random_state=args.seed
    ).fit(sequences)

    # One Viterbi path per subject is the machine's hypnogram.
    states = np.concatenate([model.predict(sequence) for sequence in sequences])

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        weights=mixture.weights_,
        gmm_means=mixture.means_,
        gmm_covariances=mixture.covariances_,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        startprob=model.startprob_,
        transmat=model.transmat_,
        means=model.means_,
        covars=model.covars_,
        log_likelihood=np.float64(model.log_likelihood_),
    )

    states_path = Path(args.states_out)
    np.savez_compressed(
        states_path,
        states=states.astype(np.int32),
        labels=np.asarray(dataset["labels"], dtype=str),
        subjects=dataset["subjects"],
        ages=dataset["ages"],
        lengths=dataset["lengths"],
        projection=projection.astype(np.float32),
        feature_names=dataset["feature_names"],
    )

    n_params = gmm.n_parameters(args.n_states, args.components)
    print(f"PCA {args.components} dims, variance kept {pca.explained_variance_ratio_.sum():.3f}")
    print(f"GMM K={args.n_states} log-likelihood {mixture.log_likelihood_:.1f}")
    print(
        f"HMM Baum-Welch: {model.n_iter_} iterations, log-likelihood {model.log_likelihood_:.1f} "
        f"({model.log_likelihood_ / len(projection):.3f} per epoch)"
    )
    print(f"BIC (GMM, {n_params} params) {gmm.bic(mixture.log_likelihood_, n_params, len(projection)):.1f}")
    print(f"saved {model_path} and {states_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
