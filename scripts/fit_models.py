"""Reduce features with PCA and fit the from-scratch GMM (SS-03)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleepstates import features, gmm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/features.npz", help="feature dataset")
    parser.add_argument("--out", default="data/gmm.npz", help="where to save the fitted model")
    parser.add_argument("--components", type=int, default=5, help="PCA dimensions")
    parser.add_argument("--n-states", type=int, default=5, help="GMM components / HMM states")
    parser.add_argument("--restarts", type=int, default=3, help="EM restarts")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args(argv)

    dataset = features.load_feature_dataset(args.features)
    projection, pca = gmm.fit_pca(dataset["X"], n_components=args.components, random_state=args.seed)

    model = gmm.GMM(
        n_components=args.n_states,
        n_init=args.restarts,
        random_state=args.seed,
    ).fit(projection)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        weights=model.weights_,
        means=model.means_,
        covariances=model.covariances_,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        projection=projection.astype(np.float32),
        log_likelihood=np.float64(model.log_likelihood_),
    )

    n_params = gmm.n_parameters(args.n_states, args.components)
    print(f"PCA {args.components} dims, variance kept {pca.explained_variance_ratio_.sum():.3f}")
    print(f"GMM K={args.n_states} log-likelihood {model.log_likelihood_:.1f} ({model.n_iter_} iters)")
    print(f"BIC {gmm.bic(model.log_likelihood_, n_params, len(projection)):.1f}")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
