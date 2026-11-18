"""PCA reduction and a from-scratch Gaussian mixture model fitted with EM (SS-03)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sklearn.decomposition import PCA

LOG_2PI = float(np.log(2.0 * np.pi))


def fit_pca(features, n_components: int = 5, random_state: int = 0):
    """Project features onto their leading principal components: (Z, pca)."""
    model = PCA(n_components=n_components, random_state=random_state)
    projection = model.fit_transform(np.asarray(features, dtype=float))
    return projection, model


def log_gaussian(points, mean, cov) -> np.ndarray:
    """Log N(x | mean, cov) for every row of points, via a Cholesky solve."""
    points = np.asarray(points, dtype=float)
    cov = np.asarray(cov, dtype=float)
    chol = np.linalg.cholesky(cov)
    diff = points - np.asarray(mean, dtype=float)
    solved = solve_triangular(chol, diff.T, lower=True).T
    log_det = 2.0 * np.log(np.diag(chol)).sum()
    return -0.5 * (points.shape[1] * LOG_2PI + log_det + (solved**2).sum(axis=1))


def kmeans_plusplus(points, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """Seed component centres with k-means++ (spread-out random starts)."""
    n_samples = len(points)
    centers = np.empty((n_clusters, points.shape[1]), dtype=float)
    centers[0] = points[rng.integers(n_samples)]
    closest = ((points - centers[0]) ** 2).sum(axis=1)
    for index in range(1, n_clusters):
        total = closest.sum()
        probabilities = closest / total if total > 0 else np.full(n_samples, 1.0 / n_samples)
        centers[index] = points[rng.choice(n_samples, p=probabilities)]
        closest = np.minimum(closest, ((points - centers[index]) ** 2).sum(axis=1))
    return centers


def lloyd(points, centers, n_iter: int = 10) -> np.ndarray:
    """A few hard-assignment k-means iterations to polish the k-means++ seeds."""
    centers = np.array(centers, dtype=float, copy=True)
    for _ in range(n_iter):
        distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assignments = distances.argmin(axis=1)
        for index in range(len(centers)):
            members = points[assignments == index]
            if len(members):
                centers[index] = members.mean(axis=0)
    return centers


def n_parameters(n_components: int, n_features: int) -> int:
    """Free parameters of a full-covariance GMM (weights, means, covariances)."""
    return (
        (n_components - 1)
        + n_components * n_features
        + n_components * n_features * (n_features + 1) // 2
    )


def bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """Bayesian information criterion; lower is better."""
    return -2.0 * log_likelihood + n_params * np.log(n_samples)


class GMM:
    """Full-covariance Gaussian mixture fitted by expectation-maximisation."""

    def __init__(
        self,
        n_components: int = 5,
        max_iter: int = 200,
        tol: float = 1e-6,
        reg_covar: float = 1e-6,
        n_init: int = 3,
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.random_state = random_state

    # -- parameters -----------------------------------------------------
    def _set_params(self, weights, means, covariances) -> None:
        self.weights_ = np.asarray(weights, dtype=float)
        self.means_ = np.asarray(means, dtype=float)
        self.covariances_ = np.asarray(covariances, dtype=float)

    def _initialise(self, points, seed: int):
        rng = np.random.default_rng(self.random_state + seed)
        centers = lloyd(points, kmeans_plusplus(points, self.n_components, rng))
        # Hard assignment to the seeded centres, then one supervised M-step.
        distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assignments = distances.argmin(axis=1)
        resp = np.zeros((len(points), self.n_components))
        resp[np.arange(len(points)), assignments] = 1.0
        return self._m_step(points, resp)

    # -- EM -------------------------------------------------------------
    def _log_joint(self, points) -> np.ndarray:
        """(n, K) log of weight times density, i.e. log p(x_n, z_n = k)."""
        points = np.asarray(points, dtype=float)
        return np.column_stack(
            [
                np.log(max(weight, 1e-300)) + log_gaussian(points, mean, cov)
                for weight, mean, cov in zip(
                    self.weights_, self.means_, self.covariances_, strict=True
                )
            ]
        )

    def _log_resp(self, points) -> np.ndarray:
        """(n, K) log responsibilities; rows sum to 1 in probability space."""
        joint = self._log_joint(points)
        return joint - logsumexp(joint, axis=1, keepdims=True)

    def _m_step(self, points, resp: np.ndarray):
        counts = resp.sum(axis=0) + 1e-10
        weights = counts / counts.sum()
        means = (resp.T @ points) / counts[:, None]
        n_features = points.shape[1]
        covariances = np.empty((self.n_components, n_features, n_features))
        for index in range(self.n_components):
            diff = points - means[index]
            scatter = (resp[:, index][:, None] * diff).T @ diff / counts[index]
            covariances[index] = scatter + self.reg_covar * np.eye(n_features)
        return weights, means, covariances

    def fit(self, points) -> GMM:
        """Fit with several restarts and keep the highest-likelihood solution."""
        points = np.asarray(points, dtype=float)
        if len(points) < self.n_components:
            raise ValueError("fewer samples than components")

        best_ll = -np.inf
        best_history: list[float] = []
        best_params = None
        for seed in range(self.n_init):
            self._set_params(*self._initialise(points, seed))
            history: list[float] = []
            previous = None
            for _ in range(self.max_iter):
                joint = self._log_joint(points)
                # Normalise before the M-step, but score before normalising:
                # normalised rows always have logsumexp() == 0.
                log_resp = joint - logsumexp(joint, axis=1, keepdims=True)
                log_likelihood = float(logsumexp(joint, axis=1).sum())
                history.append(log_likelihood)
                self._set_params(*self._m_step(points, np.exp(log_resp)))
                if previous is not None and abs(log_likelihood - previous) <= self.tol * abs(
                    log_likelihood
                ):
                    break
                previous = log_likelihood
            if history and history[-1] > best_ll:
                best_ll = history[-1]
                best_history = history
                best_params = (self.weights_, self.means_, self.covariances_)

        self._set_params(*best_params)
        self.history_ = best_history
        self.log_likelihood_ = best_ll
        self.n_iter_ = len(best_history)
        return self

    # -- inference ------------------------------------------------------
    def predict_proba(self, points) -> np.ndarray:
        """Posterior component membership per sample."""
        return np.exp(self._log_resp(np.asarray(points, dtype=float)))

    def predict(self, points) -> np.ndarray:
        """Hard component assignment (argmax posterior)."""
        return self.predict_proba(points).argmax(axis=1)

    def total_log_likelihood(self, points) -> float:
        """Sum of log densities over samples."""
        return float(logsumexp(self._log_joint(points), axis=1).sum())

    def score(self, points) -> float:
        """Mean log likelihood per sample."""
        points = np.asarray(points, dtype=float)
        return self.total_log_likelihood(points) / len(points)

    def score_samples(self, points) -> np.ndarray:
        """Per-sample log likelihood."""
        return logsumexp(self._log_joint(points), axis=1)
