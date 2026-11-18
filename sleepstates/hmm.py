"""Hand-written Gaussian HMM with log-space forward-backward (SS-04)."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from sleepstates.gmm import log_gaussian


def safe_log(probabilities) -> np.ndarray:
    """log() that maps exact zeros to -inf instead of warning."""
    values = np.asarray(probabilities, dtype=float)
    with np.errstate(divide="ignore"):
        return np.log(values)


class GaussianHMM:
    """Single-Gaussian-per-state HMM; every recursion runs in log space."""

    def __init__(
        self,
        n_components: int = 5,
        max_iter: int = 100,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        # Uniform defaults; emission parameters are set by set_params or fit.
        self.startprob_ = np.full(n_components, 1.0 / n_components)
        self.transmat_ = np.full((n_components, n_components), 1.0 / n_components)
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None

    # -- parameters -----------------------------------------------------
    def set_params(self, startprob=None, transmat=None, means=None, covars=None):
        """Assign parameters directly (used by tests and the hmmlearn check)."""
        if startprob is not None:
            self.startprob_ = np.asarray(startprob, dtype=float)
        if transmat is not None:
            self.transmat_ = np.asarray(transmat, dtype=float)
        if means is not None:
            self.means_ = np.asarray(means, dtype=float)
        if covars is not None:
            self.covars_ = np.asarray(covars, dtype=float)
        return self

    def _check_ready(self) -> None:
        if self.means_ is None or self.covars_ is None:
            raise RuntimeError("model has no emission parameters; call fit or set_params")

    @property
    def n_features(self) -> int:
        self._check_ready()
        return int(self.means_.shape[1])

    # -- emissions ------------------------------------------------------
    def log_emission(self, data) -> np.ndarray:
        """(T, K) log N(x_t | mu_k, Sigma_k)."""
        self._check_ready()
        points = np.atleast_2d(np.asarray(data, dtype=float))
        if points.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {points.shape[1]}")
        return np.column_stack(
            [
                log_gaussian(points, mean, covar)
                for mean, covar in zip(self.means_, self.covars_, strict=True)
            ]
        )

    # -- forward / backward ---------------------------------------------
    def _forward(self, log_emit: np.ndarray):
        n_steps = log_emit.shape[0]
        log_alpha = np.empty_like(log_emit)
        log_alpha[0] = safe_log(self.startprob_) + log_emit[0]
        log_trans = safe_log(self.transmat_)
        for step in range(1, n_steps):
            log_alpha[step] = log_emit[step] + logsumexp(
                log_alpha[step - 1][:, None] + log_trans, axis=0
            )
        return log_alpha, float(logsumexp(log_alpha[-1]))

    def _backward(self, log_emit: np.ndarray) -> np.ndarray:
        n_steps = log_emit.shape[0]
        log_beta = np.zeros_like(log_emit)
        log_trans = safe_log(self.transmat_)
        for step in range(n_steps - 2, -1, -1):
            log_beta[step] = logsumexp(
                log_trans + (log_emit[step + 1] + log_beta[step + 1])[None, :], axis=1
            )
        return log_beta

    def forward(self, data):
        """Log forward messages and the sequence log likelihood."""
        return self._forward(self.log_emission(data))

    def backward(self, data) -> np.ndarray:
        """Log backward messages."""
        return self._backward(self.log_emission(data))

    def score(self, data) -> float:
        """Log likelihood of a sequence under the current parameters."""
        return self.forward(data)[1]

    # -- Viterbi --------------------------------------------------------
    def viterbi(self, data):
        """Most likely state sequence and its joint log probability."""
        log_emit = self.log_emission(data)
        n_steps = log_emit.shape[0]
        log_delta = np.empty_like(log_emit)
        backpointer = np.zeros((n_steps, self.n_components), dtype=int)
        log_trans = safe_log(self.transmat_)

        log_delta[0] = safe_log(self.startprob_) + log_emit[0]
        for step in range(1, n_steps):
            candidates = log_delta[step - 1][:, None] + log_trans
            backpointer[step] = candidates.argmax(axis=0)
            log_delta[step] = log_emit[step] + candidates.max(axis=0)

        path = np.empty(n_steps, dtype=int)
        path[-1] = int(log_delta[-1].argmax())
        for step in range(n_steps - 1, 0, -1):
            path[step - 1] = backpointer[step, path[step]]
        return path, float(log_delta[-1].max())

    def predict(self, data) -> np.ndarray:
        """Viterbi state sequence (the hypnogram we compare to experts)."""
        return self.viterbi(data)[0]

    # -- posteriors -----------------------------------------------------
    def posterior(self, data):
        """Return (gamma, xi, log_likelihood) in probability space."""
        log_emit = self.log_emission(data)
        log_alpha, log_likelihood = self._forward(log_emit)
        log_beta = self._backward(log_emit)

        log_gamma = log_alpha + log_beta - log_likelihood
        gamma = np.exp(log_gamma)
        gamma /= gamma.sum(axis=1, keepdims=True)

        n_steps = log_emit.shape[0]
        xi = np.empty((max(n_steps - 1, 0), self.n_components, self.n_components))
        log_trans = safe_log(self.transmat_)
        for step in range(n_steps - 1):
            log_xi = (
                log_alpha[step][:, None]
                + log_trans
                + (log_emit[step + 1] + log_beta[step + 1])[None, :]
                - log_likelihood
            )
            block = np.exp(log_xi)
            xi[step] = block / block.sum()
        return gamma, xi, log_likelihood
