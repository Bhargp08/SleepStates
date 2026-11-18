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


def sticky_transmat(n_components: int, stay: float = 0.9) -> np.ndarray:
    """Transition matrix biased towards staying put, as sleep stages mostly do."""
    if n_components == 1:
        return np.ones((1, 1))
    matrix = np.full((n_components, n_components), (1.0 - stay) / (n_components - 1))
    np.fill_diagonal(matrix, stay)
    return matrix


def as_sequences(sequences) -> list[np.ndarray]:
    """Normalise a single (T, D) array or a list of them into a list."""
    if isinstance(sequences, np.ndarray) and sequences.ndim == 2:
        return [np.asarray(sequences, dtype=float)]
    return [np.atleast_2d(np.asarray(sequence, dtype=float)) for sequence in sequences]


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

    # -- parameter initialisation and Baum-Welch -----------------------
    @classmethod
    def from_gmm(cls, mixture, transmat=None, **kwargs) -> GaussianHMM:
        """Initialise emissions from a fitted GMM, as the project requires."""
        model = cls(n_components=mixture.n_components, **kwargs)
        model.startprob_ = np.array(mixture.weights_, dtype=float)
        model.means_ = np.array(mixture.means_, dtype=float)
        model.covars_ = np.array(mixture.covariances_, dtype=float)
        model.transmat_ = (
            sticky_transmat(model.n_components)
            if transmat is None
            else np.asarray(transmat, dtype=float)
        )
        return model

    def _init_from_mixture(self, sequences: list[np.ndarray]) -> None:
        from sleepstates.gmm import GMM

        stacked = np.vstack(sequences)
        mixture = GMM(
            n_components=self.n_components, random_state=self.random_state
        ).fit(stacked)
        self.startprob_ = np.array(mixture.weights_, dtype=float)
        self.means_ = np.array(mixture.means_, dtype=float)
        self.covars_ = np.array(mixture.covariances_, dtype=float)
        self.transmat_ = sticky_transmat(self.n_components)

    def _e_step(self, sequences: list[np.ndarray]):
        """Posteriors for every sequence and the summed log likelihood."""
        gammas, xis, total = [], [], 0.0
        for sequence in sequences:
            gamma, xi, log_likelihood = self.posterior(sequence)
            gammas.append(gamma)
            xis.append(xi)
            total += log_likelihood
        return gammas, xis, total

    def _m_step(self, sequences, gammas, xis) -> None:
        """Closed-form EM updates for start probs, transitions, means and covariances."""
        n_components = self.n_components
        n_features = sequences[0].shape[1]

        counts = np.zeros(n_components)
        means = np.zeros((n_components, n_features))
        starts = np.zeros(n_components)
        transitions = np.zeros((n_components, n_components))

        for sequence, gamma, xi in zip(sequences, gammas, xis, strict=True):
            counts += gamma.sum(axis=0)
            means += gamma.T @ sequence
            starts += gamma[0]
            transitions += xi.sum(axis=0)

        safe_counts = np.where(counts > 0, counts, 1.0)
        means /= safe_counts[:, None]

        covariances = np.zeros((n_components, n_features, n_features))
        for sequence, gamma in zip(sequences, gammas, strict=True):
            for state in range(n_components):
                diff = sequence - means[state]
                covariances[state] += (gamma[:, state][:, None] * diff).T @ diff
        covariances /= safe_counts[:, None, None]
        covariances += self.reg_covar * np.eye(n_features)

        row_sums = transitions.sum(axis=1, keepdims=True)
        transitions = np.divide(
            transitions, row_sums, out=np.full_like(transitions, 1.0 / n_components),
            where=row_sums > 0,
        )

        self.startprob_ = starts / len(sequences)
        self.transmat_ = transitions
        self.means_ = means
        self.covars_ = covariances

    def fit(self, sequences) -> GaussianHMM:
        """Baum-Welch on one or more sequences (one sequence per subject)."""
        data = as_sequences(sequences)
        if not data:
            raise ValueError("no sequences to fit")
        if self.means_ is None:
            self._init_from_mixture(data)

        history: list[float] = []
        previous = None
        for _ in range(self.max_iter):
            gammas, xis, log_likelihood = self._e_step(data)
            history.append(log_likelihood)
            self._m_step(data, gammas, xis)
            if previous is not None and abs(log_likelihood - previous) <= self.tol * abs(
                log_likelihood
            ):
                break
            previous = log_likelihood

        self.history_ = history
        self.log_likelihood_ = history[-1]
        self.n_iter_ = len(history)
        return self

    def total_log_likelihood(self, sequences) -> float:
        """Summed log likelihood of one or more sequences."""
        return sum(self.score(sequence) for sequence in as_sequences(sequences))

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
