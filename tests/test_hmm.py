"""Tests for sleepstates.hmm (SS-04 forward-backward and later HMM units)."""

import itertools

import numpy as np
from scipy.special import logsumexp

from sleepstates import hmm


def toy_model(n_components: int = 2) -> hmm.GaussianHMM:
    """A two-state scalar Gaussian HMM with hand-set parameters."""
    model = hmm.GaussianHMM(n_components=2)
    model.set_params(
        startprob=np.array([0.6, 0.4]),
        transmat=np.array([[0.7, 0.3], [0.2, 0.8]]),
        means=np.array([[-1.0], [1.0]]),
        covars=np.array([[[0.5]], [[0.8]]]),
    )
    return model


def observations(n: int = 3, seed: int = 0) -> np.ndarray:
    """A short sequence where both states stay plausible."""
    rng = np.random.default_rng(seed)
    return rng.normal(scale=1.5, size=(n, 1))


def brute_force(model: hmm.GaussianHMM, data: np.ndarray):
    """Exact log-likelihood and best path by enumerating all state sequences."""
    emit = model.log_emission(data)
    log_start, log_trans = np.log(model.startprob_), np.log(model.transmat_)
    total = -np.inf
    best_score, best_path = -np.inf, None
    for path in itertools.product(range(model.n_components), repeat=len(data)):
        score = log_start[path[0]] + emit[0, path[0]]
        for step in range(1, len(data)):
            score += log_trans[path[step - 1], path[step]] + emit[step, path[step]]
        total = np.logaddexp(total, score)
        if score > best_score:
            best_score, best_path = score, path
    return total, best_score, best_path


def test_forward_matches_brute_force_enumeration():
    model = toy_model()
    data = observations()
    log_alpha, log_likelihood = model.forward(data)

    exact, _, _ = brute_force(model, data)

    assert log_alpha.shape == (len(data), 2)
    np.testing.assert_allclose(log_likelihood, exact, atol=1e-10)


def test_forward_log_likelihood_equals_brute_force():
    model = toy_model()
    data = observations()
    _, log_likelihood = model.forward(data)
    exact, _, _ = brute_force(model, data)

    np.testing.assert_allclose(log_likelihood, exact, atol=1e-10)


def test_backward_matches_alpha_beta_identity():
    model = toy_model()
    data = observations()
    log_alpha, _ = model.forward(data)
    log_beta = model.backward(data)

    assert log_beta.shape == (len(data), 2)
    # At the first time step alpha + beta collapses to the sequence likelihood.
    np.testing.assert_allclose(
        logsumexp(log_alpha[0] + log_beta[0]), model.score(data), atol=1e-10
    )


def test_gamma_rows_sum_to_one():
    model = toy_model()
    data = observations()
    gamma, xi, log_likelihood = model.posterior(data)

    assert gamma.shape == (len(data), 2)
    assert xi.shape == (len(data) - 1, 2, 2)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(gamma >= 0.0)
    np.testing.assert_allclose(logsumexp(np.log(gamma), axis=1), 0.0, atol=1e-8)
    assert np.isfinite(log_likelihood)


def test_xi_marginals_reproduce_gamma():
    model = toy_model()
    data = observations(5)
    gamma, xi, _ = model.posterior(data)

    # Summing the pairwise posteriors over the destination state gives gamma.
    np.testing.assert_allclose(xi.sum(axis=2), gamma[:-1], atol=1e-10)
    np.testing.assert_allclose(xi.sum(axis=1), gamma[1:], atol=1e-10)


def test_score_agrees_with_forward():
    model = toy_model()
    data = observations()
    _, log_likelihood = model.forward(data)

    np.testing.assert_allclose(model.score(data), log_likelihood, atol=1e-12)


def test_single_state_model_is_a_plain_gaussian():
    model = hmm.GaussianHMM(n_components=1)
    model.set_params(
        startprob=np.array([1.0]),
        transmat=np.array([[1.0]]),
        means=np.array([[0.5]]),
        covars=np.array([[[2.0]]]),
    )
    data = observations(4)
    gamma, xi, log_likelihood = model.posterior(data)

    np.testing.assert_allclose(gamma, 1.0, atol=1e-12)
    assert xi.shape == (len(data) - 1, 1, 1)
    np.testing.assert_allclose(log_likelihood, model.log_emission(data).sum(), atol=1e-12)


def test_single_observation_sequence():
    model = toy_model()
    data = observations(1)
    gamma, xi, log_likelihood = model.posterior(data)

    assert xi.shape == (0, 2, 2)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-12)
    expected = logsumexp(np.log(model.startprob_) + model.log_emission(data)[0])
    np.testing.assert_allclose(log_likelihood, expected, atol=1e-12)


def test_emission_log_probabilities_match_scipy():
    model = toy_model()
    data = observations()
    from scipy.stats import norm

    expected = np.column_stack(
        [
            norm(loc=mean[0], scale=np.sqrt(covar[0, 0])).logpdf(data[:, 0])
            for mean, covar in zip(model.means_, model.covars_, strict=True)
        ]
    )
    np.testing.assert_allclose(model.log_emission(data), expected, atol=1e-10)
