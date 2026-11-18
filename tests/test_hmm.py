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


def path_log_probability(model: hmm.GaussianHMM, data: np.ndarray, path) -> float:
    """Joint log probability of one state sequence."""
    emit = model.log_emission(data)
    score = np.log(model.startprob_)[path[0]] + emit[0, path[0]]
    for step in range(1, len(path)):
        score += np.log(model.transmat_)[path[step - 1], path[step]] + emit[step, path[step]]
    return score


def test_viterbi_matches_brute_force_best_path():
    model = toy_model()
    data = observations(4)
    path, log_probability = model.viterbi(data)
    _, best_score, best_path = brute_force(model, data)

    assert tuple(int(state) for state in path) == best_path
    np.testing.assert_allclose(log_probability, best_score, atol=1e-10)
    np.testing.assert_allclose(path_log_probability(model, data, path), log_probability, atol=1e-10)


def test_viterbi_matches_brute_force_with_three_states():
    model = hmm.GaussianHMM(n_components=3)
    model.set_params(
        startprob=np.array([0.5, 0.3, 0.2]),
        transmat=np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]]),
        means=np.array([[-1.0], [0.0], [1.5]]),
        covars=np.array([[[0.4]], [[1.2]], [[0.7]]]),
    )
    data = observations(3, seed=4)

    path, log_probability = model.viterbi(data)
    _, best_score, best_path = brute_force(model, data)

    assert tuple(int(state) for state in path) == best_path
    np.testing.assert_allclose(log_probability, best_score, atol=1e-10)


def test_viterbi_handles_single_observation():
    model = toy_model()
    data = observations(1)

    path, log_probability = model.viterbi(data)

    assert path.shape == (1,)
    best_state = int(np.argmax(np.log(model.startprob_) + model.log_emission(data)[0]))
    assert int(path[0]) == best_state
    np.testing.assert_allclose(log_probability, path_log_probability(model, data, path), atol=1e-12)


def sample_sequences(model: hmm.GaussianHMM, n_sequences: int, n_steps: int, seed: int = 0):
    """Draw synthetic sequences from a model with known parameters."""
    rng = np.random.default_rng(seed)
    sequences = []
    for _ in range(n_sequences):
        state = rng.choice(model.n_components, p=model.startprob_)
        rows = [rng.multivariate_normal(model.means_[state], model.covars_[state])]
        for _ in range(n_steps - 1):
            state = rng.choice(model.n_components, p=model.transmat_[state])
            rows.append(rng.multivariate_normal(model.means_[state], model.covars_[state]))
        sequences.append(np.asarray(rows))
    return sequences


def sticky_model() -> hmm.GaussianHMM:
    """A well-separated, slow-switching 3-state model to recover."""
    model = hmm.GaussianHMM(n_components=3)
    model.set_params(
        startprob=np.array([1.0, 0.0, 0.0]),
        transmat=np.array([[0.94, 0.05, 0.01], [0.03, 0.94, 0.03], [0.01, 0.05, 0.94]]),
        means=np.array([[-4.0, 0.0], [0.0, 4.0], [4.0, 0.0]]),
        covars=np.array([np.eye(2) * 0.5, np.eye(2) * 0.5, np.eye(2) * 0.5]),
    )
    return model


def test_baum_welch_log_likelihood_is_non_decreasing():
    truth = sticky_model()
    sequences = sample_sequences(truth, n_sequences=3, n_steps=150, seed=11)

    model = hmm.GaussianHMM(n_components=3, max_iter=40, tol=1e-7, random_state=0)
    model.fit(sequences)

    history = np.asarray(model.history_)
    # It may converge quickly from a good GMM initialisation; what matters is
    # that every iteration improves the likelihood and the run terminates.
    assert 2 <= len(history) <= model.max_iter
    assert np.all(np.diff(history) >= -1e-6)
    assert history[-1] > history[0]
    np.testing.assert_allclose(model.log_likelihood_, history[-1], atol=1e-9)


def test_baum_welch_recovers_known_parameters():
    truth = sticky_model()
    sequences = sample_sequences(truth, n_sequences=4, n_steps=400, seed=5)

    model = hmm.GaussianHMM(n_components=3, max_iter=60, tol=1e-6, random_state=0)
    model.fit(sequences)

    np.testing.assert_allclose(
        np.sort(model.means_[:, 0]), np.sort(truth.means_[:, 0]), atol=0.7
    )
    np.testing.assert_allclose(
        np.sort(model.means_[:, 1]), np.sort(truth.means_[:, 1]), atol=0.7
    )
    # The generating chain is sticky; the fit should stay sticky.
    assert np.all(np.diag(model.transmat_) > 0.6)
    np.testing.assert_allclose(model.transmat_.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(model.startprob_.sum(), 1.0, atol=1e-10)
    # Covariances must stay positive definite and normalised weights unused.
    for covar in model.covars_:
        assert np.all(np.linalg.eigvalsh(covar) > 0)


def test_fit_accepts_a_single_sequence():
    sequences = sample_sequences(sticky_model(), n_sequences=1, n_steps=200, seed=3)

    model = hmm.GaussianHMM(n_components=3, max_iter=10, random_state=0).fit(sequences[0])

    assert len(model.history_) >= 2
    assert np.isfinite(model.log_likelihood_)


def test_from_gmm_initialises_emissions_from_mixture_components():
    from sleepstates import gmm

    rng = np.random.default_rng(2)
    points = np.vstack([rng.normal(loc=loc, size=(100, 2)) for loc in ([-3, -3], [3, 3])])
    mixture = gmm.GMM(n_components=2, random_state=0).fit(points)

    model = hmm.GaussianHMM.from_gmm(mixture)

    assert model.n_components == 2
    np.testing.assert_allclose(model.means_, mixture.means_)
    np.testing.assert_allclose(model.covars_, mixture.covariances_)
    np.testing.assert_allclose(model.startprob_, mixture.weights_)
    # A sticky transition matrix is a sane default for sleep staging.
    assert np.all(np.diag(model.transmat_) > 0.5)
    np.testing.assert_allclose(model.transmat_.sum(axis=1), 1.0, atol=1e-12)


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
