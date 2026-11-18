"""Tests that our hand-written HMM agrees with hmmlearn (SS-07)."""

import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM as ReferenceHMM

from sleepstates import hmm

ATOL = 1e-6


def random_parameters(seed: int = 0, n_components: int = 3, n_features: int = 2):
    """Random but valid HMM parameters."""
    rng = np.random.default_rng(seed)
    startprob = rng.random(n_components)
    startprob /= startprob.sum()

    transmat = rng.random((n_components, n_components)) + 4.0 * np.eye(n_components)
    transmat /= transmat.sum(axis=1, keepdims=True)

    means = rng.normal(scale=2.0, size=(n_components, n_features))

    covars = np.empty((n_components, n_features, n_features))
    for state in range(n_components):
        factor = rng.normal(size=(n_features, n_features))
        covars[state] = factor @ factor.T / n_features + 0.5 * np.eye(n_features)
    return startprob, transmat, means, covars


def random_data(means, n_per_state: int = 60, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack(
        [rng.normal(mean, 1.0, size=(n_per_state, len(mean))) for mean in means]
    )


def make_models(startprob, transmat, means, covars, **kwargs):
    ours = hmm.GaussianHMM(n_components=len(startprob))
    ours.set_params(startprob=startprob, transmat=transmat, means=means, covars=covars)

    reference = ReferenceHMM(
        n_components=len(startprob),
        covariance_type="full",
        init_params="",
        **kwargs,
    )
    reference.startprob_ = startprob
    reference.transmat_ = transmat
    reference.means_ = means
    reference.covars_ = covars
    return ours, reference


def test_log_likelihood_matches_hmmlearn():
    params = random_parameters()
    data = random_data(params[2])
    ours, reference = make_models(*params)

    np.testing.assert_allclose(ours.score(data), reference.score(data), atol=ATOL)


def test_log_likelihood_matches_hmmlearn_for_multiple_sequences():
    params = random_parameters(seed=3)
    data = random_data(params[2], n_per_state=40, seed=4)
    lengths = [len(data) // 2, len(data) - len(data) // 2]
    ours, reference = make_models(*params)

    ours_total = ours.score(data[: lengths[0]]) + ours.score(data[lengths[0] :])
    reference_total = reference.score(data, lengths)

    np.testing.assert_allclose(ours_total, reference_total, atol=ATOL)


def test_posteriors_match_hmmlearn():
    params = random_parameters(seed=5)
    data = random_data(params[2], seed=6)
    ours, reference = make_models(*params)

    gamma, _, _ = ours.posterior(data)

    np.testing.assert_allclose(gamma, reference.predict_proba(data), atol=ATOL)


def test_viterbi_path_and_score_match_hmmlearn():
    params = random_parameters(seed=7)
    data = random_data(params[2], seed=8)
    ours, reference = make_models(*params)

    path, log_probability = ours.viterbi(data)
    reference_logprob, reference_path = reference.decode(data, algorithm="viterbi")

    np.testing.assert_array_equal(path, reference_path)
    np.testing.assert_allclose(log_probability, reference_logprob, atol=ATOL)


def test_one_baum_welch_step_matches_hmmlearn():
    params = random_parameters(seed=9)
    data = random_data(params[2], seed=10)

    # Zero priors/jitter on both sides: hmmlearn's covars_prior=0.01 MAP term
    # would otherwise shift every covariance entry.
    ours, reference = make_models(
        *params, min_covar=0.0, covars_prior=0.0, n_iter=1, tol=0.0
    )
    ours.max_iter = 1
    ours.tol = 0.0
    ours.reg_covar = 0.0

    ours.fit(data)
    reference.fit(data)

    np.testing.assert_allclose(ours.startprob_, reference.startprob_, atol=ATOL)
    np.testing.assert_allclose(ours.transmat_, reference.transmat_, atol=ATOL)
    np.testing.assert_allclose(ours.means_, reference.means_, atol=ATOL)
    np.testing.assert_allclose(ours.covars_, reference.covars_, atol=ATOL)


def test_baum_welch_step_matches_hmmlearn_with_multiple_sequences():
    params = random_parameters(seed=11)
    data = random_data(params[2], n_per_state=40, seed=12)
    lengths = [len(data) // 3, len(data) // 3, len(data) - 2 * (len(data) // 3)]

    ours, reference = make_models(
        *params, min_covar=0.0, covars_prior=0.0, n_iter=1, tol=0.0
    )
    ours.max_iter = 1
    ours.tol = 0.0
    ours.reg_covar = 0.0

    chunks = np.split(data, np.cumsum(lengths)[:-1])
    ours.fit(chunks)
    reference.fit(data, lengths)

    np.testing.assert_allclose(ours.transmat_, reference.transmat_, atol=ATOL)
    np.testing.assert_allclose(ours.means_, reference.means_, atol=ATOL)
    np.testing.assert_allclose(ours.covars_, reference.covars_, atol=ATOL)


@pytest.mark.parametrize("n_components", [1, 4])
def test_equivalence_for_other_state_counts(n_components):
    params = random_parameters(seed=13 + n_components, n_components=n_components)
    data = random_data(params[2], n_per_state=25, seed=14)
    ours, reference = make_models(*params)

    np.testing.assert_allclose(ours.score(data), reference.score(data), atol=ATOL)
    gamma, _, _ = ours.posterior(data)
    np.testing.assert_allclose(gamma, reference.predict_proba(data), atol=ATOL)
