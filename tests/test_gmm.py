"""Tests for sleepstates.gmm (SS-03): from-scratch EM on synthetic Gaussians."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal
from sklearn.metrics import adjusted_rand_score

from sleepstates import gmm


def _two_clusters(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    first = rng.multivariate_normal([-4.0, -4.0], [[1.0, 0.3], [0.3, 1.0]], n)
    second = rng.multivariate_normal([4.0, 4.0], [[1.0, -0.2], [-0.2, 1.0]], n)
    return np.vstack([first, second]), np.array([0] * n + [1] * n)


def _sorted(means: np.ndarray) -> np.ndarray:
    return means[np.lexsort(means.T[::-1])]


def test_log_gaussian_matches_scipy():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(50, 3))
    mean = np.array([0.5, -1.0, 2.0])
    cov = np.array([[2.0, 0.4, 0.1], [0.4, 1.0, -0.3], [0.1, -0.3, 0.5]])

    ours = gmm.log_gaussian(points, mean, cov)
    reference = multivariate_normal(mean, cov).logpdf(points)

    np.testing.assert_allclose(ours, reference, atol=1e-9)


def test_gmm_recovers_two_cluster_means():
    features, _ = _two_clusters()
    model = gmm.GMM(n_components=2, random_state=0, n_init=3).fit(features)

    expected = np.array([[-4.0, -4.0], [4.0, 4.0]])
    np.testing.assert_allclose(_sorted(model.means_), expected, atol=0.4)
    np.testing.assert_allclose(sorted(model.weights_), [0.5, 0.5], atol=0.05)


def test_gmm_log_likelihood_never_decreases():
    features, _ = _two_clusters(n=120)
    model = gmm.GMM(n_components=2, random_state=0, n_init=1, max_iter=40).fit(features)

    history = np.asarray(model.history_)
    assert len(history) > 2
    # EM is monotone; allow only floating-point noise between iterations.
    assert np.all(np.diff(history) >= -1e-6)
    assert model.log_likelihood_ == pytest.approx(history[-1], rel=1e-9)


def test_gmm_predictions_match_the_generating_clusters():
    features, truth = _two_clusters()
    model = gmm.GMM(n_components=2, random_state=0, n_init=3).fit(features)

    assert adjusted_rand_score(truth, model.predict(features)) == pytest.approx(1.0)


def test_gmm_posteriors_are_normalised():
    features, _ = _two_clusters(n=60)
    model = gmm.GMM(n_components=3, random_state=1, n_init=2).fit(features)

    posteriors = model.predict_proba(features)

    assert posteriors.shape == (len(features), 3)
    assert np.all(posteriors >= 0.0)
    np.testing.assert_allclose(posteriors.sum(axis=1), 1.0, atol=1e-10)


def test_parameter_count_and_bic_formula():
    assert gmm.n_parameters(2, 5) == 1 + 2 * 5 + 2 * 5 * 6 // 2
    assert gmm.bic(-100.0, 11, 400) == pytest.approx(2 * 100.0 + 11 * np.log(400))


def test_bic_prefers_two_components_over_three():
    features, _ = _two_clusters()
    scores = {}
    for k in (2, 3):
        model = gmm.GMM(n_components=k, random_state=0, n_init=3).fit(features)
        scores[k] = gmm.bic(model.log_likelihood_, gmm.n_parameters(k, features.shape[1]), len(features))

    assert scores[2] < scores[3]


def test_fit_pca_returns_documented_shape_and_variance():
    rng = np.random.default_rng(2)
    latent = rng.normal(size=(300, 2))
    mixing = np.array([[1.0, 0.2], [0.5, -1.0], [0.0, 2.0]])
    features = latent @ mixing.T

    projected, model = gmm.fit_pca(features, n_components=2)

    assert projected.shape == (300, 2)
    # The data has rank 2, so two components explain essentially all variance.
    assert model.explained_variance_ratio_.sum() > 0.999


def test_gmm_initialisation_is_reproducible():
    features, _ = _two_clusters(n=50)
    first = gmm.GMM(n_components=3, random_state=7, n_init=1).fit(features)
    second = gmm.GMM(n_components=3, random_state=7, n_init=1).fit(features)

    np.testing.assert_allclose(first.means_, second.means_)
