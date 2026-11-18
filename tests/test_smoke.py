"""Smoke test: the package and every module import and expose a version."""

import sleepstates
from sleepstates import analysis, baseline, data, evaluate, features, gmm, hmm, plots


def test_version() -> None:
    assert sleepstates.__version__ == "0.0.1"


def test_modules_import() -> None:
    for module in (analysis, baseline, data, evaluate, features, gmm, hmm, plots):
        assert module.__name__.startswith("sleepstates.")
