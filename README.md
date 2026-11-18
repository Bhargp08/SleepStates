# SleepStates

Unsupervised sleep staging with a hand-written Gaussian HMM on Sleep-EDF Expanded EEG, compared
against expert labels. We extract per-epoch spectral features from a single EEG channel, compress
them with PCA, fit a Gaussian mixture model, and then model the temporal structure of sleep with a
Gaussian HMM implemented from scratch in NumPy (forward-backward, Viterbi, Baum-Welch). The hidden
states are matched to expert AASM stages, and we report agreement metrics plus transition, dwell-time
and age-related analyses against the annotated hypnograms. A supervised YASA model on the same
recordings provides the upper bound.

No data and no results are committed: `data/`, `*.edf` and `outputs/` are git-ignored. The whole test
suite runs on synthetic fixtures, so CI never downloads anything.

## Setup

Python 3.10, managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Quick start

```bash
uv run python scripts/run_all.py --subjects 0-19 --night 1
```

That downloads the nights, extracts features, sweeps K, fits PCA/GMM/HMM, evaluates against the
experts, runs the transition/dwell/age analysis, runs the YASA baseline and writes the interactive
hypnogram viewer.

## Step by step

```bash
uv run python scripts/download.py --subjects 0-19 --night 1   # data/epochs/sub-XX_night1.npz
uv run python scripts/extract_features.py                     # data/features.npz
uv run python scripts/select_k.py                             # outputs/model_selection.{csv,png}
uv run python scripts/fit_models.py --n-states 5              # data/model.npz, data/states.npz
uv run python scripts/evaluate.py                             # outputs/evaluation.json, per-stage, confusion
uv run python scripts/analyze.py                              # transitions, dwell times, age groups
uv run python scripts/baseline.py                             # YASA supervised baseline
uv run python scripts/viewer.py                               # outputs/hypnogram.html
```

`scripts/download.py` fetches 20 subjects of night 1 by default, cuts 30 s epochs of Fpz-Cz EEG,
merges R&K stages 3/4 into N3, drops movement/unscored epochs and trims wake to 30 min at either end.

## Outputs

| File | Contents |
| --- | --- |
| `outputs/model_selection.csv` / `.png` | BIC and held-out (leave-subjects-out) likelihood per K |
| `outputs/best_k.json` | K chosen by BIC and by held-out likelihood |
| `outputs/evaluation.json` | state-to-stage mapping, accuracy, Cohen's kappa, adjusted Rand index |
| `outputs/confusion_matrix.csv` / `.png` | machine states vs expert stages |
| `outputs/per_stage.csv` | per-stage recall, precision and support |
| `outputs/transitions.csv`, `transition_matrix.png` | fitted transition matrix |
| `outputs/dwell_times.csv` / `.png` | expected `1/(1-a_ii)` vs empirical dwell times |
| `outputs/age_summary.csv`, `age_comparison.png` | younger vs older half, Mann-Whitney U |
| `outputs/baseline.json`, `baseline.csv` | YASA kappa per subject and pooled |
| `outputs/hypnogram.html` | expert vs machine hypnograms, dropdown per subject |

## Tests and CI

```bash
uv run ruff check .
uv run pytest
```

GitHub Actions runs both on every push and pull request, on Python 3.10 with `uv sync --locked`.

## Layout

```
sleepstates/          package
  data.py             Sleep-EDF download, 30 s epoching, wake trimming, .npz caching
  features.py         multitaper band powers, ratios, per-subject z-scoring
  gmm.py              PCA and a from-scratch full-covariance GMM (EM, k-means++ init)
  hmm.py              Gaussian HMM: forward-backward, Viterbi, Baum-Welch
  evaluate.py         Hungarian / majority state matching, kappa, ARI, per-stage scores
  analysis.py         K sweep (BIC + held-out), transitions, dwell times, age comparison
  baseline.py         YASA supervised staging baseline
  plots.py            matplotlib figures and the Plotly hypnogram viewer
scripts/              download, features, select_k, fit_models, evaluate, analyze, baseline,
                      viewer, run_all
tests/                synthetic-fixture tests (no downloads)
docs/writeup.md       write-up with EM derivations and the disagreement analysis
```

## Units

| Unit | Description |
| --- | --- |
| SS-00 | Scaffold |
| SS-01 | Data download + epoching |
| SS-02 | Spectral features |
| SS-03 | PCA + GMM |
| SS-04 | HMM forward-backward |
| SS-05 | Viterbi |
| SS-06 | Baum-Welch |
| SS-07 | hmmlearn equivalence check |
| SS-08 | K sweep (BIC + held-out LL) |
| SS-09 | Matching to expert stages + metrics |
| SS-10 | Transitions / dwell / age analysis |
| SS-11 | YASA baseline |
| SS-12 | Hypnogram viewer |
| SS-13 | Write-up |
