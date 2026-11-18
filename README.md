# SleepStates

Unsupervised sleep staging with a hand-written Gaussian HMM on Sleep-EDF Expanded EEG, compared against expert labels. We extract per-epoch spectral features from a single EEG channel, compress them with PCA, fit a Gaussian mixture model, and then model the temporal structure of sleep with a Gaussian HMM implemented from scratch (forward-backward, Viterbi, Baum-Welch). The resulting hidden states are matched to expert sleep stages, and we report agreement metrics along with transition, dwell-time and age-related analyses against the annotated hypnograms.

## Setup

```bash
uv sync
```

## Data

Sleep-EDF Expanded (PhysioNet sleep-cassette) is downloaded on demand and cached outside git.
The command below fetches 20 subjects for night 1, cuts 30 s epochs of Fpz-Cz EEG, merges
R&K stages 3/4 into N3, drops movement/unscored epochs and trims wake to 30 min at either end:

```bash
uv run python scripts/download.py --subjects 0-19 --night 1
```

Epochs are written to `data/epochs/sub-XX_night1.npz` (`data/` and `*.edf` are git-ignored).

## Tests

```bash
uv run pytest
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
