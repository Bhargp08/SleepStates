# Machine sleep staging without labels

## 1. Question

Sleep is scored by hand: a trained scorer reads a night of EEG in 30 s epochs and assigns a stage
(W, N1, N2, N3, REM). Can the *structure* of sleep be recovered without ever showing the model a
label — and if we then line the machine's own states up with the expert's stages, how close do we
get, where does it break, and what does that tell us about the information in a single EEG channel?

We answer it with a deliberately transparent model: spectral features, a Gaussian mixture, and a
Gaussian hidden Markov model written from scratch in NumPy. The model sees only EEG spectra and
their temporal order. Expert hypnograms are used afterwards, only for scoring.

## 2. Data

- **Sleep-EDF Expanded** (PhysioNet, `sleep-cassette` subset), night 1, 20 subjects by default
  (`scripts/download.py --subjects 0-19 --night 1`).
- Single EEG derivation **Fpz-Cz** sampled at 100 Hz, plus the expert hypnogram shipped with each
  record.
- 30 s epochs. Sleep-EDF is scored with R&K rules, so stages 3 and 4 are merged into **N3** to match
  AASM; `Movement time`, `Sleep stage ?` and unscored stretches are **dropped** rather than guessed.
- Wake is trimmed to at most 30 min before the first and after the last sleep epoch, so the night is
  not dominated by quiet wake.
- Ages come from the dataset's own `SC-subjects.xls` table, which the download step caches; when it
  cannot be fetched the age analysis is skipped instead of being faked.

Nothing is committed: `data/`, `*.edf` and `outputs/` are git-ignored, and the test suite runs on
synthetic fixtures only.

## 3. Methods

### 3.1 Features (SS-02)

Per-epoch multitaper PSD (`mne.time_frequency.psd_array_multitaper`, 0.5–30 Hz). We integrate power
in five bands — delta 0.5–4, theta 4–8, alpha 8–12, sigma 12–15, beta 15–30 Hz — take logs, and add
three log ratios: delta/theta, theta/alpha and sigma/total. That is 8 features per epoch. Each
feature is z-scored **within subject**, so between-subject amplitude differences (electrode
impedance, skull thickness) do not become "states".

### 3.2 Dimensionality reduction and mixture model (SS-03)

PCA keeps 5 components. A full-covariance Gaussian mixture with `K` components is fitted by EM
written in NumPy: k-means++ seeding, log-sum-exp responsibilities, covariance regularisation
(`1e-6`) and several random restarts keeping the highest likelihood. The mixture supplies the
initial emission parameters for the HMM.

### 3.3 Hidden Markov model (SS-04 to SS-06)

One Gaussian per state (mean + full covariance), with a `K × K` transition matrix and a start
distribution. Everything runs in log space:

- forward–backward for the likelihood and the posteriors `gamma` / `xi`,
- Viterbi for the single most likely state sequence,
- Baum-Welch for training, with one sequence per subject so transitions are never learned across a
  night boundary.

The transition matrix is initialised "sticky" (0.9 on the diagonal): sleep stages persist for
minutes, and this prior keeps EM out of degenerate solutions where states alternate every epoch.

### 3.4 Model selection (SS-08)

`K = 3 … 8`, several restarts each, scored two ways:

- **BIC** with the exact free-parameter count,
  `p = (K-1) + K(K-1) + K·D + K·D(D+1)/2`;
- **held-out log-likelihood** with subjects left out in folds: PCA, GMM and HMM are all refitted on
  the training subjects, and held-out subjects are projected into the training PCA space before
  scoring. No subject influences the space it is scored in.

### 3.5 Matching states to stages (SS-09)

Unsupervised states have no names. We therefore build the state × stage confusion counts and solve
a one-to-one assignment with `scipy.optimize.linear_sum_assignment` (Hungarian). When `K > 5`, some
states cannot get a unique stage; those fall back to a **majority (many-to-one) mapping**, which is
explicitly reported rather than hidden.

Metrics: per-epoch accuracy, Cohen's kappa, adjusted Rand index (clustering agreement, invariant to
relabelling), and per-stage recall/precision with support.

### 3.6 Transition structure and age (SS-10)

From the fitted transition matrix: the heatmap, the stationary distribution (left eigenvector for
eigenvalue 1) and expected dwell times `1/(1-a_ii)`. From the Viterbi paths: empirical run lengths
per state, and per-subject transition rate (transitions per epoch) and mean dwell. Subjects are
split at the median age into younger/older halves and compared with a two-sided Mann-Whitney U test.

### 3.7 Supervised baseline (SS-11)

YASA's pretrained `SleepStaging` is run on the same recordings and the same 30 s grid, and scored
against the expert labels with the same kappa. This is the supervised upper bound: it has seen
thousands of labelled nights, our model has seen none.

## 4. Results

**No measured numbers are committed to this repository.** Data and outputs are git-ignored by
design, so the tables below are the *schema* the pipeline writes — run
`uv run python scripts/run_all.py` and read `outputs/` for the real values. Every figure referenced
here is produced by that run.

| Quantity | File | Columns |
| --- | --- | --- |
| K sweep | `outputs/model_selection.csv` | `k`, `n_parameters`, `train_loglik`, `bic`, `heldout_loglik_per_epoch`, `heldout_std` |
| K sweep figure | `outputs/model_selection.png` | BIC and held-out likelihood vs K |
| Chosen K | `outputs/best_k.json` | `k_by_bic`, `k_by_heldout` |
| Agreement | `outputs/evaluation.json` | `method`, `mapping`, `accuracy`, `kappa`, `adjusted_rand`, `n_epochs` |
| Confusion | `outputs/confusion_matrix.csv`, `.png` | rows = machine states, columns = W/N1/N2/N3/REM |
| Per stage | `outputs/per_stage.csv` | `stage`, `support`, `recall`, `precision` |
| Dwell times | `outputs/dwell_times.csv`, `.png` | `state`, `expected`, `empirical` |
| Transitions | `outputs/transitions.csv`, `transition_matrix.png` | fitted `p(state → state)` |
| Age groups | `outputs/age_summary.csv`, `age_comparison.png` | `group`, `n`, `transition_rate`, `mean_dwell` |
| Baseline | `outputs/baseline.json`, `baseline.csv` | per-subject and pooled `kappa`, `accuracy` |
| Viewer | `outputs/hypnogram.html` | expert vs machine hypnogram, dropdown per subject |

How to read the run:

1. `model_selection.csv` — if BIC and held-out likelihood disagree on K, prefer the held-out curve
   (BIC over-penalises the many parameters of a full-covariance HMM) and say so explicitly.
2. `evaluation.json` — kappa is the headline; accuracy alone is inflated by N2 dominance.
3. `per_stage.csv` — recall per stage shows *which* stage the model cannot find, not just that it
   misses.
4. `confusion_matrix.png` — the row-normalised view makes state collapse obvious (a row that is one
   hot cell means a state that never discriminates).
5. `baseline.json` — the gap between our kappa and YASA's kappa is the value of the labels.

## 5. Where the machine and the experts disagree, and why

These are the failure modes to look for in `confusion_matrix.png`, `per_stage.csv` and
`hypnogram.html`; each is a prediction from how the model is built, to be confirmed or refuted by
the run rather than assumed.

- **N1 is the weak stage.** N1 is low-amplitude, mixed-frequency EEG between wake and N2, with no
  defining spectral landmark. A GMM/HMM has no reason to invent a state for it: N1 epochs are
  usually absorbed into the neighbouring wake or REM state. Expect low N1 recall and a confusion row
  that splits between W and N2/REM.
- **REM versus N1.** Both are low-voltage, mixed-frequency. The classic discriminators are muscle
  tone (EMG) and eye movements (EOG), which we deliberately do not use. Only temporal context
  (REM follows N2/N3 and recurs in ~90 min cycles) lets the HMM separate them, so short REM bouts
  are the most likely place to see state flips.
- **Wake versus N1.** Quiet wake with closed eyes has strong alpha; eyes-open wake does not, and
  then looks like N1. Epochs around sleep onset are where the machine's state boundary will drift
  from the expert's.
- **N2 splits and merges.** Spindles (sigma) and K-complexes are broadband transients. If sigma power
  is weak relative to delta, extra states tend to split N2 rather than resolve it; with `K > 5` the
  Hungarian assignment runs out of stages and the majority mapping merges the surplus states, which
  depresses kappa even when the state structure is informative.
- **Boundary smoothing.** The sticky transition prior is a model of real sleep, but it also smooths
  brief awakenings and short stage changes. Expect the machine hypnogram to look cleaner than the
  expert's, with isolated epochs following their neighbours — better macro structure, worse per-epoch
  accuracy at boundaries.
- **Label imbalance.** N2 is roughly half the night, N3 and N1 are much rarer. Accuracy flatters a
  model that only gets the large stages right; kappa and per-stage recall do not.
- **Night-to-night and subject-to-subject drift.** Features are z-scored per subject, which removes
  offsets but also removes real amplitude information (e.g. a subject with genuinely low delta).
  An unusually fragmented night can therefore be pushed towards a "typical" transition pattern.

## 6. Limitations

- One EEG channel, no EOG/EMG, no artifact channel: the physical signals that define REM and arousal
  are absent.
- Unsupervised states are only as interpretable as the matching step; `K` is chosen by likelihood,
  not by clinical meaning.
- The final model's PCA is fitted on all subjects (the sweep uses fold-local PCA). The sweep's
  held-out numbers are the leakage-free ones.
- Sleep-EDF is a small, relatively healthy, mostly middle-aged-to-older sample; nothing here is a
  claim about clinical populations.
- YASA is a moving target: its pretrained weights and channel expectations differ between versions,
  so the baseline is pinned by `uv.lock` and should be re-run, not quoted, if the lockfile changes.
- No statistical test is applied to the kappa difference between our model and YASA; with 20 nights
  a paired bootstrap would be the honest next step.

## 7. Reproducing

```bash
uv sync
uv run python scripts/run_all.py            # SS-01 through SS-12
uv run pytest -q                            # tests (synthetic fixtures only)
```

Individual steps, if you want them separately:

```bash
uv run python scripts/download.py --subjects 0-19 --night 1
uv run python scripts/extract_features.py
uv run python scripts/select_k.py
uv run python scripts/fit_models.py --n-states 5
uv run python scripts/evaluate.py
uv run python scripts/analyze.py
uv run python scripts/baseline.py
uv run python scripts/viewer.py
```

## Appendix A — EM for the Gaussian mixture

Latent variable `z_n ∈ {1..K}` with `P(z_n = k) = π_k` and `x_n | z_n = k ~ N(μ_k, Σ_k)`. The
observed log-likelihood

`L(θ) = Σ_n log Σ_k π_k N(x_n | μ_k, Σ_k)`

is not directly optimisable, so EM maximises the expected complete-data log-likelihood

`Q(θ, θ') = Σ_n Σ_k r_nk [ log π_k + log N(x_n | μ_k, Σ_k) ]`,

where the E-step posterior is

`r_nk = π_k N(x_n | μ_k, Σ_k) / Σ_j π_j N(x_n | μ_j, Σ_j)`.

In practice we compute `log q_nk = log π_k + log N(x_n | μ_k, Σ_k)` and obtain `r_nk` by a softmax
over `k` (subtract the row max before exponentiating, then renormalise).

The M-step is a constrained maximisation of `Q`. With `N_k = Σ_n r_nk`:

- `π_k = N_k / N` (Lagrange multiplier for `Σ_k π_k = 1`),
- `μ_k = (1/N_k) Σ_n r_nk x_n`,
- `Σ_k = (1/N_k) Σ_n r_nk (x_n - μ_k)(x_n - μ_k)^T + ε I`.

`ε I` keeps every covariance positive definite, so the Cholesky factor used for the density always
exists. Each iteration cannot decrease `L`, because

`L(θ) - L(θ') ≥ Q(θ, θ') - Q(θ', θ') ≥ 0`,

the first inequality being Jensen's and the second the definition of the M-step. Our tests assert
the non-decreasing property directly on the recorded likelihood history.

The Gaussian density itself is evaluated through the Cholesky factor `Σ = L Lᵀ`:
`log N(x) = -½ [ d log 2π + 2 Σ_i log L_ii + ||L⁻¹(x - μ)||² ]`, avoiding any determinant of a
possibly ill-conditioned matrix.

## Appendix B — Forward–backward in log space

With emissions `b_k(x_t) = N(x_t | μ_k, Σ_k)`, transitions `A_ij` and start probabilities `π_k`:

`α_1(k) = π_k b_k(x_1)`, `α_t(j) = [Σ_i α_{t-1}(i) A_ij] b_j(x_t)`,
`β_T(k) = 1`, `β_t(i) = Σ_j A_ij b_j(x_{t+1}) β_{t+1}(j)`.

The sequence likelihood is `P(x_{1:T}) = Σ_k α_T(k)`, and the posteriors are

`γ_t(i) = α_t(i) β_t(i) / P`, `ξ_t(i,j) = α_t(i) A_ij b_j(x_{t+1}) β_{t+1}(j) / P`.

A night is ~1000 epochs, so `α` and `β` underflow in probability space almost immediately. We
therefore keep logs and replace every sum with

`logsumexp(z) = max(z) + log Σ_i exp(z_i - max(z))`,

which is what `scipy.special.logsumexp` does. The recursion becomes

`log α_t(j) = log b_j(x_t) + logsumexp_i [ log α_{t-1}(i) + log A_ij ]`,

with zeros in `A` mapped to `-inf` (they contribute `exp(-inf) = 0`). `log β` follows symmetrically.
The tests compare the result against explicit enumeration of all `K^T` state sequences for a tiny
model, and against `hmmlearn` to 1e-6.

## Appendix C — Viterbi

Viterbi is the same recursion in the max-plus semiring: replace `Σ` with `max` and keep a
backpointer.

`δ_1(k) = log π_k + log b_k(x_1)`

`δ_t(j) = log b_j(x_t) + max_i [ δ_{t-1}(i) + log A_ij ]`, `ψ_t(j) = argmax_i [ ... ]`.

The best final state is `argmax_k δ_T(k)`; walking `ψ` backwards yields the path, and `max_k δ_T(k)`
is its joint log probability. The difference from forward–backward is exactly sum-product versus
max-product: forward–backward marginalises over paths, Viterbi takes the single best one. The tests
enumerate all paths and require both the path and its score to match, and `hmmlearn.decode` to
agree.

## Appendix D — Baum-Welch

Baum-Welch is EM for the HMM, with the state sequence as the latent variable. The E-step is
forward–backward, giving `γ` and `ξ`. The M-step maximises the expected complete-data
log-likelihood, which for one sequence gives

- `π̂_k = γ_1(k)`,
- `Â_ij = Σ_{t=1}^{T-1} ξ_t(i,j) / Σ_{t=1}^{T-1} γ_t(i)`,
- `μ̂_k = Σ_t γ_t(k) x_t / Σ_t γ_t(k)`,
- `Σ̂_k = Σ_t γ_t(k) (x_t - μ̂_k)(x_t - μ̂_k)^T / Σ_t γ_t(k) + ε I`.

With one sequence per subject (`S` sequences), the numerators and denominators are summed over all
sequences, and the start-probability counts are summed over each sequence's first epoch and divided
by `S` — transitions are never counted across a night boundary. `Â` rows that receive no expected
transitions fall back to uniform rather than `NaN`.

Initialisation matters more than the update formulas: `π` and the emissions come from the fitted GMM,
and `A` starts sticky (`0.9` diagonal). Each Baum-Welch iteration is an EM iteration, so the
likelihood is non-decreasing; the tests check that, check parameter recovery on data simulated from a
known three-state model, and check a full EM step against `hmmlearn` to 1e-6 with priors and jitter
zeroed on both sides.
