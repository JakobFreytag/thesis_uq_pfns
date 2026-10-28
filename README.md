# Uncertainty Quantification for Prior-Data Fitted Networks

Code for the Master's thesis *Uncertainty Quantification for Prior-Data Fitted Networks*.

The thesis studies two ways of getting epistemic uncertainty out of TabPFN within the predictive
view of Bayesian inference:

- **Approximate martingale posteriors (AMP)**, Nagler and Rügamer (2025): TabPFN's predictive is
  propagated with copula updates that are martingales by construction.
- **The predictive CLT**, Fortini et al. (2026a, b): TabPFN's own predictive sequence is kept and
  credible intervals come from an estimate $V_n$ of the asymptotic variance.

On top of these, the repo contains a **probit-scale CLT** (intervals that stay in $[0, 1]$ by
construction), diagnostics of the quasi-martingale conditions on TabPFN's predictive sequence,
coverage experiments, and the **entropic uncertainty decomposition** (aleatoric vs. epistemic) for
binary and multiclass classification.

All experiments use **TabPFN 2.0.5** (TabPFN v2). The code is inspired by the AMP implementation of
Nagler and Rügamer (2025).

---

## Repository layout

```
src/uqpfn/                       the method code
  martingale.py                  AMP: model-agnostic martingale-posterior update (get_posterior),
                                 alpha schedules, interval extraction
  asymptotic.py                  predictive CLT: V_n, pointwise CDF bounds, probit CLT,
                                 entropy decompositions (Beta, probit, Dirichlet, multiclass probit)
  tabpfn.py                      TabPFN helpers: analytic CDF of the binned predictive, predictive
                                 and classifier trajectories on expanding prefixes

scripts/                         one script per experiment (see the table below)
  common.py                      shared by all scripts: fit / plot run cache, CLI flags, thesis plot style
  quasi_martingale.py            TabPFN rollouts for the quasi-martingale diagnostics
  coverage/                      regression experiments (ensemble size, coverage, probit CLT)
    coverage_utils.py            linear, dependent and spline DGPs, exact spline posterior, CDF helpers
  entropy_ud/                    classification experiments (entropic uncertainty decomposition)
    entropy_utils.py             TabPFN trajectory, (g_n, V_n), binary Beta / probit decomposition

reproduce.sh                     runs the experiment scripts: the tables and figures of the thesis
tabpfn_diagnostics.ipynb         figures for the quasi-martingale diagnostics, from the saved rollouts
results/
  tabpfn_diagnostics_rollouts/   the 20 rollouts used in the thesis (raw CSVs, r1 ... r20)
  artifacts/                     default output folder of all scripts (figures, tables, caches);
                                 git-ignored, regenerate with the scripts
```

## Setup

Python >= 3.12. The dependencies are pinned in `pyproject.toml` and locked in `uv.lock`
(the `dev` group adds `ipykernel` for the notebook).

```bash
uv sync
source .venv/bin/activate
```

`uqpfn` is not installed as a package; the scripts put `src/` on the path themselves.

All scripts are run **from the repository root**: output paths such as `results/artifacts/...` are
relative to the working directory. Every script has `--help`.

## Thesis ↔ code

Each script writes the tables and figures of its thesis section to `results/artifacts/<script>/` (spline:
`spline_<mode>/`); the settings of the thesis are fixed in the scripts.

| Thesis | Script | Outputs |
|---|---|---|
| Sec. 4.1, App. A.2, A.4, A.5 — quasi-martingale diagnostics | `quasi_martingale.py` (rollouts), then `tabpfn_diagnostics.ipynb` (figures) | rollouts in `results/tabpfn_diagnostics_rollouts/`, figures in `tabpfn_diagnostics/` |
| Sec. 4.3: Tab. 4.1, Fig. 4.5 — $V_n$ vs `n_estimators`, linear DGP | `coverage/n_estimators_linear.py` | `coverage.csv`, `rows_n200_row*` |
| Sec. 4.3: Tab. 4.1, App. A.6: Fig. A.5 — same, balanced spline DGP | `coverage/n_estimators_spline.py` | `coverage.csv`, `rows_n200_row*` |
| Sec. 4.4.1: Tab. 4.2, Fig. 4.6, App. A.7: Fig. A.6 — AMP vs CLT, linear and dependent DGP | `coverage/linear_dependent.py` | `coverage.csv`, `linear_n200`, `dependent_n200_detail` |
| Sec. 4.4.2: Tab. 4.3, Fig. 4.7, 4.8, App. A.7: Tab. A.1 — AMP vs CLT vs optimal, unbalanced spline DGP | `coverage/spline.py` | `coverage.csv`, `methods_n200`, `overview_n200` |
| App. A.7: Tab. A.2 — same, balanced design | `coverage/spline.py --mode balanced` | `coverage.csv` |
| Sec. 4.4.3: Tab. 4.4, 4.5, App. A.7: Tab. A.3–A.5 — CLT vs probit CLT, Poisson DGP | `coverage/poisson_probit.py` | `coverage.csv`, `paired.csv` |
| Sec. 4.5.1, App. A.8: Fig. A.7 — logistic replication | `entropy_ud/replication_logistic.py` | `grid.png` |
| Sec. 4.5.1: Fig. 4.9, App. A.8: Fig. A.8 — two-moons replication | `entropy_ud/replication_moons.py` | `overview_moons1.png`, `overview_moons2.png` |
| Sec. 4.5.2: Fig. 4.10 — frequency setting | `entropy_ud/frequency.py` | `frequency.png`, `frequency.csv` |
| Sec. 4.5.2: Fig. 4.11, App. A.8: Tab. A.6 — Bayesian logistic benchmark | `entropy_ud/logistic_optimum.py` | `ue_examples_shape`, `ue_examples_tabpfn`, `relative_error.csv` |
| Sec. 4.5.3: Fig. 4.12, 4.13 — spirals replication | `entropy_ud/replication_spirals.py` | `replication.png`, `dirichlet_sign.png` |
| Sec. 4.5.4: Fig. 4.14–4.17 — Gaussian-cluster benchmark | `entropy_ud/multiclass_blops.py` | `ue_exact.png`, `transect.png`, `sign_check.png`, `ue_tabpfn.png` |

All scripts live in `scripts/`. The data-generating processes and settings are listed in App. A.1 of the
thesis, implementation details (response grid, degenerate prefixes, sup-t bands) in App. A.3.

## Running the experiments

To reproduce everything:

```bash
./reproduce.sh          # fit what is not cached yet, then write all tables and figures (takes long)
./reproduce.sh plot     # only rebuild the tables and figures from the existing caches
./reproduce.sh quick    # test run of the whole pipeline with n_estimators = 1, into results/artifacts/quick/
```

The quasi-martingale diagnostics are not part of `reproduce.sh`: the rollouts are committed, the figures come
from `tabpfn_diagnostics.ipynb`, and recomputing the rollouts with `scripts/quasi_martingale.py` is by far the
most expensive step.

The ensemble size can also be set for a single script with `UQPFN_N_ESTIMATORS=1 python scripts/...`
(use a separate `--outdir`).

All scripts are run from the repository root and work the same way. The TabPFN fits are the expensive
part, so each script separates *fit* from *plot* (implemented once, in `scripts/common.py`):

```
results/artifacts/<script>/
  config.json               the settings of the fit; a cache made with other settings is refused
  cache/n0200/rep000.npz    one file per unit of work, usually one (sample size, repetition),
                            written atomically, so an interrupted run never leaves a broken file
  *.csv, *.pdf, *.png       the thesis tables and figures, rebuilt from the whole cache
```

- `--run fit` computes only the missing cache files, so runs can be stopped and resumed, and a larger
  `--n-reps` only computes the new repetitions.
- `--run plot` rebuilds the tables and figures from the cache, without TabPFN.
- `--run all` (default) does both; `--overwrite` deletes the cache before fitting.
- `--threads` sets the torch CPU threads, `--outdir` the parent folder (default `results/artifacts`).
- Scripts with repetitions also take `--n-train-list` and `--n-reps`, the regression scripts `--plot-n` (sample size of
  the figures).

```bash
python scripts/coverage/spline.py               # fit n = 50, 100, 200 with R = 20, then tables and figures
python scripts/coverage/spline.py --run plot    # only redraw from the cache
```

The quasi-martingale script has no plot step: its cache are the rollout CSVs in
`results/tabpfn_diagnostics_rollouts/` (one per rollout, existing ones are skipped), which the notebook reads.

## TabPFN license

The experiments use TabPFN 2.0.5 (Hollmann et al., 2025), whose model weights are downloaded on first use. The
weights are licensed under the Prior Labs License (Apache 2.0 with an additional attribution requirement), see
[PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN). Built with TabPFN.

## References

- Fortini, S., Ng, K., Petrone, S., Rousseau, J. and Wei, S. (2026a). A principled framework for
  uncertainty decomposition in TabPFN. [arXiv:2602.04596v1](https://arxiv.org/abs/2602.04596v1).
- Fortini, S., Ng, K., Petrone, S., Rousseau, J. and Wei, S. (2026b). Uncertainty decomposition for
  Bayes-filtered transformers via Bayesian predictive inference. [arXiv:2602.04596v2](https://arxiv.org/abs/2602.04596v2).
- Hollmann, N. et al. (2025). Accurate predictions on small data with a tabular foundation model.
  *Nature* 637: 319–326.
- Nagler, T. and Rügamer, D. (2025). Uncertainty quantification for prior-data fitted networks using
  martingale posteriors. [arXiv:2505.11325](https://arxiv.org/abs/2505.11325).

Full references are in the thesis.
