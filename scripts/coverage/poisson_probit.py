#!/usr/bin/env python3
"""
Poisson DGP of Fortini et al. (2026b): coverage of the CLT and the probit CLT near the boundary.

Thesis Sec. 4.4.3 (Tab. 4.4, 4.5: 95%) and App. A.7 (Tab. A.3: paired comparison; Tab. A.4, A.5: 80%).
Setup of Fortini et al. (2026b), App. J.5 (code: github.com/weiyaw/ud4pfn, experiments/coverage):
    X ~ scrambled Sobol on [-1, 1]^10, fresh scramble per repetition
    lambda(x) = exp(0.5 w0'x),  w0 = (-1, 2, -3, ..., 10) / sqrt(385)
    Y = (C - lambda) / sqrt(lambda),  C ~ Poisson(lambda)
    target f0(x) = P(Y <= 2 | x) on m = 100 Sobol grid points (their seed 50194), n_estimators = 16
Both methods use the same TabPFN trajectory and V_n:
    CLT         F_n +- c sqrt(V_n / n), clipped to [0, 1]
    Probit CLT  Phi(Phi^-1(F_n) +- c sqrt(V_n / n) / phi(Phi^-1(F_n)))
with c = z_{1-alpha/2} (pointwise) or the sup-t critical value (simultaneous, Fortini et al. App. C).

Outputs in results/artifacts/poisson_probit/:
    coverage.csv   per level, band, n and method: coverage (SE), mean width and the misses by side; pointwise
                   the share of grid points (mean over repetitions), simultaneous the number of repetitions
    paired.csv     simultaneous bands: repetitions in which both cover, only one fails, or both fail
Cache: one TabPFN trajectory per (n, repetition).

Usage
    python scripts/coverage/poisson_probit.py               # n = 50, 100, 200, R = 50
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm, poisson, qmc

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [p for p in (str(ROOT), str(ROOT / "src")) if p not in sys.path]

from scripts.common import RunCache, add_run_args, add_sample_args, n_estimators, parse_rep_key, quiet_warnings, rep_key, set_threads  # noqa: E402,E501
from uqpfn import asymptotic  # noqa: E402

quiet_warnings()

SEED = 1000
D = 10
T_THRESH = 2.0
GRID_SEED = 50194
GRID_SIZE = 100
N_ESTIMATORS = n_estimators(16)
ALPHAS = (0.05, 0.2)                # 95% (Sec. 4.4.3) and 80% (App. A.7)
SUPT_DRAWS = 10_000
BANDS = ("pointwise", "simultaneous")
METHODS = ("CLT", "Probit CLT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poisson DGP: CLT vs probit CLT coverage.")
    add_run_args(parser)
    add_sample_args(parser, n_list=(50, 100, 200), n_reps=50)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# DGP (Fortini et al. 2026b, PoissonLinearMultivariate)
# ---------------------------------------------------------------------------
_i = np.arange(1, D + 1, dtype=float)
W0 = (-1.0) ** _i * _i / np.linalg.norm(_i)     # (-1, 2, -3, ..., 10) / sqrt(385)


def _rate(X: np.ndarray) -> np.ndarray:
    return np.clip(np.exp(0.5 * (X @ W0)), 1e-3, None)


def true_cdf(X: np.ndarray) -> np.ndarray:
    """f0(x) = P(Y <= 2 | x), a Poisson CDF depending on x only through w0'x."""
    lam = _rate(X)
    return poisson.cdf(np.floor(lam + T_THRESH * np.sqrt(lam)), lam)


def _sobol(n: int, seed: int) -> np.ndarray:
    try:
        sampler = qmc.Sobol(d=D, scramble=True, rng=np.random.default_rng(seed))
    except TypeError:  # scipy < 1.15
        sampler = qmc.Sobol(d=D, scramble=True, seed=np.random.default_rng(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # n not a power of 2
        return 2.0 * sampler.random(n) - 1.0


def sample_data(n: int, rng: np.random.Generator):
    X = _sobol(n, int(rng.integers(2**31 - 1)))
    lam = _rate(X)
    return X, (rng.poisson(lam) - lam) / np.sqrt(lam)


def _cache(args) -> RunCache:
    config = {"seed": SEED, "grid_size": GRID_SIZE, "n_estimators": N_ESTIMATORS}
    return RunCache(args.outdir / "poisson_probit", config, version=None)


# ---------------------------------------------------------------------------
# Fit: one trajectory per (n, repetition)
# ---------------------------------------------------------------------------
def run_fit(args) -> None:
    set_threads(args.threads)
    from uqpfn import tabpfn

    cache = _cache(args)
    cache.prepare(args.overwrite)
    X_eval = _sobol(GRID_SIZE, GRID_SEED)
    truth = true_cdf(X_eval)
    todo = [(rep, n) for rep in range(args.n_reps) for n in args.n_train_list if not cache.has(rep_key(n, rep))]
    print(f"[fit] {len(todo)} runs to do in {cache.folder}; f0 on the grid in [{truth.min():.3f}, {truth.max():.3f}]")

    for i, (rep, n) in enumerate(todo, 1):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, n, rep]))
        X, y = sample_data(n, rng)
        perm = rng.permutation(n)                      # V_n depends on the order of the context
        t0 = time.perf_counter()
        traj = tabpfn.build_predictive_trajectory(
            X[perm], y[perm], X_eval, np.array([T_THRESH]), n_estimators=N_ESTIMATORS
        )[:, 0, :]                                     # (n, m): F_k(x_j, 2), k = 1..n
        cache.save(rep_key(n, rep), X=X, y=y, perm=perm, traj=traj, truth=truth)
        print(f"[fit {i}/{len(todo)}] n={n} rep={rep}: {time.perf_counter() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Plot: tables from the cache
# ---------------------------------------------------------------------------
def supt_critical(A: np.ndarray, s: np.ndarray, alpha: float, n_draws: int, rng) -> float:
    """sup-t critical value for N(0, A'A). Draws W = xi A (no Cholesky: A'A is singular for n - 1 < m)."""
    ok = s > 0
    W = rng.standard_normal((n_draws, A.shape[0])) @ A[:, ok]
    return float(np.quantile(np.max(np.abs(W) / s[ok], axis=1), 1 - alpha))


def rep_bands(traj, n, rep, alphas, n_draws, seed) -> dict:
    """{(alpha, band, method): (lo, hi)} for one repetition, plus F_n under key 'F'."""
    F = traj[-1]
    v_n = asymptotic.compute_vn_pointwise(traj)                     # (1/n) sum_k k^2 Delta_k^2
    s = np.sqrt(v_n / n)
    A = np.arange(2, n + 1)[:, None] * np.diff(traj, axis=0) / n     # A'A = V_n / n (full matrix)
    eta, s_eta = asymptotic.probit_moment_match(F, v_n, n, recenter=False)
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, rep, 1]))
    out = {"F": F}
    for a in alphas:
        for band, c in (("pointwise", norm.ppf(1 - a / 2)), ("simultaneous", supt_critical(A, s, a, n_draws, rng))):
            out[(a, band, "CLT")] = (np.clip(F - c * s, 0, 1), np.clip(F + c * s, 0, 1))
            out[(a, band, "Probit CLT")] = (norm.cdf(eta - c * s_eta), norm.cdf(eta + c * s_eta))
    return out


def run_plot(args) -> None:
    import pandas as pd

    cache = _cache(args)
    rows = []
    for key in cache.require():
        n, rep, _ = parse_rep_key(key)
        z = cache.load(key)
        truth = z["truth"]
        bands = rep_bands(z["traj"].astype(float), n, rep, ALPHAS, SUPT_DRAWS, SEED)
        for (a, band, meth), (lo, hi) in ((k, v) for k, v in bands.items() if k != "F"):
            cov = (truth >= lo) & (truth <= hi)
            below, above = truth < lo, truth > hi      # below: F_n overestimates f0
            simultaneous = band == "simultaneous"
            rows.append({
                "alpha": a, "band": band, "n": n, "rep": rep, "method": meth,
                "coverage": float(cov.all()) if simultaneous else cov.mean(),
                "width": np.mean(hi - lo),
                "miss_below": float(below.any()) if simultaneous else below.mean(),
                "miss_above": float(above.any()) if simultaneous else above.mean(),
            })
    res = pd.DataFrame(rows)

    def se(v):
        return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan

    summary = (res.groupby(["alpha", "band", "n", "method"])
               .agg(coverage=("coverage", "mean"), se_coverage=("coverage", se), width=("width", "mean"),
                    miss_below=("miss_below", "mean"), miss_above=("miss_above", "mean"), n_reps=("rep", "nunique"))
               .reset_index())
    simultaneous = summary["band"] == "simultaneous"
    for col in ("miss_below", "miss_above"):            # simultaneous: number of repetitions
        summary.loc[simultaneous, col] = (summary.loc[simultaneous, col] * summary.loc[simultaneous, "n_reps"]).round()
    summary.to_csv(cache.folder / "coverage.csv", index=False, float_format="%.6f")

    sim = res[res["band"] == "simultaneous"].pivot_table(index=["alpha", "n", "rep"], columns="method",
                                                         values="coverage")
    clt_ok, probit_ok = sim["CLT"] == 1, sim["Probit CLT"] == 1
    paired = pd.DataFrame({"both_cover": clt_ok & probit_ok, "only_clt_fails": ~clt_ok & probit_ok,
                           "only_probit_fails": clt_ok & ~probit_ok, "both_fail": ~clt_ok & ~probit_ok})
    paired = paired.groupby(level=["alpha", "n"]).sum().reset_index()
    paired.to_csv(cache.folder / "paired.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", None, "display.precision", 3):
        print(summary.to_string(index=False))
        print(paired.to_string(index=False))
    print(f"[table] {cache.folder / 'coverage.csv'}, {cache.folder / 'paired.csv'}")


def main() -> None:
    args = parse_args()
    if args.run in ("fit", "all"):
        run_fit(args)
    if args.run in ("plot", "all"):
        run_plot(args)


if __name__ == "__main__":
    main()
