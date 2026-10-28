#!/usr/bin/env python3
"""
Entropic UD on Bayesian logistic regression: the Beta and the recentred probit against the Bayes-optimal
benchmark, with exact inputs and with TabPFN's inputs.

Thesis Sec. 4.5.2 (Bayesian logistic, Fig. 4.11) and App. A.8 (Tab. A.6).
DGP (one dataset per repetition, seed = 2026 + rep):
  (b0, b1) ~ N(0, tau^2 I) with tau = 2,  x ~ N(0, 1),  y ~ Bernoulli(sigmoid(b0 + b1 x))
  the dataset of size n is the first n rows of one draw of size max(n), so all n are read off ONE TabPFN
  trajectory; test points x* on [-4, 4] in steps of 0.1
Benchmark: the exact posterior of (b0, b1) under the same prior on a 400 x 400 grid over [-5 tau, 5 tau]^2
(checked against 800 x 800 on the first three datasets). With G = sigmoid(b0 + b1 x*):
  ref_g = E[G],  ref_var = Var[G],  U_a = E[h(G)],  U_e = h(ref_g) - U_a
Inputs to both approximations: exact (ref_g, ref_var) and TabPFN's (g_n, V_n / n).

Outputs in results/artifacts/logistic_optimum/:
    ue_examples_shape, ue_examples_tabpfn (.pdf, .png)   U_e over x* for datasets 0, 1, 2 at n = 20 with exact
                                                        and with TabPFN's inputs (Fig. 4.11 top, bottom)
    relative_error.csv                                  Tab. A.6: mean absolute relative error of U_e in %,
                                                        over the points of a region (|x*| <= 2: in, >= 2.5:
                                                        out, U_e^ref >= 1e-3), then over datasets, and the
                                                        number of datasets in which the Beta is better
Cache: one file per dataset (data, benchmark moments for every n, TabPFN trajectory).

Usage
    python scripts/entropy_ud/logistic_optimum.py               # n = 20, 50, 100, R = 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, log_expit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [p for p in (str(ROOT), str(ROOT / "src")) if p not in sys.path]

from scripts.common import RunCache, add_run_args, add_sample_args, parse_rep_key, quiet_warnings, rep_key, seed_all, set_threads  # noqa: E402,E501
from scripts.entropy_ud.entropy_utils import METHODS, N_ESTIMATORS, decompose, gn_vn  # noqa: E402
from uqpfn import asymptotic  # noqa: E402

quiet_warnings()

SEED = 2026
TAU = 2.0                           # prior sd of (b0, b1)
GRID = (-4.0, 4.0, 0.1)             # test points x*: lo, hi, step
THETA_GRID = 400                    # K for the K x K posterior grid over [-5 tau, 5 tau]^2
CHECK_GRID, CHECK_REPS, GRID_TOL = 800, 3, 0.01     # benchmark accuracy check
MIN_REF_UE = 1e-3                   # points with U_e^ref below this are left out of the relative errors
REGION_IN, REGION_OUT = 2.0, 2.5    # |x*| <= 2: in the data, |x*| >= 2.5: extrapolation
EXAMPLE_REPS, EXAMPLE_SIZES = (0, 1, 2), (20,)
CACHE_VERSION = 2

METHOD_LABELS = {"beta": "Beta", "probit_recentred": "Probit (recentred)"}
METHOD_COLORS = {"beta": "C0", "probit_recentred": "C2"}
REFERENCE_LABEL = "Reference"
INPUTS = {"shape": "exact", "tabpfn": "TabPFN"}
SETTING_TITLES = {"shape": "Shape only: reference mean and variance",
                  "tabpfn": "TabPFN: $g_n$ and $V_n/n$"}
FIG_W = 6.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entropic UD on Bayesian logistic regression vs the exact benchmark.")
    add_run_args(parser)
    add_sample_args(parser, n_list=(20, 50, 100), n_reps=5)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data and benchmark
# ---------------------------------------------------------------------------
def make_x_eval(lo, hi, step):
    # rounded so that the region bounds (|x*| = 2, 2.5) are hit exactly
    return np.round(np.arange(lo, hi + step / 2, step), 10)


def make_data(n, tau, rng):
    """Unchanged from the previous version, so datasets are identical for a given seed."""
    beta = tau * rng.standard_normal(2)
    x = rng.normal(0.0, 1.0, size=n)
    y = (rng.random(n) < expit(beta[0] + beta[1] * x)).astype(int)
    return x.reshape(-1, 1), y, beta


def region_of(x_star, r_in, r_out):
    a = np.abs(np.asarray(x_star, dtype=float))
    return np.where(a <= r_in + 1e-9, "in", np.where(a >= r_out - 1e-9, "out", "mid"))


class ReferenceGrid:
    """
    Grid posterior of (b0, b1) under N(0, tau^2 I) on a K x K grid over
    [-5 tau, 5 tau]^2, and the moments of G = sigmoid(b0 + b1 x*) under it.

    eta = b0 + b1 x* does not depend on the data, so with ``tables=True`` the
    (n_eval, K^2) arrays sigmoid(eta) and h(sigmoid(eta)) are built once and
    every reference is a matrix-vector product. ``tables=False`` (the finer
    accuracy-check grid) recomputes them per chunk of x* instead of holding them.
    """

    def __init__(self, tau, K, x_eval, tables=True, chunk=16):
        self.x_eval = np.asarray(x_eval, dtype=float)
        b = np.linspace(-5.0 * tau, 5.0 * tau, K)
        B0, B1 = np.meshgrid(b, b, indexing="ij")
        self.b0, self.b1 = B0.ravel(), B1.ravel()
        self.log_prior = -(self.b0 ** 2 + self.b1 ** 2) / (2.0 * tau ** 2)
        self.chunk = chunk
        self._tables = None
        if tables:
            P, H = zip(*(self._rows(j, min(j + chunk, len(self.x_eval)))
                         for j in range(0, len(self.x_eval), chunk)))
            self._tables = (np.vstack(P), np.vstack(H))

    def _rows(self, j0, j1):
        """sigmoid(eta) and h(sigmoid(eta)) for x*[j0:j1], via log-sigmoids (exact in the tails)."""
        eta = self.b0[None, :] + self.b1[None, :] * self.x_eval[j0:j1, None]
        p, q = expit(eta), expit(-eta)
        h = -(p * log_expit(eta) + q * log_expit(-eta))
        return p, h

    def weights(self, x, y, sizes):
        """Normalised posterior weights for the nested datasets D_n = first n rows."""
        x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y).ravel()
        logpost = self.log_prior.copy()
        out, k = {}, 0
        for n in sorted(sizes):
            for i in range(k, n):
                eta = self.b0 + self.b1 * x[i]
                # log Bern(y | sigmoid(eta)) = y eta - log(1 + e^eta)
                logpost += y[i] * eta - np.logaddexp(0.0, eta)
            k = n
            w = np.exp(logpost - logpost.max())
            out[n] = w / w.sum()
        return out

    def moments(self, W):
        """
        W: {n: weights}. Returns {n: dict(mean, var, ua, ue)}, each (n_eval,).
        Var is the two-pass sum w (p - ref_g)^2, i.e. sum w p^2 - ref_g^2
        without the cancellation.
        """
        keys = list(W)
        Wm = np.column_stack([W[k] for k in keys])                     # (K^2, S)
        n_eval, S = len(self.x_eval), len(keys)
        g, var, ua = (np.empty((n_eval, S)) for _ in range(3))
        for j0 in range(0, n_eval, self.chunk):
            j1 = min(j0 + self.chunk, n_eval)
            if self._tables is not None:
                P, H = self._tables[0][j0:j1], self._tables[1][j0:j1]
            else:
                P, H = self._rows(j0, j1)
            g[j0:j1] = P @ Wm
            ua[j0:j1] = H @ Wm
            for s in range(S):
                var[j0:j1, s] = ((P - g[j0:j1, s, None]) ** 2) @ Wm[:, s]
        out = {}
        for s, k in enumerate(keys):
            ue = asymptotic.binary_entropy(g[:, s]) - ua[:, s]
            out[k] = {"mean": g[:, s], "var": var[:, s], "ua": ua[:, s], "ue": ue}
        return out


def check_grid_accuracy(x, y, sizes, ref, x_eval, rep):
    """Recompute the benchmark on a finer grid; fail if ref_var or ref_ue moves by more than GRID_TOL."""
    fine = ReferenceGrid(TAU, CHECK_GRID, x_eval, tables=False)
    ref_f = fine.moments(fine.weights(x, y, sizes))
    worst = 0.0
    for n in sizes:
        a, b = ref[n], ref_f[n]
        mask = b["ue"] >= MIN_REF_UE        # the points that enter the relative errors
        if not mask.any():
            mask = np.ones_like(mask)
        dv = np.max(np.abs(a["var"][mask] - b["var"][mask]) / b["var"][mask])
        du = np.max(np.abs(a["ue"][mask] - b["ue"][mask]) / b["ue"][mask])
        worst = max(worst, dv, du)
        if max(dv, du) > GRID_TOL:
            raise RuntimeError(f"benchmark grid too coarse (rep {rep}, n={n}): ref_var changes by {dv:.2%}, "
                               f"ref_ue by {du:.2%} going to K={CHECK_GRID}")
    print(f"  grid check, rep {rep}: K {THETA_GRID} -> {CHECK_GRID} changes ref_var and ref_ue by <= {worst:.0e}")


def _cache(args) -> RunCache:
    config = {"seed": SEED, "tau": TAU, "grid": list(GRID), "theta_grid": THETA_GRID, "n_estimators": N_ESTIMATORS}
    return RunCache(args.outdir / "logistic_optimum", config, version=CACHE_VERSION)


# ---------------------------------------------------------------------------
# Fit: one cached file per dataset
# ---------------------------------------------------------------------------
def run_fit(args) -> None:
    set_threads(args.threads)
    seed_all(SEED)
    cache = _cache(args)
    cache.prepare(args.overwrite)
    x_eval = make_x_eval(*GRID)
    sizes, n_max = sorted(args.n_train_list), max(args.n_train_list)
    todo = [rep for rep in range(args.n_reps) if not cache.has(rep_key(n_max, rep))]
    print(f"[fit] {len(todo)} datasets to do in {cache.folder}")
    if not todo:
        return

    from uqpfn import tabpfn
    grid = ReferenceGrid(TAU, THETA_GRID, x_eval)
    model = tabpfn.make_classifier(n_estimators=N_ESTIMATORS)
    for rep in todo:
        t0 = time.time()
        rng = np.random.default_rng(SEED + rep)
        x, y, beta = make_data(n_max, TAU, rng)
        ref = grid.moments(grid.weights(x, y, sizes))
        if rep < CHECK_REPS:
            check_grid_accuracy(x, y, sizes, ref, x_eval, rep)
        # one trajectory on the rows in their generated order: they are i.i.d., so this order is already a
        # random permutation, and the first n rows are a trajectory for D_n
        traj = tabpfn.build_classifier_trajectory(x, y, x_eval.reshape(-1, 1), n_estimators=N_ESTIMATORS,
                                                  model=model)
        arrays = {"x": x, "y": y, "beta": beta, "sizes": np.array(sizes), "traj": traj}
        for k in ("mean", "var", "ua", "ue"):
            arrays[f"ref_{k}"] = np.stack([ref[n][k] for n in sizes])          # (len(sizes), n_eval)
        cache.save(rep_key(n_max, rep), **arrays)
        print(f"[fit] rep {rep}: b0={beta[0]:+.2f}, b1={beta[1]:+.2f}  ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# Plot: Tab. A.6 and Fig. 4.11 from the cache
# ---------------------------------------------------------------------------
def _point_frame(cache: RunCache) -> tuple[pd.DataFrame, pd.DataFrame]:
    """U_e of the benchmark and of both methods at every (dataset, n, x*, inputs), and the datasets."""
    x_eval = make_x_eval(*GRID)
    frames, datasets = [], []
    for key in cache.require():
        rep = parse_rep_key(key)[1]
        z = cache.load(key)
        for i, n in enumerate(int(n) for n in z["sizes"]):
            g, v = gn_vn(z["traj"], n)
            inputs = {"shape": (z["ref_mean"][i], z["ref_var"][i]), "tabpfn": (g, v / n)}
            for evaluation, (m, s2) in inputs.items():
                for method in METHODS:
                    d = decompose(m, s2, 1, method)         # n = 1: s2 is the variance itself
                    frames.append(pd.DataFrame({"rep": rep, "n": n, "x_star": x_eval, "evaluation": evaluation,
                                                "method": method, "ref_ue": z["ref_ue"][i],
                                                "ue_hat": d["epistemic_raw"]}))
        datasets.append(pd.DataFrame({"rep": rep, "b0": z["beta"][0], "b1": z["beta"][1],
                                      "row": np.arange(len(z["y"])), "x": z["x"].ravel(), "y": z["y"]}))
    return pd.concat(frames, ignore_index=True), pd.concat(datasets, ignore_index=True)


def relative_error_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tab. A.6: mean |U_e_hat - U_e^ref| / U_e^ref in %, per region over points, then over datasets."""
    a = np.abs(df["x_star"])
    df = df.assign(region=np.where(a <= REGION_IN + 1e-9, "in", np.where(a >= REGION_OUT - 1e-9, "out", "mid")))
    df = df[(df["region"] != "mid") & (df["ref_ue"] >= MIN_REF_UE)]
    df = df.assign(abs_rel_err=100 * np.abs(df["ue_hat"] - df["ref_ue"]) / df["ref_ue"])
    per_dataset = df.groupby(["evaluation", "n", "region", "method", "rep"])["abs_rel_err"].mean().unstack("method")
    rows = []
    for (evaluation, n, region), d in per_dataset.groupby(level=["evaluation", "n", "region"]):
        rows.append({"inputs": INPUTS[evaluation], "n": n, "region": region,
                     "beta": d["beta"].mean(), "probit_recentred": d["probit_recentred"].mean(),
                     "beta_better": int((d["beta"] < d["probit_recentred"]).sum()), "n_datasets": len(d)})
    out = pd.DataFrame(rows)
    out["_i"] = out["inputs"].map({v: i for i, v in enumerate(INPUTS.values())})
    return out.sort_values(["_i", "region", "n"]).drop(columns="_i").reset_index(drop=True)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "legend.fontsize": 10})
    return plt


def _save(fig, run_dir, name):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(run_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {run_dir / name}.png/.pdf")


def _ue_lines(ax, sub, methods):
    """Reference U_e (black) and each family's U_e_hat over x* for one slice of results."""
    sub = sub.sort_values("x_star")
    ref = sub[sub["method"] == methods[0]]
    ax.plot(ref["x_star"], ref["ref_ue"], "k", lw=1.8, label=REFERENCE_LABEL)
    for m in methods:
        s = sub[sub["method"] == m]
        ax.plot(s["x_star"], s["ue_hat"], color=METHOD_COLORS[m], lw=1.3,
                label=METHOD_LABELS[m])


def plot_examples(df, ds, evaluation, methods, run_dir):
    """A few datasets (columns) at a few n (rows), with the training x as a rug."""
    plt = _plt()
    reps = [r for r in EXAMPLE_REPS if r in set(df["rep"])]
    ns = [n for n in EXAMPLE_SIZES if n in set(df["n"])]
    e = df[df["evaluation"] == evaluation]
    fig, axes = plt.subplots(len(ns), len(reps), figsize=(FIG_W, 2.1 * len(ns) + 0.6),
                             sharex=True, sharey="row", squeeze=False)
    for j, rep in enumerate(reps):
        data = ds[ds["rep"] == rep].sort_values("row")
        for i, n in enumerate(ns):
            ax = axes[i, j]
            _ue_lines(ax, e[(e["rep"] == rep) & (e["n"] == n)], methods)
            x, y = data["x"].to_numpy()[:n], data["y"].to_numpy()[:n]
            ax.plot(x[y == 0], np.zeros((y == 0).sum()), "|", color="C4", ms=8)
            ax.plot(x[y == 1], np.zeros((y == 1).sum()), "|", color="C1", ms=8)
            ax.set_xlim(GRID[0], GRID[1])
            if i == 0:
                ax.set_title(f"$\\beta_0$={data['b0'].iloc[0]:.2f}, "
                             f"$\\beta_1$={data['b1'].iloc[0]:.2f}")
            if j == 0:
                ax.set_ylabel(f"$U_e$, $n$={n}")
            if i == len(ns) - 1:
                ax.set_xlabel("$x^*$")
    fig.suptitle(SETTING_TITLES[evaluation], y=1.08)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
               ncol=len(methods) + 1, frameon=False, bbox_to_anchor=(0.5, 1.04))
    _save(fig, run_dir, f"ue_examples_{evaluation}")


def run_plot(args) -> None:
    cache = _cache(args)
    df, ds = _point_frame(cache)
    table = relative_error_table(df)
    table.to_csv(cache.folder / "relative_error.csv", index=False, float_format="%.1f")
    print("Mean absolute relative error of U_e (%); beta_better: datasets in which the Beta has the lower error")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"[table] {cache.folder / 'relative_error.csv'}")
    for evaluation in INPUTS:
        plot_examples(df, ds, evaluation, METHODS, cache.folder)


def main() -> None:
    args = parse_args()
    if args.run in ("fit", "all"):
        run_fit(args)
    if args.run in ("plot", "all"):
        run_plot(args)


if __name__ == "__main__":
    main()
