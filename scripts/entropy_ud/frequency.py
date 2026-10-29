#!/usr/bin/env python3
"""
Frequency setting: does TabPFN's epistemic term have the right size for a coin flip?

Thesis Sec. 4.5.2 (Frequency Setting, Fig. 4.10) and App. A.1.2.
DGP (one independent dataset per (theta, n, rep)):
  y ~ Bernoulli(theta),  theta in {0.5, 0.9},   x ~ N(0, 1) pure noise (TabPFN needs a feature),
  test point x* = 0,  S = sum(y),  n in {50, 100},  R = 10
References (exact, from S and n only):
  frequency   g_inf | z_1:n ~ Beta(S, n-S)       (martingale posterior of S_k/k; NaN if S in {0, n})
  bayes       g_inf | z_1:n ~ Beta(S+1, n-S+1)   (uniform prior)
  For Beta(a, b), m = a/(a+b):
    U_a = psi(a+b+1) - m psi(a+1) - (1-m) psi(b+1),   U_e = h(m) - U_a   (~ 1/(2n))
TabPFN: prefix trajectory on the shuffled rows -> g_n, V_n, decomposed with the Beta and the recentred probit.

Outputs in results/artifacts/frequency/: frequency.png (U_e per theta and n, TabPFN as mean +- SE over the
datasets, the references as horizontal lines) and frequency.csv (mean and SE of U_a, U_e per theta, n, method).
Cache: the trajectory of each (theta, n, rep).

Usage
    python scripts/entropy_ud/frequency.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import digamma

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [p for p in (str(ROOT), str(ROOT / "src")) if p not in sys.path]

from scripts.common import (  # noqa: E402
    RunCache, add_n_estimators_arg, add_run_args, add_sample_args, add_seed_arg, quiet_warnings, rep_key,
    set_threads,
)
from scripts.entropy_ud.entropy_utils import N_ESTIMATORS_THESIS, decompose as decompose_tabpfn, gn_vn  # noqa: E402
from uqpfn import asymptotic  # noqa: E402

quiet_warnings()

CACHE_VERSION = 2

TABPFN_METHODS = ("beta", "probit_recentred")
REF_METHODS = ("frequency", "bayes")
METHODS = REF_METHODS + TABPFN_METHODS
COLORS = {"beta": "C0", "probit_recentred": "C2", "frequency": "0.45", "bayes": "k"}
LABELS = {"beta": "TabPFN, Beta", "probit_recentred": "TabPFN, probit (recentred)",
          "frequency": "frequency rule, Beta(S, n-S)", "bayes": "Bayes, Beta(S+1, n-S+1)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frequency setting: TabPFN's U_e against exact Beta references.")
    add_run_args(parser)
    add_sample_args(parser, n_list=(50, 100), n_reps=10)
    parser.add_argument("--thetas", type=float, nargs="+", default=[0.5, 0.9],
                        help="success probabilities theta (default: %(default)s)")
    add_seed_arg(parser, 2026)
    add_n_estimators_arg(parser, N_ESTIMATORS_THESIS)
    return parser.parse_args()


def _cache(args) -> RunCache:
    return RunCache(args.outdir / "frequency", {"seed": args.seed, "n_estimators": args.n_estimators},
                    version=CACHE_VERSION)


def _key(theta: float, n: int, rep: int) -> str:
    return f"theta{round(theta * 1000):04d}/{rep_key(n, rep)}"


def make_data(n, theta, rng):
    x = rng.normal(0.0, 1.0, size=(n, 1))
    y = (rng.random(n) < theta).astype(int)
    perm = rng.permutation(n)  # the trajectory is order-dependent: shuffle the context
    return x[perm], y[perm]


def run_fit(args) -> None:
    from uqpfn import tabpfn

    set_threads(args.threads)
    cache = _cache(args)
    cache.prepare(args.overwrite)
    todo = [(t, n, r) for t in args.thetas for n in sorted(args.n_train_list) for r in range(args.n_reps)
            if not cache.has(_key(t, n, r))]
    print(f"[fit] {len(todo)} datasets to do in {cache.folder}")
    model = tabpfn.make_classifier(n_estimators=args.n_estimators) if todo else None
    for theta, n, rep in todo:
        t0 = time.time()
        rng = np.random.default_rng([args.seed, int(round(theta * 1000)), n, rep])
        x, y = make_data(n, theta, rng)
        traj = tabpfn.build_classifier_trajectory(x, y, np.zeros((1, 1)), n_estimators=args.n_estimators,
                                                  model=model)
        cache.save(_key(theta, n, rep), traj=traj, x=x, y=y)
        print(f"[fit] theta={theta:g} n={n} rep={rep}: {time.time() - t0:.0f}s", flush=True)


def beta_split(a, b):
    """Exact (m, total, U_a, U_e) for g_inf ~ Beta(a, b)."""
    m = a / (a + b)
    ua = digamma(a + b + 1) - m * digamma(a + 1) - (1 - m) * digamma(b + 1)
    total = float(asymptotic.binary_entropy(np.array([m]))[0])
    return {"g_n": m, "v_n": np.nan, "total": total, "ua": ua, "ue": total - ua}


def decompose(traj, y, method):
    n, s = len(y), float(np.sum(y))
    if method == "frequency":
        if s in (0, n):
            return {k: np.nan for k in ("g_n", "v_n", "total", "ua", "ue")}
        return beta_split(s, n - s)
    if method == "bayes":
        return beta_split(s + 1.0, n - s + 1.0)
    g_n, v_n = gn_vn(traj, n)
    d = decompose_tabpfn(g_n, v_n, n, method)
    return {"g_n": float(g_n[0]), "v_n": float(v_n[0]), "total": float(d["total"][0]),
            "ua": float(d["aleatoric"][0]), "ue": float(d["epistemic_raw"][0])}


def summarise(df):
    """Mean and SE over reps of U_a and U_e per (theta, n, method)."""
    g = df.groupby(["theta", "n", "method"], sort=False)
    out = g[["ua", "ue", "g_n"]].mean().add_suffix("_mean")
    out[["ua_se", "ue_se"]] = g[["ua", "ue"]].sem().to_numpy()
    out["n_valid"] = g["ue"].count()
    return out.reset_index()


def print_table(summary, n_estimators):
    """U_e (x 1e3) as mean +- SE: one row per (theta, n), one column per method."""
    fmt = summary.assign(cell=[f"{m * 1e3:6.2f} ± {s * 1e3:4.2f}"
                               for m, s in zip(summary["ue_mean"], summary["ue_se"])])
    table = fmt.pivot(index=["theta", "n"], columns="method", values="cell")[list(METHODS)]
    table.columns = [LABELS[m] for m in METHODS]
    print(f"\nU_e x 1e3, mean ± SE over datasets (TabPFN n_estimators={n_estimators})")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table.to_string())


def plot_ue(summary, out_path):
    """
    One panel per theta, one slot per n. TabPFN methods as points with SE bars;
    the two references as short horizontal lines across the slot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thetas = sorted(summary["theta"].unique())
    sizes = sorted(summary["n"].unique())
    s = summary.set_index(["theta", "n", "method"])
    offsets = dict(zip(TABPFN_METHODS, (-0.12, 0.12)))

    fig, axes = plt.subplots(1, len(thetas), figsize=(4.4 * len(thetas), 3.8), squeeze=False)
    for ax, theta in zip(axes[0], thetas):
        for i, n in enumerate(sizes):
            first = i == 0
            for m, ls in zip(REF_METHODS, ("--", "-")):
                ax.hlines(s.loc[(theta, n, m), "ue_mean"], i - 0.3, i + 0.3, color=COLORS[m],
                          lw=1.0, ls=ls, label=LABELS[m] if first else None)
            for m in TABPFN_METHODS:
                r = s.loc[(theta, n, m)]
                ax.errorbar(i + offsets[m], r["ue_mean"], yerr=r["ue_se"], fmt="o", ms=5,
                            capsize=3, color=COLORS[m], label=LABELS[m] if first else None)
        ax.set_xticks(range(len(sizes)), [f"n = {n}" for n in sizes])
        ax.set_xlim(-0.6, len(sizes) - 0.4)
        ax.set_title(f"$\\theta$ = {theta:g}")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("$U_e$")
    axes[0, -1].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_plot(args) -> None:
    cache = _cache(args)
    rows = []
    for key in cache.require():
        theta = int(key.split("/")[0][5:]) / 1000
        z = cache.load(key)
        n, rep = len(z["y"]), int(key.split("rep")[-1])
        for m in METHODS:
            rows.append({"theta": theta, "n": n, "rep": rep, "method": m, **decompose(z["traj"], z["y"], m)})
    summary = summarise(pd.DataFrame(rows))
    summary.to_csv(cache.folder / "frequency.csv", index=False)
    print_table(summary, args.n_estimators)
    plot_ue(summary, cache.folder / "frequency.png")
    print(f"[plot] {cache.folder / 'frequency.png'}, frequency.csv")


def main() -> None:
    args = parse_args()
    if args.run in ("fit", "all"):
        run_fit(args)
    if args.run in ("plot", "all"):
        run_plot(args)


if __name__ == "__main__":
    main()
