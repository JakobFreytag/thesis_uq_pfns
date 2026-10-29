#!/usr/bin/env python3
"""
Three-class spirals: replication of Fortini et al. (2026b) with the moment-matched Dirichlet and the per-class
recentred probit.

Thesis Sec. 4.5.3. One dataset of n = 200 points on three noisy spiral arms (seed 0), 100 x 100 grid on
[-4, 4]^2; both decompositions use the same g_n and per-class V_n from TabPFN's trajectory.

The Dirichlet has one concentration alpha_0, so with the mean matched its per-class variances are
Var_k = g_k (1 - g_k) / (alpha_0 + 1): it is exact per class only if the normalised CLT variances
r_k = (V_{n,k} / n) / (g_k (1 - g_k)) are equal, r_k = r_Dir = 1 / (alpha_0 + 1). Fig. 4.13 compares the
rarest class (smallest g_k among the classes with g_k >= 0.01) with r_Dir.

Outputs in results/artifacts/replication_spirals/:
    replication.png      total entropy | U_e Dirichlet | U_e probit, one colour scale (Fig. 4.12)
    dirichlet_sign.png   log2(r_rare / r_Dir) | U_e Dirichlet - U_e probit (Fig. 4.13)
Cache: the data and the multiclass TabPFN trajectory.

Usage
    python scripts/entropy_ud/replication_spirals.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [p for p in (str(ROOT), str(ROOT / "src")) if p not in sys.path]

from scripts.common import (  # noqa: E402
    RunCache, add_n_estimators_arg, add_run_args, add_seed_arg, quiet_warnings, rep_key, set_threads,
)
from scripts.entropy_ud.entropy_utils import N_ESTIMATORS_THESIS  # noqa: E402
from uqpfn import asymptotic  # noqa: E402

quiet_warnings()

K = 3               # three spiral arms
CACHE_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dirichlet vs recentred probit on three-class spirals.")
    add_run_args(parser)
    add_seed_arg(parser, 0)
    parser.add_argument("--n-train", type=int, default=200, help="sample size n (default: %(default)s)")
    parser.add_argument("--noise", type=float, default=0.1, help="sd of the noise on the arms (default: %(default)s)")
    parser.add_argument("--grid-size", type=int, default=100,
                        help="evaluation grid points per coordinate on [-4, 4] (default: %(default)s)")
    add_n_estimators_arg(parser, N_ESTIMATORS_THESIS)
    parser.add_argument("--g-min", type=float, default=0.01,
                        help="classes with g_k below this get no ratio r_k in Fig. 4.13 (default: %(default)s)")
    parser.add_argument("--font-scale", type=float, default=1.0, help="multiply every font size")
    return parser.parse_args()


def make_spirals(n: int = 200, n_classes: int = 3, noise: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    per = [n // n_classes + (1 if c < n % n_classes else 0) for c in range(n_classes)]
    xs, ys = [], []
    for c in range(n_classes):
        t = rng.uniform(0.0, 1.0, size=per[c])
        r = 4.0 * t
        theta = 2.0 * np.pi * 2.0 * t + 2.0 * np.pi * c / n_classes
        x1 = r * np.cos(theta) + rng.normal(0.0, noise, size=per[c])
        x2 = r * np.sin(theta) + rng.normal(0.0, noise, size=per[c])
        xs.append(np.column_stack([x1, x2]))
        ys.append(np.full(per[c], c, dtype=int))
    X, y = np.vstack(xs), np.concatenate(ys)
    perm = rng.permutation(n)              # the permutation the CLT needs
    return X[perm], y[perm]


def make_grid(size: int = 100, lo: float = -4.0, hi: float = 4.0):
    ax = np.linspace(lo, hi, size)
    g1, g2 = np.meshgrid(ax, ax)
    return np.column_stack([g1.ravel(), g2.ravel()]), ax


def _cache(args) -> RunCache:
    config = {"n_train": args.n_train, "grid_size": args.grid_size, "n_estimators": args.n_estimators,
              "seed": args.seed, "noise": args.noise}
    return RunCache(args.outdir / "replication_spirals", config, version=CACHE_VERSION)


def run_fit(args) -> None:
    set_threads(args.threads)
    cache = _cache(args)
    cache.prepare(args.overwrite)
    key = rep_key(args.n_train, 0)
    if cache.has(key):
        print(f"[fit] cached: {cache.path(key)}")
        return
    from uqpfn import tabpfn
    X, y = make_spirals(n=args.n_train, n_classes=K, noise=args.noise, seed=args.seed)
    X_eval, _ = make_grid(args.grid_size)
    traj = tabpfn.build_multiclass_trajectory(X, y, X_eval, n_classes=K, n_estimators=args.n_estimators,
                                              verbose=True)
    print(f"[fit] {cache.save(key, X=X, y=y, trajectory=traj)}")


def panel(ax, values, axis, title, X=None, y=None, **kw):
    size = len(axis)
    im = ax.imshow(values.reshape(size, size), origin="lower", aspect="equal",
                   extent=[axis[0], axis[-1], axis[0], axis[-1]],
                   **{"cmap": "viridis", **kw})
    if X is not None:
        for c in np.unique(y):
            sel = y == c
            ax.scatter(X[sel, 0], X[sel, 1], s=4, lw=0.2, edgecolor="w")
    ax.set_title(title, fontsize=0.9 * plt.rcParams["font.size"])   # 9 pt at scale 1
    ax.set_xticks([]); ax.set_yticks([])
    return im


def run_plot(args) -> None:
    cache = _cache(args)
    z = cache.load(cache.require()[0])
    X, y, traj = z["X"], z["y"], z["trajectory"]
    _, axis = make_grid(args.grid_size)
    plt.rcParams["font.size"] = 10.0 * args.font_scale   # ticks, colorbar labels, titles

    g_n = traj[-1]                                   # (n_eval, K)
    v_n = asymptotic.compute_vn_multiclass(traj)     # (n_eval, K)
    n = len(X)

    # the two decompositions
    dir_ = asymptotic.entropy_decomposition_multiclass(g_n, v_n, n)
    prb = asymptotic.entropy_decomposition_multiclass_probit(g_n, v_n, n, recenter=True)
    ue_d, ue_p = dir_["epistemic_raw"], prb["epistemic_raw"]
    diff = ue_d - ue_p

    # normalised CLT variance per class, and the rarest class against the Dirichlet's common value
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (v_n / n) / (g_n * (1.0 - g_n))
    r[(g_n < args.g_min) | ~(r > 0)] = np.nan
    n_valid = np.sum(~np.isnan(r), axis=1)           # points with < 2 valid classes: nothing to compare
    r_dir = 1.0 / (dir_["alpha0"] + 1.0)
    rare = np.argmin(np.where(np.isnan(r), np.inf, g_n), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log_ratio = np.log2(r[np.arange(len(r)), rare] / r_dir)
    log_ratio[n_valid < 2] = np.nan

    def save(fig, name):
        fig.savefig(cache.folder / name, dpi=160)
        plt.close(fig)
        print(f"[plot] {cache.folder / name}")

    # Fig. 4.12: one colour scale for all three
    maps = [(dir_["total"], "Total entropy"),
            (ue_d, "Epistemic (Dirichlet)"),
            (ue_p, "Epistemic (probit, recentred)")]
    lo = min(float(v.min()) for v, _ in maps)
    hi = max(float(v.max()) for v, _ in maps)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
    for ax, (vals, title) in zip(axes, maps):
        im = panel(ax, vals, axis, title, X, y, vmin=lo, vmax=hi)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="nats")
    save(fig, "replication.png")

    # Fig. 4.13: rarest class vs the Dirichlet, next to the difference of the two methods
    diff_kw = dict(vmin=-(float(np.max(np.abs(diff))) or 1.0),
                   vmax=float(np.max(np.abs(diff))) or 1.0, cmap="RdBu_r")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), layout="constrained")
    lim = float(np.nanmax(np.abs(log_ratio))) or 1.0
    im = panel(axes[0], log_ratio, axis, r"$\log_2(r_{rare} / r_{Dir})$", X, y,
               vmin=-lim, vmax=lim, cmap="PuOr_r")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    im = panel(axes[1], diff, axis, r"$U_e^{Dir} - U_e^{probit}$", X, y, **diff_kw)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
    save(fig, "dirichlet_sign.png")


def main() -> None:
    args = parse_args()
    if args.run in ("fit", "all"):
        run_fit(args)
    if args.run in ("plot", "all"):
        run_plot(args)


if __name__ == "__main__":
    main()
