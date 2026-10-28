#!/usr/bin/env python3
"""
Entropic uncertainty decomposition on the 1D logistic DGP: replication of Fortini et al. (2026b), Fig. 25.

Thesis Sec. 4.5.1 and App. A.8 (Fig. A.7). One dataset per n in {15, 50, 75, 150} at a fixed seed,
x_i ~ N(1.5, 3^2), y_i ~ Bern(sigma(0.25 x_i - 0.5)), test grid [-15, 15] in steps of 0.2. g_n and V_n come
from TabPFN's trajectory on the randomly permuted rows; total h(g_n) is split under the moment-matched Beta
(left column) and the recentred probit (right column).

Outputs in results/artifacts/replication_logistic/: grid.png (per panel the total and aleatoric uncertainty
and the training data as class-coloured lines). Cache: the data and trajectory of each n.

Usage
    python scripts/entropy_ud/replication_logistic.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [p for p in (str(ROOT), str(ROOT / "src")) if p not in sys.path]

from scripts.common import RunCache, add_run_args, parse_rep_key, quiet_warnings, rep_key, seed_all, set_threads  # noqa: E402,E501
from scripts.entropy_ud.entropy_utils import METHODS, N_ESTIMATORS, decompose, gn_vn, permuted_trajectory  # noqa: E402,E501

quiet_warnings()

SEED = 2026
SIZES = (15, 50, 75, 150)
GRID = (-15.0, 15.0, 0.2)           # test grid: lo, hi, step
CACHE_VERSION = 2

# DGP (Fortini et al. 2026b, App. L.2.1)
X_MEAN = 1.5
X_SD = 3.0
BETA = 0.25
BETA0 = -0.5

METHOD_LABELS = {
    "beta": "Beta",
    "probit_recentred": "Probit CLT",
}

# font sizes of the figure (points, at the native figure size)
FS_TITLE = 11
FS_LABEL = 11
FS_TICK = 10
FS_LEGEND = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entropic UD on the 1D logistic DGP (Fortini et al. Fig. 25).")
    add_run_args(parser)
    return parser.parse_args()


def _make_data(n: int, seed: int) -> dict:
    """
    x_i ~ N(1.5, 3.0^2),  p_i = sigma(BETA * x_i + BETA0),  y_i ~ Bernoulli(p_i).
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(X_MEAN, X_SD, size=n)
    p = 1.0 / (1.0 + np.exp(-(BETA * x + BETA0)))
    y = (rng.random(n) < p).astype(int)
    return {"x_train": x.reshape(-1, 1), "y_train": y, "p_train": p}


def _cache(args) -> RunCache:
    config = {"seed": SEED, "n_estimators": N_ESTIMATORS, "grid": list(GRID)}
    return RunCache(args.outdir / "replication_logistic", config, version=CACHE_VERSION)


def _x_eval() -> np.ndarray:
    lo, hi, step = GRID
    return np.linspace(lo, hi, int(round((hi - lo) / step)) + 1)


def run_fit(args) -> None:
    from uqpfn import tabpfn

    set_threads(args.threads)
    seed_all(SEED)
    cache = _cache(args)
    cache.prepare(args.overwrite)
    todo = [n for n in SIZES if not cache.has(rep_key(n, 0))]
    print(f"[fit] {len(todo)} datasets to do in {cache.folder}")
    model = tabpfn.make_classifier(n_estimators=N_ESTIMATORS) if todo else None
    for n in todo:
        t0 = time.time()
        data = _make_data(n, seed=SEED)                 # one dataset per n at a fixed seed, as in the paper
        rng = np.random.default_rng(SEED + 7919 * n)
        traj = permuted_trajectory(data["x_train"], data["y_train"], _x_eval().reshape(-1, 1), model=model, rng=rng)
        cache.save(rep_key(n, 0), x_train=data["x_train"], y_train=data["y_train"], traj=traj)
        print(f"[fit] n={n}: {time.time() - t0:.0f}s", flush=True)


def _save_grid_plot(results: list[dict], methods: tuple[str, ...], out_path: Path) -> None:
    """
    Figure 25 layout, unchanged: per panel, the total uncertainty, the aleatoric
    uncertainty, and the training data as vertical dashed lines coloured by
    class. Total is h(g_n), the paper's definition, in every panel.

    One row per dataset size, one column per method (Beta left, probit right),
    so each panel is a like-for-like Figure 25 panel. The figure is sized close
    to a printed text width so fonts are not shrunk when included in LaTeX at
    width=\\linewidth.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    nrows, ncols = len(results), len(methods)
    panel_w, panel_h = 3.6, 2.5
    fig_w = panel_w * ncols if ncols > 1 else 5.0
    fig_h = panel_h * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             squeeze=False, sharex=True, sharey=True)

    c0, c1 = "violet", "turquoise"
    show_method = len(methods) > 1

    for i, entry in enumerate(results):
        xs = entry["x_eval"]
        x_tr = entry["x_train"].ravel()
        y_tr = entry["y_train"]
        lw = 0.7 if len(x_tr) <= 75 else 0.5
        alpha = 0.65 if len(x_tr) <= 75 else 0.45

        for j, method in enumerate(methods):
            ax = axes[i, j]

            for xv in x_tr[y_tr == 0]:
                ax.axvline(xv, color=c0, linestyle="--", linewidth=lw, alpha=alpha, zorder=1)
            for xv in x_tr[y_tr == 1]:
                ax.axvline(xv, color=c1, linestyle="--", linewidth=lw, alpha=alpha, zorder=1)

            d = entry["dec"][method]
            ax.plot(xs, d["total"], color="C0", linewidth=1.6, zorder=3)
            ax.plot(xs, d["aleatoric"], color="C1", linewidth=1.6, zorder=3)

            title = f"n={entry['n']}"
            if show_method:
                title = f"{METHOD_LABELS[method]} — {title}"
            ax.set_title(title, fontsize=FS_TITLE)
            ax.tick_params(labelsize=FS_TICK)
            ax.grid(True, alpha=0.25)
            if i == nrows - 1:
                ax.set_xlabel("Test covariate $x^*$", fontsize=FS_LABEL)
        axes[i, 0].set_ylabel("Uncertainty", fontsize=FS_LABEL)

    handles = [
        Line2D([], [], color="C0", label="Total Uncertainty"),
        Line2D([], [], color="C1", label="Aleatoric Uncertainty"),
        Line2D([], [], color=c0, linestyle="--", label="Data (y=0)"),
        Line2D([], [], color=c1, linestyle="--", label="Data (y=1)"),
    ]
    legend_cols = 4 if ncols > 1 else 2
    legend_h = 0.40 if legend_cols == 4 else 0.65   # inches reserved at the top
    fig.legend(handles=handles, loc="upper center", ncol=legend_cols,
               fontsize=FS_LEGEND, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1.0 - legend_h / fig_h))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_plot(args) -> None:
    cache = _cache(args)
    results = []
    for key in cache.require():
        n = parse_rep_key(key)[0]
        z = cache.load(key)
        g_n, v_n = gn_vn(z["traj"], n)
        results.append({"n": n, "x_eval": _x_eval(), "x_train": z["x_train"], "y_train": z["y_train"],
                        "dec": {m: decompose(g_n, v_n, n, m) for m in METHODS}})
    results.sort(key=lambda r: r["n"])
    path = cache.folder / "grid.png"
    _save_grid_plot(results, METHODS, path)
    print(f"[plot] {path}")


def main() -> None:
    args = parse_args()
    if args.run in ("fit", "all"):
        run_fit(args)
    if args.run in ("plot", "all"):
        run_plot(args)


if __name__ == "__main__":
    main()
