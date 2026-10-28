#!/usr/bin/env python3
"""
Entropic uncertainty decomposition on two moons: replication of Fortini et al. (2026b), Fig. 27.

Thesis Sec. 4.5.1 (Fig. 4.9: noise 0.1) and App. A.8 (Fig. A.8: noise 0.4). sklearn make_moons with noise
0.1 (moons1) and 0.4 (moons2), one dataset per n in {30, 100} at a fixed seed, 100 x 100 evaluation grid.
g_n and V_n come from TabPFN's trajectory on the randomly permuted rows; the total h(g_n) is split under the
moment-matched Beta and the recentred probit.

Outputs in results/artifacts/replication_moons/: overview_<regime>.png, one figure per regime with rows
g_n, V_n/n, total, aleatoric, epistemic and one column per (n, method). Cache: the data and trajectory of each
(regime, n).

Usage
    python scripts/entropy_ud/replication_moons.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [p for p in (str(ROOT), str(ROOT / "src")) if p not in sys.path]

from scripts.common import RunCache, add_run_args, quiet_warnings, seed_all, set_threads  # noqa: E402
from scripts.entropy_ud.entropy_utils import METHODS, N_ESTIMATORS, decompose, gn_vn, permuted_trajectory  # noqa: E402,E501

quiet_warnings()

SEED = 2026
SIZES = (30, 100)
GRID_POINTS = 100                   # per coordinate
CACHE_VERSION = 2

# regimes and their evaluation grids (Fortini et al. 2026b, App. L.2.2)
REGIMES = {
    "moons1": {"noise": 0.1, "x1": (-1.5, 2.6), "x2": (-1.5, 2.6)},
    "moons2": {"noise": 0.4, "x1": (-3.0, 3.6), "x2": (-2.5, 3.1)},
}

SHORT_LABELS = {
    "beta": "Beta",
    "probit_recentred": "Probit",
}

# overview rows: method-independent first, then per method
SHARED_ROWS = (("g_n", "$g_n(x^*)$"), ("v_over_n", "$V_n(x^*)/n$"), ("total", "Total"))
METHOD_ROWS = (("aleatoric", "Aleatoric"), ("epistemic", "Epistemic"))

# figure size and font sizes (inches and points, at the native figure size)
OV_PANEL_W = 2.25     # per column (one column per (n, method))
OV_ROW_H = 2.1        # per row
FS_SUPTITLE = 12
FS_OV_TITLE = 9
FS_OV_LABEL = 9
FS_OV_TICK = 8
FS_OV_CBAR = 7
FS_OV_LEGEND = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entropic UD on two moons (Fortini et al. Fig. 27).")
    add_run_args(parser)
    return parser.parse_args()


def _make_data(regime: str, n: int, seed: int) -> dict:
    """make_moons with additive Gaussian noise at the regime's sigma."""
    from sklearn.datasets import make_moons

    x, y = make_moons(n_samples=n, noise=REGIMES[regime]["noise"], random_state=seed)
    return {"x_train": np.asarray(x, dtype=float), "y_train": np.asarray(y, dtype=int)}


def _make_grid(regime: str, n_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = REGIMES[regime]
    a = np.linspace(cfg["x1"][0], cfg["x1"][1], n_points)
    b = np.linspace(cfg["x2"][0], cfg["x2"][1], n_points)
    aa, bb = np.meshgrid(a, b)                       # (n_points, n_points)
    return a, b, np.column_stack([aa.ravel(), bb.ravel()])


def _cache(args) -> RunCache:
    config = {"seed": SEED, "n_estimators": N_ESTIMATORS, "grid_points": GRID_POINTS}
    return RunCache(args.outdir / "replication_moons", config, version=CACHE_VERSION)


def _key(regime: str, n: int) -> str:
    return f"{regime}/n{n:04d}/rep000"


def run_fit(args) -> None:
    from uqpfn import tabpfn

    set_threads(args.threads)
    seed_all(SEED)
    cache = _cache(args)
    cache.prepare(args.overwrite)
    todo = [(r, n) for r in REGIMES for n in SIZES if not cache.has(_key(r, n))]
    print(f"[fit] {len(todo)} datasets to do in {cache.folder}")
    model = tabpfn.make_classifier(n_estimators=N_ESTIMATORS) if todo else None
    for regime, n in todo:
        t0 = time.time()
        _, _, x_eval = _make_grid(regime, GRID_POINTS)
        data = _make_data(regime, n, seed=SEED)
        rng = np.random.default_rng(SEED + 7919 * n)
        traj = permuted_trajectory(data["x_train"], data["y_train"], x_eval, model=model, rng=rng)
        cache.save(_key(regime, n), x_train=data["x_train"], y_train=data["y_train"], traj=traj)
        print(f"[fit] {regime} n={n}: {time.time() - t0:.0f}s", flush=True)


def _shared_field(entry: dict, key: str, methods: tuple[str, ...]) -> np.ndarray:
    """Method-independent quantities: g_n, V_n/n and the total h(g_n)."""
    if key == "g_n":
        return entry["g_n"]
    if key == "v_over_n":
        return entry["v_n"] / entry["n"]
    return entry["dec"][methods[0]][key]


def _save_regime_overview(entries: list[dict], methods: tuple[str, ...], out_path: Path) -> None:
    """
    One figure per noise regime: a full grid of square heatmaps.

    Columns: for each n, one column per method (Beta | Probit).
    Rows:    g_n, V_n/n, Total, Aleatoric, Epistemic.

    g_n, V_n/n and Total are method independent, so those panels repeat across
    the Beta and Probit column of each n. Each row shares one colour scale
    (across all n and methods) with a single colorbar on the far right; g_n is
    drawn on [0, 1]. The top row carries the column headers.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regime = entries[0]["regime"]
    rows = SHARED_ROWS + METHOD_ROWS
    shared_keys = {key for key, _ in SHARED_ROWS}
    nm, nb = len(methods), len(entries)
    nrows, ncols = len(rows), nm * nb
    fig_w = OV_PANEL_W * ncols
    fig_h = OV_ROW_H * nrows + 0.4
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             squeeze=False, layout="constrained")

    def _field(entry: dict, key: str, m: str) -> np.ndarray:
        if key in shared_keys:
            return _shared_field(entry, key, methods)
        return entry["dec"][m][key]

    for r, (key, label) in enumerate(rows):
        # one colour scale per row, across all n and methods
        if key == "g_n":
            vmin, vmax = 0.0, 1.0
        else:
            vals = np.concatenate([_field(e, key, m) for e in entries for m in methods])
            vmin, vmax = float(vals.min()), float(vals.max())

        for k, entry in enumerate(entries):
            a, b = entry["axis1"], entry["axis2"]
            extent = (a[0], a[-1], b[0], b[-1])
            npts = len(a)
            x_tr, y_tr = entry["x_train"], entry["y_train"]

            for j, m in enumerate(methods):
                c = k * nm + j
                ax = axes[r, c]
                im = ax.imshow(_field(entry, key, m).reshape(npts, npts), origin="lower",
                               extent=extent, aspect="auto",
                               cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_box_aspect(1)
                ax.scatter(x_tr[y_tr == 0, 0], x_tr[y_tr == 0, 1], s=10, c="white",
                           edgecolors="black", linewidths=0.5, label="y=0", zorder=3)
                ax.scatter(x_tr[y_tr == 1, 0], x_tr[y_tr == 1, 1], s=10, c="crimson",
                           edgecolors="black", linewidths=0.5, marker="^", label="y=1",
                           zorder=3)

                title = label
                if r == 0:
                    title = f"{SHORT_LABELS[m]}, n={entry['n']}\n{label}"
                ax.set_title(title, fontsize=FS_OV_TITLE)

                ax.tick_params(labelsize=FS_OV_TICK)
                if r == nrows - 1:
                    ax.set_xlabel("$x_1^*$", fontsize=FS_OV_LABEL)
                else:
                    ax.tick_params(labelbottom=False)
                if c == 0:
                    ax.set_ylabel("$x_2^*$", fontsize=FS_OV_LABEL)
                else:
                    ax.tick_params(labelleft=False)

        # single colorbar for the row, attached to the right-most panel
        cax = axes[r, -1].inset_axes([1.06, 0.0, 0.06, 1.0])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=FS_OV_CBAR)

    axes[0, 0].legend(loc="upper right", fontsize=FS_OV_LEGEND, framealpha=0.85)
    fig.suptitle(f"{regime} (noise={REGIMES[regime]['noise']})", fontsize=FS_SUPTITLE)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def run_plot(args) -> None:
    cache = _cache(args)
    cache.require()
    for regime in REGIMES:
        axis1, axis2, _ = _make_grid(regime, GRID_POINTS)
        entries = []
        for n in SIZES:
            if not cache.has(_key(regime, n)):
                continue
            z = cache.load(_key(regime, n))
            g_n, v_n = gn_vn(z["traj"], n)
            entries.append({"regime": regime, "n": n, "axis1": axis1, "axis2": axis2, "x_train": z["x_train"],
                            "y_train": z["y_train"], "g_n": g_n, "v_n": v_n,
                            "dec": {m: decompose(g_n, v_n, n, m) for m in METHODS}})
        if entries:
            path = cache.folder / f"overview_{regime}.png"
            _save_regime_overview(entries, METHODS, path)
            print(f"[plot] {path}")


def main() -> None:
    args = parse_args()
    if args.run in ("fit", "all"):
        run_fit(args)
    if args.run in ("plot", "all"):
        run_plot(args)


if __name__ == "__main__":
    main()
