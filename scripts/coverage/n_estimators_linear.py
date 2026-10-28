#!/usr/bin/env python3
"""
Linear DGP: CLT V_n intervals for different TabPFN ensemble sizes n_estimators.

Thesis Sec. 4.3: Tab. 4.1 (linear row) and Fig. 4.5. Pointwise 95% intervals for P(Y <= 0 | X = x) at
n = 200, R = 10, n_estimators in {4, 8, 16, 32, 64}; the same datasets for all ensemble sizes (and the same
as linear_dependent.py).

Outputs in results/artifacts/n_estimators_linear/: coverage.csv and rows_n200_row1, _row2 (.pdf, .png), the
panels of repetition 0 at n = --plot-n in rows of three, to be stacked in LaTeX. Cache: one file per
(n, repetition, n_estimators), seed = 42 + 1000 rep + n.

Usage
    python scripts/coverage/n_estimators_linear.py
    python scripts/coverage/n_estimators_linear.py --n-train-list 500 1000 --n-reps 1 --plot-n 500   # App. A.6
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [p for p in (str(ROOT), str(ROOT / "src")) if p not in sys.path]

from scripts.common import (  # noqa: E402
    RunCache, add_run_args, add_sample_args, add_style_args, fixed_axes_figure, n_estimators,
    panel_geometry, parse_rep_key, quiet_warnings, rep_key, save_fig, se, set_threads, setup_style,
)
from scripts.coverage.coverage_utils import generate_linear, threshold_grid, true_prob_linear  # noqa: E402
from uqpfn import asymptotic  # noqa: E402

quiet_warnings()

SEED = 42
T_THRESH = 0.0                      # target P(Y <= t | X = x)
Y_GRID_SIZE = 250
N_EVAL = 100
ALPHA = 0.05                        # 95% intervals
PANELS_PER_ROW = 3
LEGEND_SPACE = 0.30
CACHE_VERSION = 2

LABEL_CLT = r"$\mathrm{CLT}\ V_n$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear DGP: CLT V_n intervals for different n_estimators.")
    add_run_args(parser)
    add_sample_args(parser, n_list=(200,), n_reps=10)
    parser.add_argument("--n-estimators-list", type=int, nargs="+",
                        default=sorted({n_estimators(e) for e in (4, 8, 16, 32, 64)}))
    parser.add_argument("--plot-n", type=int, default=200, help="sample size shown in the figures")
    add_style_args(parser)
    return parser.parse_args()


def _panel_title(n_est: int, usetex: bool) -> str:
    est = r"n\_estimators" if usetex else "n_estimators"
    return f"{LABEL_CLT} ({est} = {n_est})"


def _save_row_plot(
    plt,
    x: np.ndarray,
    truth: np.ndarray,
    entries: list[tuple[int, dict]],
    out_stem: Path,
    geo: dict,
    *,
    legend: bool,
    t_thresh: float,
    ci_label: str,
    usetex: bool,
) -> list[Path]:
    order = np.argsort(x)
    xs = x[order]
    fig, axes = fixed_axes_figure(plt, len(entries), geo, extra_bottom=LEGEND_SPACE if legend else 0.0, sharey=True)

    for i, (ax, (n_est, e)) in enumerate(zip(axes, entries)):
        ax.plot(xs, truth[order], "--", color="C0", label=rf"True $P(Y \leq {t_thresh:g} \mid x)$")
        ax.plot(xs, e["med"][order], "-", color="C2", label=rf"TabPFN $P(Y \leq {t_thresh:g} \mid x)$")
        ax.fill_between(xs, e["lo"][order], e["hi"][order], color="C0", alpha=0.25, linewidth=0, label=ci_label)
        # base font size (not +1 as in the spline script): the longer titles would otherwise touch each other
        ax.set_title(_panel_title(n_est, usetex), fontsize=plt.rcParams["font.size"])
        ax.set_xlabel(r"$x$")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.set_ylabel(rf"$P(Y \leq {t_thresh:g} \mid X = x)$")
        else:
            ax.tick_params(labelleft=False)

    if legend:
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
                   bbox_to_anchor=(0.5, 0.0), handlelength=2.2, columnspacing=1.4)

    paths = save_fig(fig, out_stem)
    plt.close(fig)
    return paths


def _cache(args) -> RunCache:
    config = {"seed": SEED, "t_thresh": T_THRESH, "y_grid_size": Y_GRID_SIZE, "n_eval": N_EVAL}
    return RunCache(args.outdir / "n_estimators_linear", config, version=CACHE_VERSION)


def _key(n: int, rep: int, n_est: int) -> str:
    return rep_key(n, rep, f"_est{n_est:03d}")


def _fit_one(n_train: int, rep: int, n_est: int) -> dict:
    """V_n trajectory on the shuffled data; the final CDF and V_n are cached."""
    from uqpfn import tabpfn

    seed = SEED + rep * 1000 + n_train
    rng = np.random.default_rng(seed)
    X_train, y_train = generate_linear(n_train, rng)
    X_eval = np.linspace(-10, 10, N_EVAL).reshape(-1, 1)
    y_grid = threshold_grid(y_train, Y_GRID_SIZE, T_THRESH)

    shuffle_idx = rng.permutation(n_train)          # the same permutation for every n_estimators
    pred_trajectory = tabpfn.build_predictive_trajectory(
        X_train[shuffle_idx], y_train[shuffle_idx], X_eval, y_grid, n_estimators=n_est
    )
    return {
        "seed": seed, "n_train": n_train, "n_est": n_est, "t_thresh": T_THRESH,
        "x_train": X_train, "y_train": y_train, "x_eval": X_eval, "y_grid": y_grid,
        "final_cdf": pred_trajectory[-1], "vn_variance": asymptotic.compute_vn(pred_trajectory, type="pointwise"),
    }


def run_fit(args) -> None:
    set_threads(args.threads)
    cache = _cache(args)
    cache.prepare(args.overwrite)
    todo = [(n, rep, e) for n in args.n_train_list for rep in range(args.n_reps) for e in args.n_estimators_list
            if not cache.has(_key(n, rep, e))]
    print(f"[fit] {len(todo)} runs to do in {cache.folder}")
    for i, (n, rep, e) in enumerate(todo, 1):
        t0 = time.perf_counter()
        cache.save(_key(n, rep, e), **_fit_one(n, rep, e))
        print(f"[fit {i}/{len(todo)}] n={n} rep={rep} n_estimators={e}: {time.perf_counter() - t0:.0f}s", flush=True)


def _intervals(z: dict) -> dict:
    """CLT V_n interval for P(Y <= t | x) from the cached final CDF and V_n."""
    idx = int(np.argmin(np.abs(z["y_grid"] - T_THRESH)))
    pred = z["final_cdf"][idx]
    lo, hi = asymptotic.pointwise_cdf_bounds(pred, z["vn_variance"][idx], int(z["n_train"]), alpha=ALPHA)
    truth = true_prob_linear(z["x_eval"], T_THRESH)
    return {"lo": lo, "med": pred, "hi": hi, "cov": (truth >= lo) & (truth <= hi), "len": hi - lo}


def run_plot(args) -> None:
    cache = _cache(args)
    metrics, example = {}, {}           # metrics: (n, n_est) -> {"cov": [...], "len": [...]}
    for key in cache.require():
        n, rep, est_s = parse_rep_key(key)
        n_est = int(est_s[4:])
        z = cache.load(key)
        e = _intervals(z)
        acc = metrics.setdefault((n, n_est), {"cov": [], "len": []})
        acc["cov"].append(float(np.mean(e["cov"])))
        acc["len"].append(float(np.mean(e["len"])))
        if n == args.plot_n and rep == 0:
            example[n_est] = (z, e)

    path = cache.folder / "coverage.csv"
    print(f"Linear DGP, CLT V_n, 95% intervals for P(Y <= {T_THRESH:g} | x)")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "n_estimators", "coverage", "se_coverage", "width", "n_reps"])
        for (n, n_est), m in sorted(metrics.items()):
            writer.writerow([n, n_est, f"{np.mean(m['cov']):.6f}", f"{se(m['cov']):.6f}",
                             f"{np.mean(m['len']):.6f}", len(m["cov"])])
            print(f"  n={n:<5} n_estimators={n_est:<3} coverage {np.mean(m['cov']):.3f}  "
                  f"width {np.mean(m['len']):.3f}  ({len(m['cov'])} reps)")
    print(f"[table] {path}")

    if not example:
        print(f"[plot] repetition 0 at n={args.plot_n} is not cached; no figures")
        return
    plt = setup_style(args.font_size, args.usetex)
    geo = panel_geometry(args.fig_width, n_cols=PANELS_PER_ROW)
    ci_label = r"95\% CI" if args.usetex else "95% CI"
    ests = sorted(example)
    rows = [ests[i:i + PANELS_PER_ROW] for i in range(0, len(ests), PANELS_PER_ROW)]
    z0 = example[ests[0]][0]
    truth = true_prob_linear(z0["x_eval"], T_THRESH)
    for k, row in enumerate(rows, 1):
        paths = _save_row_plot(
            plt, z0["x_eval"][:, 0], truth, [(e, example[e][1]) for e in row],
            cache.folder / f"rows_n{args.plot_n}_row{k}", geo,
            legend=(k == len(rows)), t_thresh=T_THRESH, ci_label=ci_label, usetex=args.usetex,
        )
        print(f"[plot] {', '.join(map(str, paths))}")


def main() -> None:
    args = parse_args()
    if args.run in ("fit", "all"):
        run_fit(args)
    if args.run in ("plot", "all"):
        run_plot(args)


if __name__ == "__main__":
    main()
