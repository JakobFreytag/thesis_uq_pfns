#!/usr/bin/env python3
"""
Linear and dependent DGP: coverage of AMP and the CLT V_n on the CDF scale.

Thesis Sec. 4.4.1: Tab. 4.2 and Fig. 4.6 (linear), App. A.7: Fig. A.6 (dependent). Pointwise 95% intervals
for P(Y <= 0 | X = x) on 100 points in [-10, 10], n = 200, 500, 1000, R = 10.
    linear     X ~ U(-10, 10),  Y = 0.2 X + N(0, 1)
    dependent  X ~ U(-10, 10),  Y | X = x ~ N(0.5 x + 1, (0.5 + 0.5 |x|)^2)
CLT V_n: trajectory of TabPFN's CDF on expanding prefixes of the shuffled data. AMP: TabPFN's CDF on all n
points propagated by the copula updates (B = 100, N = 50); percentile interval, point estimate the median.

Outputs in results/artifacts/linear_dependent/: coverage.csv and, per DGP, for repetition 0 at n = --plot-n:
<dgp>_n200 (.pdf, .png; training data | AMP | CLT V_n; linear: Fig. 4.6, dependent: Fig. A.6).
Cache: one file per (DGP, n, repetition), seed = 42 + 1000 rep + n.

Usage
    python scripts/coverage/linear_dependent.py
    python scripts/coverage/linear_dependent.py --dgps dependent --n-train-list 200
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
    RunCache, add_n_estimators_arg, add_run_args, add_sample_args, add_seed_arg, add_style_args, ci_label,
    parse_rep_key, quiet_warnings, rep_key, save_fig, se, set_threads, setup_style,
)
from scripts.coverage.coverage_utils import DGPS, threshold_grid  # noqa: E402
from uqpfn import asymptotic, martingale  # noqa: E402

quiet_warnings()

CACHE_VERSION = 2

LABEL_AMP = r"$\mathrm{AMP}$"
LABEL_CLT = r"$\mathrm{CLT}\ V_n$"
PLAIN_NAMES = {LABEL_AMP: "AMP", LABEL_CLT: "CLT V_n"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AMP vs CLT V_n coverage on the linear and dependent DGP.")
    add_run_args(parser)
    parser.add_argument("--dgps", nargs="+", choices=list(DGPS), default=list(DGPS))
    add_sample_args(parser, n_list=(200, 500, 1000), n_reps=10)
    add_seed_arg(parser, 42)
    parser.add_argument("--t-thresh", type=float, default=0.0, help="target P(Y <= t | X = x) (default: %(default)s)")
    parser.add_argument("--y-grid-size", type=int, default=250, help="size of the response grid (default: %(default)s)")
    parser.add_argument("--n-eval", type=int, default=100, help="evaluation points on [-10, 10] (default: %(default)s)")
    add_n_estimators_arg(parser, 64)
    parser.add_argument("--b-draws", type=int, default=100, help="AMP posterior draws B (default: %(default)s)")
    parser.add_argument("--t-steps", type=int, default=50, help="AMP forward steps N (default: %(default)s)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="intervals at level 1 - alpha, applied in the plot step (default: %(default)s)")
    parser.add_argument("--plot-n", type=int, default=200, help="sample size shown in the figures")
    add_style_args(parser)
    return parser.parse_args()


def _cache(args) -> RunCache:
    config = {"seed": args.seed, "t_thresh": args.t_thresh, "y_grid_size": args.y_grid_size, "n_eval": args.n_eval,
              "n_estimators": args.n_estimators, "b_draws": args.b_draws, "t_steps": args.t_steps}
    return RunCache(args.outdir / "linear_dependent", config, version=CACHE_VERSION)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def _fit_one(args, dgp: str, n_train: int, rep: int) -> dict:
    from uqpfn import tabpfn

    generate, _ = DGPS[dgp]
    seed = args.seed + rep * 1000 + n_train
    rng = np.random.default_rng(seed)
    X_train, y_train = generate(n_train, rng)
    X_eval = np.linspace(-10, 10, args.n_eval).reshape(-1, 1)

    # full response grid (AMP needs it for the copula updates), with t inserted so it is hit exactly
    y_grid = threshold_grid(y_train, args.y_grid_size, args.t_thresh)
    target_idx = int(np.where(y_grid == args.t_thresh)[0][0])

    # CLT V_n: trajectory on expanding prefixes of the shuffled data
    shuffle_idx = rng.permutation(n_train)
    pred_trajectory = tabpfn.build_predictive_trajectory(
        X_train[shuffle_idx], y_train[shuffle_idx], X_eval, y_grid, n_estimators=args.n_estimators
    )
    vn_variance = asymptotic.compute_vn(pred_trajectory, type="pointwise")

    # AMP: TabPFN's CDF on all n points, propagated by the copula updates
    pfn_cdf_init = tabpfn.analytic_cdf_from_tabpfn(X_train, y_train, X_eval, y_grid,
                                                   n_estimators=args.n_estimators)
    pfn_res = martingale.get_posterior(
        y_grid=y_grid, n_train=n_train, B_postsamples=args.b_draws, T_fwdsamples=args.t_steps,
        cdf_conditionals=pfn_cdf_init, alpha_dimension=X_train.shape[1],
    )
    amp_draws = np.array([res["cdf"][target_idx, :] for res in pfn_res])   # (n_eval, B): P_N^(b)(x, t)

    return {
        "seed": seed, "n_train": n_train, "x_train": X_train, "y_train": y_train, "x_eval": X_eval,
        "target_idx": target_idx, "final_cdf_t": pred_trajectory[-1][target_idx],
        "vn_t": vn_variance[target_idx], "amp_draws": amp_draws,
    }


def run_fit(args) -> None:
    set_threads(args.threads)
    cache = _cache(args)
    cache.prepare(args.overwrite)
    todo = [(dgp, n, rep) for dgp in args.dgps for n in args.n_train_list for rep in range(args.n_reps)
            if not cache.has(f"{dgp}/{rep_key(n, rep)}")]
    print(f"[fit] {len(todo)} runs to do in {cache.folder}")
    for i, (dgp, n, rep) in enumerate(todo, 1):
        t0 = time.perf_counter()
        cache.save(f"{dgp}/{rep_key(n, rep)}", **_fit_one(args, dgp, n, rep))
        print(f"[fit {i}/{len(todo)}] {dgp} n={n} rep={rep}: {time.perf_counter() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Plot: table and figures from the cache
# ---------------------------------------------------------------------------
def _entry(lo, med, hi, truth) -> dict:
    return {"lo": lo, "med": med, "hi": hi, "cov": (truth >= lo) & (truth <= hi), "len": hi - lo}


def _methods(z: dict, truth: np.ndarray, alpha: float) -> dict:
    pred_t = z["final_cdf_t"]
    clt_lo, clt_hi = asymptotic.pointwise_cdf_bounds(pred_t, z["vn_t"], int(z["n_train"]), alpha=alpha)
    draws = z["amp_draws"]
    amp_lo = np.percentile(draws, 100 * alpha / 2, axis=1)
    amp_hi = np.percentile(draws, 100 * (1 - alpha / 2), axis=1)
    return {
        LABEL_AMP: _entry(amp_lo, np.median(draws, axis=1), amp_hi, truth),
        LABEL_CLT: _entry(clt_lo, pred_t, clt_hi, truth),
    }


def _save_example_plot(plt, z: dict, truth: np.ndarray, methods: dict, out_stem: Path, fig_width: float,
                       ci_label: str) -> list[Path]:
    x = z["x_eval"][:, 0]
    order = np.argsort(x)
    xs = x[order]
    t = float(z["t_thresh"])

    # same geometry as the spline overview figure: extra gap after the training panel for the CDF axis label
    left, right, bottom, top, legend_space = 0.50, 0.08, 0.42, 0.28, 0.30
    ws = (0.55, 0.12)
    panel_w = (fig_width - left - right - sum(ws)) / 3
    panel_h = 0.9 * panel_w
    fig_h = bottom + legend_space + panel_h + top
    fig = plt.figure(figsize=(fig_width, fig_h))
    x0s = (left, left + panel_w + ws[0], left + 2 * panel_w + ws[0] + ws[1])
    rect = lambda x0: [x0 / fig_width, (bottom + legend_space) / fig_h, panel_w / fig_width, panel_h / fig_h]  # noqa: E731
    ax0 = fig.add_axes(rect(x0s[0]))
    ax1 = fig.add_axes(rect(x0s[1]))
    ax2 = fig.add_axes(rect(x0s[2]), sharey=ax1)

    ax0.scatter(z["x_train"][:, 0], z["y_train"], color="black", s=6, linewidths=0, label="Train data")
    ax0.axhline(t, color="red", linestyle="--", linewidth=1.3, label=rf"$t = {t:g}$")
    ax0.set_title(r"$\mathrm{Training\ data}$")
    ax0.set_xlabel(r"$x$")
    ax0.set_ylabel(r"$y$")
    ax0.grid(True, alpha=0.25)

    for i, (ax, (name, e)) in enumerate(zip((ax1, ax2), methods.items())):
        ax.plot(xs, truth[order], "--", color="C0", label=rf"True $P(Y \leq {t:g} \mid x)$")
        ax.plot(xs, e["med"][order], color="C2", label=rf"TabPFN $P(Y \leq {t:g} \mid x)$")
        ax.fill_between(xs, e["lo"][order], e["hi"][order], color="C0", alpha=0.25, linewidth=0, label=ci_label)
        ax.set_title(name)
        ax.set_xlabel(r"$x$")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.set_ylabel(rf"$P(Y \leq {t:g} \mid X = x)$")
        else:
            ax.tick_params(labelleft=False)

    handles, labels = [], []
    for ax in (ax0, ax1):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.8, columnspacing=1.2, handletextpad=0.4)

    paths = save_fig(fig, out_stem)
    plt.close(fig)
    return paths


def run_plot(args) -> None:
    cache = _cache(args)
    metrics, examples = {}, {}          # metrics: (dgp, n, method) -> {"cov": [...], "len": [...]}
    for key in cache.require():
        dgp, rest = key.split("/", 1)
        n, rep, _ = parse_rep_key(rest)
        z = cache.load(key)
        truth = DGPS[dgp][1](z["x_eval"], args.t_thresh)
        methods = _methods(z, truth, args.alpha)
        for name, e in methods.items():
            acc = metrics.setdefault((dgp, n, PLAIN_NAMES[name]), {"cov": [], "len": []})
            acc["cov"].append(float(np.mean(e["cov"])))
            acc["len"].append(float(np.mean(e["len"])))
        if n == args.plot_n and rep == 0:
            examples[dgp] = (z, truth, methods)

    path = cache.folder / "coverage.csv"
    print(f"{ci_label(args.alpha).removesuffix(' CI')} pointwise intervals for P(Y <= {args.t_thresh:g} | x)")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dgp", "n", "method", "coverage", "se_coverage", "width", "n_reps"])
        for (dgp, n, name), m in sorted(metrics.items(), key=lambda kv: (list(DGPS).index(kv[0][0]), kv[0][1])):
            writer.writerow([dgp, n, name, f"{np.mean(m['cov']):.6f}", f"{se(m['cov']):.6f}",
                             f"{np.mean(m['len']):.6f}", len(m["cov"])])
            print(f"  {dgp:<9} n={n:<5} {name:<8} coverage {np.mean(m['cov']):.3f}  width {np.mean(m['len']):.3f}  "
                  f"({len(m['cov'])} reps)")
    print(f"[table] {path}")

    for dgp, (z, truth, methods) in examples.items():
        plt = setup_style(args.font_size, args.usetex)
        stem = cache.folder / f"{dgp}_n{args.plot_n}"
        z = dict(z, t_thresh=args.t_thresh)
        paths = _save_example_plot(plt, z, truth, methods, stem, args.fig_width, ci_label(args.alpha, args.usetex))
        print(f"[plot] {', '.join(map(str, paths))}")


def main() -> None:
    args = parse_args()
    if args.run in ("fit", "all"):
        run_fit(args)
    if args.run in ("plot", "all"):
        run_plot(args)


if __name__ == "__main__":
    main()
