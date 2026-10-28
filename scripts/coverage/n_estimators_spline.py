#!/usr/bin/env python3
"""
Balanced spline DGP: CLT V_n intervals for different TabPFN ensemble sizes n_estimators.

Thesis Sec. 4.3: Tab. 4.1 (spline rows) and App. A.6 (Fig. A.5). Pointwise 90% intervals for the median and
the 0.9 quantile of y | x at n = 200, R = 10, n_estimators in {4, 8, 16, 32, 64}. Data, seeds and shuffling
as in spline.py --mode balanced, so n_estimators = 64 is the CLT V_n of that script.

Outputs in results/artifacts/n_estimators_spline/: coverage.csv and rows_n200_row1, _row2 (.pdf, .png), the
median panels of repetition 0 at n = --plot-n in rows of three with a common y-range, to be stacked in
LaTeX. Cache: one file per (n, repetition, n_estimators).

Usage
    python scripts/coverage/n_estimators_spline.py
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
    panel_geometry, parse_rep_key, quiet_warnings, rep_key, save_fig, se, seed_all, set_threads, setup_style,
)
from scripts.coverage.coverage_utils import TARGETS, invert_cdf, make_spline_data  # noqa: E402
from uqpfn import asymptotic  # noqa: E402

quiet_warnings()

SEED = 2026
N_EVAL = 100
ALPHA = 0.10                        # 90% intervals
PANELS_PER_ROW = 3
LEGEND_SPACE = 0.30
CACHE_VERSION = 2

LABEL_CLT = r"$\mathrm{CLT}\ V_n$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Balanced spline DGP: CLT V_n for different n_estimators.")
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
    true_q: np.ndarray,
    posterior_q: np.ndarray,
    entries: list[tuple[int, dict]],
    out_stem: Path,
    geo: dict,
    *,
    legend: bool,
    ylim: tuple[float, float],
    target_name: str,
    ci_label: str,
    usetex: bool,
) -> list[Path]:
    order = np.argsort(x)
    xs = x[order]
    fig, axes = fixed_axes_figure(plt, len(entries), geo, extra_bottom=LEGEND_SPACE if legend else 0.0, sharey=True)

    for i, (ax, (n_est, e)) in enumerate(zip(axes, entries)):
        ax.plot(xs, true_q[order], "--", color="C0", label=f"True {target_name}")
        ax.plot(xs, posterior_q[order], ":", color="C1", label=f"Optimal {target_name}")
        ax.plot(xs, e["med"][order], color="C2", label=f"TabPFN {target_name}")
        ax.fill_between(xs, e["lo"][order], e["hi"][order], color="C0", alpha=0.25, linewidth=0, label=ci_label)
        # base font size (not +1 as in the spline script): the longer titles would otherwise touch each other
        ax.set_title(_panel_title(n_est, usetex), fontsize=plt.rcParams["font.size"])
        ax.set_xlabel(r"$x_1$")
        ax.set_ylim(*ylim)   # same y-range in every row file, so the stacked rows line up
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.set_ylabel(r"$y$")
        else:
            ax.tick_params(labelleft=False)

    if legend:
        handles, labels = axes[-1].get_legend_handles_labels()
        # compact spacing: four entries must fit under two panels, otherwise the legend widens the saved file
        # and the row is no longer centred when stacked
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
                   bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=1.0, handletextpad=0.4)

    paths = save_fig(fig, out_stem)
    plt.close(fig)
    return paths


def _cache(args) -> RunCache:
    return RunCache(args.outdir / "n_estimators_spline", {"seed": SEED, "n_eval": N_EVAL}, version=CACHE_VERSION)


def _key(n: int, rep: int, n_est: int) -> str:
    return rep_key(n, rep, f"_est{n_est:03d}")


def _fit_one(n: int, rep: int, n_est: int) -> dict:
    """The CLT V_n part of one balanced spline.py repetition, with the given n_estimators."""
    from uqpfn import tabpfn

    current_seed = SEED + rep * 99
    seed_all(current_seed)
    data = make_spline_data(mode="balanced", seed=current_seed, n_train=n, n_eval=N_EVAL)

    shuffle_idx = np.random.permutation(n)          # the same permutation for every n_estimators
    pred_trajectory = tabpfn.build_predictive_trajectory(
        data["x_train"][shuffle_idx], data["y_train"][shuffle_idx], data["x_eval"], data["y_grid"],
        n_estimators=n_est,
    )
    keep = ("x_train", "y_train", "x_eval", "y_grid") + tuple(
        f"{kind}_{t}" for t in TARGETS for kind in ("true", "posterior"))
    return {k: data[k] for k in keep} | {
        "seed": current_seed, "n_train": n, "n_est": n_est,
        "final_cdf": pred_trajectory[-1], "vn_variance": asymptotic.compute_vn(pred_trajectory),
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


def _intervals(z: dict, target: str) -> dict:
    """CLT V_n interval for the q-quantile of y | x: pointwise CDF bounds, inverted at q."""
    q = np.array([TARGETS[target]])
    lower_cdf, upper_cdf = asymptotic.pointwise_cdf_bounds(z["final_cdf"], z["vn_variance"], int(z["n_train"]),
                                                           alpha=ALPHA)
    lo = invert_cdf(z["y_grid"], upper_cdf, q)[0]
    hi = invert_cdf(z["y_grid"], lower_cdf, q)[0]
    med = invert_cdf(z["y_grid"], z["final_cdf"], q)[0]
    truth = z[f"true_{target}"]
    return {"lo": lo, "med": med, "hi": hi, "cov": (truth >= lo) & (truth <= hi), "len": hi - lo}


def run_plot(args) -> None:
    cache = _cache(args)
    metrics, example = {}, {}           # metrics: (n, target, n_est) -> {"cov": [...], "len": [...]}
    for key in cache.require():
        n, rep, est_s = parse_rep_key(key)
        n_est = int(est_s[4:])
        z = cache.load(key)
        for t in TARGETS:
            e = _intervals(z, t)
            acc = metrics.setdefault((n, t, n_est), {"cov": [], "len": []})
            acc["cov"].append(float(np.mean(e["cov"])))
            acc["len"].append(float(np.mean(e["len"])))
            if t == "median" and n == args.plot_n and rep == 0:
                example[n_est] = (z, e)

    path = cache.folder / "coverage.csv"
    print("Balanced spline DGP, CLT V_n, 90% intervals")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "target", "n_estimators", "coverage", "se_coverage", "width", "n_reps"])
        for (n, t, n_est), m in sorted(metrics.items(), key=lambda kv: (kv[0][0], list(TARGETS).index(kv[0][1]), kv[0][2])):
            writer.writerow([n, t, n_est, f"{np.mean(m['cov']):.6f}", f"{se(m['cov']):.6f}",
                             f"{np.mean(m['len']):.6f}", len(m["cov"])])
            print(f"  n={n:<5} {t:<6} n_estimators={n_est:<3} coverage {np.mean(m['cov']):.3f}  "
                  f"width {np.mean(m['len']):.3f}  ({len(m['cov'])} reps)")
    print(f"[table] {path}")

    if not example:
        print(f"[plot] repetition 0 at n={args.plot_n} is not cached; no figures")
        return
    plt = setup_style(args.font_size, args.usetex)
    geo = panel_geometry(args.fig_width, n_cols=PANELS_PER_ROW)
    ci_label = r"90\% CI" if args.usetex else "90% CI"
    ests = sorted(example)
    rows = [ests[i:i + PANELS_PER_ROW] for i in range(0, len(ests), PANELS_PER_ROW)]
    z0 = example[ests[0]][0]
    true_q, posterior_q = z0["true_median"], z0["posterior_median"]

    # common y-range over all panels of all rows (sharey only works within one figure)
    curves = [true_q, posterior_q] + [example[e][1][k] for e in ests for k in ("lo", "hi", "med")]
    y_min, y_max = min(float(np.min(c)) for c in curves), max(float(np.max(c)) for c in curves)
    pad = 0.05 * (y_max - y_min)
    for k, row in enumerate(rows, 1):
        paths = _save_row_plot(
            plt, z0["x_eval"][:, 0], true_q, posterior_q, [(e, example[e][1]) for e in row],
            cache.folder / f"rows_n{args.plot_n}_row{k}", geo, legend=(k == len(rows)),
            ylim=(y_min - pad, y_max + pad), target_name="median", ci_label=ci_label, usetex=args.usetex,
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
