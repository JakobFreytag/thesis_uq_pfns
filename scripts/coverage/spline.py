#!/usr/bin/env python3
"""
Spline DGP: coverage of AMP, the CLT V_n and the optimal (exact posterior) interval.

Thesis Sec. 4.4.2 and App. A.7. Pointwise 90% intervals for the median and the 0.9 quantile of y | x,
evaluated over all points and separately in the sparsely sampled gap [-1, 1] and outside of it.
    --mode unbalanced   Tab. 4.3 (median), Tab. A.1 (0.9 quantile), Fig. 4.7 (methods), Fig. 4.8 (overview)
    --mode balanced     Tab. A.2

Outputs in results/artifacts/spline_<mode>/: coverage.csv, methods_n200 and overview_n200 (.pdf, .png)
for repetition 0 at n = --plot-n. Cache: one file per (n, repetition), seed = 2026 + 99 rep.

Usage
    python scripts/coverage/spline.py                     # unbalanced design, n = 50, 100, 200, R = 20
    python scripts/coverage/spline.py --mode balanced
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [p for p in (str(ROOT), str(ROOT / "src")) if p not in sys.path]

from scripts.common import (  # noqa: E402
    RunCache, add_run_args, add_sample_args, add_style_args, fixed_axes_figure, n_estimators,
    panel_geometry, parse_rep_key, quiet_warnings, rep_key, save_fig, se, seed_all, set_threads, setup_style,
)
from scripts.coverage.coverage_utils import (  # noqa: E402
    GAP, TARGETS, analytic_ci, invert_cdf, make_spline_data, median_ci_from_resamples,
)
from uqpfn import asymptotic, martingale  # noqa: E402

quiet_warnings()

SEED = 2026
N_EVAL = 100
N_ESTIMATORS = n_estimators(64)
B_DRAWS, T_STEPS = 100, 50          # AMP posterior draws and forward steps
CI_ALPHA = 0.10                     # 90% intervals
CACHE_VERSION = 3

# method names in the plots (math mode so V_n renders properly) and in the table
LABEL_OPTIMAL = r"$\mathrm{Optimal}$"
LABEL_AMP = r"$\mathrm{AMP}$"
LABEL_CLT = r"$\mathrm{CLT}\ V_n$"
PLAIN_NAMES = {LABEL_OPTIMAL: "Optimal", LABEL_AMP: "AMP", LABEL_CLT: "CLT V_n"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spline DGP: AMP vs CLT V_n vs the optimal interval.")
    add_run_args(parser)
    parser.add_argument("--mode", choices=("unbalanced", "balanced"), default="unbalanced", help="design")
    add_sample_args(parser, n_list=(50, 100, 200), n_reps=20)
    parser.add_argument("--plot-n", type=int, default=200, help="sample size shown in the figures")
    add_style_args(parser)
    return parser.parse_args()


def _cache(args) -> RunCache:
    config = {"mode": args.mode, "seed": SEED, "n_eval": N_EVAL, "n_estimators": N_ESTIMATORS,
              "b_draws": B_DRAWS, "t_steps": T_STEPS}
    return RunCache(args.outdir / f"spline_{args.mode}", config, version=CACHE_VERSION)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def _fit_one_rep(args, n: int, rep: int) -> dict:
    current_seed = SEED + rep * 99
    seed_all(current_seed)
    data = make_spline_data(mode=args.mode, seed=current_seed, n_train=n, n_eval=N_EVAL)

    out = {k: data[k] for k in ("sigma", "x_train", "y_train", "x_eval", "y_grid")} | {"seed": current_seed}

    # optimal: exact posterior interval of the q-quantile
    for t, q in TARGETS.items():
        lo_an, hi_an = analytic_ci(data["z_ev"], data["mu_train"], data["sigma_train"], sigma=data["sigma"],
                                   q=q, level=1 - CI_ALPHA)
        out[f"true_{t}"] = data[f"true_{t}"]
        out[f"posterior_{t}"] = data[f"posterior_{t}"]
        out[f"lo_an_{t}"] = lo_an
        out[f"hi_an_{t}"] = hi_an
        out[f"med_an_{t}"] = data[f"posterior_{t}"]

    # TabPFN trajectory on the permuted rows, shared by the CLT and AMP
    from uqpfn import tabpfn
    n_train = len(data["x_train"])
    shuffle_idx = np.random.permutation(n_train)
    pred_trajectory = tabpfn.build_predictive_trajectory(
        data["x_train"][shuffle_idx], data["y_train"][shuffle_idx], data["x_eval"], data["y_grid"],
        n_estimators=N_ESTIMATORS,
    )
    final_cdf = pred_trajectory[-1]     # TabPFN's prediction on all n points, the centre of both methods

    # CLT V_n: pointwise CDF bounds, inverted at q (the lower bound comes from the upper CDF curve)
    vn_variance = asymptotic.compute_vn(pred_trajectory)
    lower_cdf, upper_cdf = asymptotic.pointwise_cdf_bounds(final_cdf, vn_variance, n_train, alpha=CI_ALPHA)
    for t, q in TARGETS.items():
        out[f"fortini_lo_{t}"] = invert_cdf(data["y_grid"], upper_cdf, np.array([q]))[0]
        out[f"fortini_hi_{t}"] = invert_cdf(data["y_grid"], lower_cdf, np.array([q]))[0]
        out[f"fortini_med_{t}"] = invert_cdf(data["y_grid"], final_cdf, np.array([q]))[0]

    # AMP: copula updates starting from the same prediction; percentile interval of the q-quantile over the
    # draws, point estimate the q-quantile of the mixture CDF
    pfn_res = martingale.get_posterior(
        data["y_grid"], n_train, B_DRAWS, T_STEPS, final_cdf, alpha_dimension=data["x_train"].shape[1],
    )
    pfn_mean_cdf = np.stack([np.mean(entry["cdf"], axis=1) for entry in pfn_res]).T
    for t, q in TARGETS.items():
        out[f"pfn_ci_{t}"] = median_ci_from_resamples(
            pfn_res, data["y_grid"], lo=100 * CI_ALPHA / 2, hi=100 * (1 - CI_ALPHA / 2), q=q
        )
        out[f"pfn_med_{t}"] = invert_cdf(data["y_grid"], pfn_mean_cdf, np.array([q]))[0]

    # for the overview figure
    out["final_cdf"] = final_cdf
    out["vn_variance"] = vn_variance
    return out


def run_fit(args) -> None:
    set_threads(args.threads)
    cache = _cache(args)
    cache.prepare(args.overwrite)
    todo = [(rep, n) for rep in range(args.n_reps) for n in args.n_train_list if not cache.has(rep_key(n, rep))]
    print(f"[fit] {len(todo)} runs to do in {cache.folder}")
    for i, (rep, n) in enumerate(todo, 1):
        t0 = time.perf_counter()
        cache.save(rep_key(n, rep), **_fit_one_rep(args, n, rep))
        print(f"[fit {i}/{len(todo)}] n={n} rep={rep}: {time.perf_counter() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Plot: table and figures from the cache
# ---------------------------------------------------------------------------
def _entry(lo, med, hi, truth) -> dict:
    return {"lo": lo, "med": med, "hi": hi, "cov": (truth >= lo) & (truth <= hi), "len": hi - lo}


def _methods_from_cache(z: dict, t: str) -> dict:
    truth = z[f"true_{t}"]
    return {
        LABEL_OPTIMAL: _entry(z[f"lo_an_{t}"], z[f"med_an_{t}"], z[f"hi_an_{t}"], truth),
        LABEL_AMP: _entry(z[f"pfn_ci_{t}"][:, 0], z[f"pfn_med_{t}"], z[f"pfn_ci_{t}"][:, 1], truth),
        LABEL_CLT: _entry(z[f"fortini_lo_{t}"], z[f"fortini_med_{t}"], z[f"fortini_hi_{t}"], truth),
    }


def _save_methods_plot(
    plt,
    x: np.ndarray,
    true_median: np.ndarray,
    posterior_median: np.ndarray,
    methods: dict,
    out_stem: Path,
    geo: dict,
    ci_label: str,
) -> list[Path]:
    order = np.argsort(x)
    xs = x[order]
    true_s = true_median[order]
    posterior_s = posterior_median[order]

    legend_space = 0.30
    fig, axes = fixed_axes_figure(plt, len(methods), geo, extra_bottom=legend_space, sharey=True)

    for i, (ax, (name, entry)) in enumerate(zip(axes, methods.items())):
        lo = entry["lo"][order]
        med = entry["med"][order]
        hi = entry["hi"][order]

        ax.plot(xs, true_s, "--", color="C0", label="True median")
        ax.plot(xs, posterior_s, ":", color="C1", label="Optimal median")
        # in the Optimal panel the predicted median IS the optimal (posterior) median,
        # so only draw the TabPFN median for the TabPFN-based methods
        if name != LABEL_OPTIMAL:
            ax.plot(xs, med, color="C2", label="TabPFN median")
        ax.fill_between(xs, lo, hi, color="C0", alpha=0.25, linewidth=0, label=ci_label)
        ax.set_title(name)
        ax.set_xlabel(r"$x_1$")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.set_ylabel(r"$y$")
        else:
            ax.tick_params(labelleft=False)

    handles, labels = axes[-1].get_legend_handles_labels()  # last panel has all four entries
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.2, columnspacing=1.4)

    paths = save_fig(fig, out_stem)
    plt.close(fig)
    return paths


def _save_overview_plot(plt, z: dict, out_stem: Path, fig_width: float, usetex: bool,
                        show_gap: bool = True) -> list[Path]:
    """Training data, predictive spread (TabPFN vs optimal) and V_n along x for one repetition."""
    x = z["x_eval"][:, 0]
    order = np.argsort(x)
    xs = x[order]
    y_grid, final_cdf, vn = z["y_grid"], z["final_cdf"], z["vn_variance"]
    z95, sig = norm.ppf(0.95), float(z["sigma"])

    # TabPFN predictive 5 / 50 / 95% quantiles
    q05, q50, q95 = invert_cdf(y_grid, final_cdf, np.array([0.05, 0.5, 0.95]))
    # optimal posterior predictive N(q_Pi, sigma^2 + v(x)); v(x) recovered from the 90% analytic CI of f
    q_pi = z["posterior_median"]
    sd_f = (z["hi_an_median"] - z["lo_an_median"]) / (2 * z95)
    sd_pred = np.sqrt(sig**2 + sd_f**2)
    # V_n at the threshold where the predictive CDF is closest to 0.5
    idx_med = np.argmin(np.abs(final_cdf - 0.5), axis=0)
    vn_med = vn[idx_med, np.arange(vn.shape[1])]

    pct, dash = (r"\%", "--") if usetex else ("%", "\u2013")

    # geometry: same total width as the methods figure, extra gap before the V_n panel for its tick labels
    left, right, bottom, top, legend_space = 0.50, 0.08, 0.42, 0.28, 0.45
    ws = (0.12, 0.55)
    panel_w = (fig_width - left - right - sum(ws)) / 3
    panel_h = 0.9 * panel_w
    fig_h = bottom + legend_space + panel_h + top
    fig = plt.figure(figsize=(fig_width, fig_h))
    x0s = (left, left + panel_w + ws[0], left + 2 * panel_w + ws[0] + ws[1])
    rect = lambda x0: [x0 / fig_width, (bottom + legend_space) / fig_h, panel_w / fig_width, panel_h / fig_h]  # noqa: E731
    ax0 = fig.add_axes(rect(x0s[0]))
    ax1 = fig.add_axes(rect(x0s[1]), sharey=ax0)
    ax2 = fig.add_axes(rect(x0s[2]))

    for i, ax in enumerate((ax0, ax1, ax2)):
        if show_gap:
            ax.axvspan(*GAP, color="0.5", alpha=0.15, linewidth=0, label=r"Gap $[-1, 1]$" if i == 0 else None)
        ax.set_xlabel(r"$x_1$")
        ax.grid(True, alpha=0.25)

    ax0.scatter(z["x_train"][:, 0], z["y_train"], color="black", s=6, linewidths=0, label="Train data")
    ax0.set_title(r"$\mathrm{Training\ data}$")
    ax0.set_ylabel(r"$y$")

    ax1.fill_between(xs, q05[order], q95[order], color="C2", alpha=0.25, linewidth=0,
                     label=f"TabPFN predictive 5{dash}95{pct}")
    ax1.plot(xs, q50[order], color="C2", label="TabPFN median")
    ax1.plot(xs, (q_pi - z95 * sd_pred)[order], ":", color="C1", label=f"Optimal predictive 5{dash}95{pct}")
    ax1.plot(xs, (q_pi + z95 * sd_pred)[order], ":", color="C1")
    ax1.set_title(r"$\mathrm{Predictive\ spread}$")
    ax1.tick_params(labelleft=False)

    ax2.plot(xs, vn_med[order], color="C3")
    ax2.set_title(r"$V_n\ \mathrm{at\ the\ median}$")
    ax2.set_ylabel(r"$V_n$")
    ax2.set_ylim(bottom=0)

    handles, labels = [], []
    for ax in (ax0, ax1):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.8, columnspacing=1.2, handletextpad=0.4)

    paths = save_fig(fig, out_stem)
    plt.close(fig)
    return paths


def run_plot(args) -> None:
    cache = _cache(args)
    metrics, example = {}, None     # metrics: (n, target, region, method) -> {"cov": [...], "len": [...]}
    for key in cache.require():
        n, rep, _ = parse_rep_key(key)
        z = cache.load(key)
        x = z["x_eval"][:, 0]
        in_gap = (x >= GAP[0]) & (x <= GAP[1])
        regions = {"all": np.ones_like(in_gap), "gap": in_gap, "outside": ~in_gap}
        for t in TARGETS:
            for name, e in _methods_from_cache(z, t).items():
                for region, m in regions.items():
                    acc = metrics.setdefault((n, t, region, PLAIN_NAMES[name]), {"cov": [], "len": []})
                    acc["cov"].append(float(np.mean(e["cov"][m])))
                    acc["len"].append(float(np.mean(e["len"][m])))
        if n == args.plot_n and rep == 0:
            example = z

    # mean over x, then over repetitions; SE across repetitions
    path = cache.folder / "coverage.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "target", "region", "method", "coverage", "se_coverage", "width", "n_reps"])
        for (n, t, region, name), m in sorted(metrics.items(), key=lambda kv: (kv[0][0], list(TARGETS).index(kv[0][1]))):
            writer.writerow([n, t, region, name, f"{np.mean(m['cov']):.6f}", f"{se(m['cov']):.6f}",
                             f"{np.mean(m['len']):.6f}", len(m["cov"])])
    print(f"Spline DGP ({args.mode}), 90% intervals: coverage / width over all, gap, outside")
    for n in sorted({k[0] for k in metrics}):
        for t in TARGETS:
            print(f"n={n}, {t} ({len(metrics[(n, t, 'all', 'AMP')]['cov'])} reps)")
            for name in PLAIN_NAMES.values():
                cells = "   ".join(f"{np.mean(metrics[(n, t, r, name)]['cov']):.3f} / "
                                   f"{np.mean(metrics[(n, t, r, name)]['len']):.3f}" for r in ("all", "gap", "outside"))
                print(f"  {name:<8} {cells}")
    print(f"[table] {path}")

    if example is None:
        print(f"[plot] repetition 0 at n={args.plot_n} is not cached; no figures")
        return
    plt = setup_style(args.font_size, args.usetex)
    geo = panel_geometry(args.fig_width, n_cols=3)
    ci_label = r"90\% CI" if args.usetex else "90% CI"
    x = example["x_eval"][:, 0]
    paths = _save_methods_plot(plt, x, example["true_median"], example["posterior_median"],
                               _methods_from_cache(example, "median"), cache.folder / f"methods_n{args.plot_n}",
                               geo, ci_label)
    paths += _save_overview_plot(plt, example, cache.folder / f"overview_n{args.plot_n}", args.fig_width,
                                 args.usetex, show_gap=args.mode == "unbalanced")
    print(f"[plot] {', '.join(map(str, paths))}")


def main() -> None:
    args = parse_args()
    if args.run in ("fit", "all"):
        run_fit(args)
    if args.run in ("plot", "all"):
        run_plot(args)


if __name__ == "__main__":
    main()
