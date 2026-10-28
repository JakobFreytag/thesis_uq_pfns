#!/usr/bin/env python3
"""
Three Gaussian clusters: the multiclass Dirichlet and the per-class recentred probit against an exact Bayesian
benchmark, with exact inputs and with TabPFN's inputs.

Thesis Sec. 4.5.4. Both approximations turn a predictive mean g_k(x*) and per-class variances V_k(x*) into a
law for g_inf(x*) and U_e = H(g) - E[H(g_inf)]. The Dirichlet has one shared normalised variance
rbar = 1 / (alpha_0 + 1), the probit keeps every class's r_k = V_k / (g_k (1 - g_k)).

DGP and benchmark (App. A.1.2):
  mu_1 = (-1.25, 0),  mu_2 = (1.25, 0),  mu_3 = (0, 2),  x | y = k ~ N(mu_k, I),  30 points per class
  mu_k ~ N(0, 5^2 I) with the class spread and pi = 1/3 known, so mu_k | D is Gaussian and g_inf(x*) | D is
  p(x*; mu) = softmax_k(x*'mu_k - ||mu_k||^2 / 2) under the posterior; from S = 5000 exact posterior draws:
  g_k = mean p_k,  V_k = var p_k,  U_e^ref = H(g) - mean H(p)  (a second set of draws gives its MC error)
  evaluated on a 50 x 50 grid and at 200 points along x1 = 0; TabPFN on the same points, rows permuted once

Outputs in results/artifacts/multiclass_blops/:
    ue_exact.png    U_e^ref | Dirichlet | probit with exact inputs (Fig. 4.14)
    transect.png    along x1 = 0: g_k | r_k and rbar | U_e; solid exact, dotted TabPFN inputs (Fig. 4.15)
    sign_check.png  rbar - r_k (top) against U_e,k^Dir - U_e,k^ref (bottom), exact inputs (Fig. 4.16)
    ue_tabpfn.png   U_e^ref | Dirichlet | probit with TabPFN's inputs (Fig. 4.17)
    summary.csv     error of U_e per method and inputs over the grid, and the MC error of the benchmark
Cache: the multiclass TabPFN trajectory; the benchmark takes seconds and is recomputed by the plot step.

Usage
    python scripts/entropy_ud/multiclass_blops.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import digamma, log_softmax

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [p for p in (str(ROOT), str(ROOT / "src")) if p not in sys.path]

from scripts.common import RunCache, add_run_args, quiet_warnings, rep_key, set_threads  # noqa: E402
from scripts.entropy_ud.entropy_utils import N_ESTIMATORS  # noqa: E402
from uqpfn import asymptotic  # noqa: E402

quiet_warnings()

SEED = 2026
A, H = 1.25, 2.0                    # centres (-A, 0), (A, 0), (0, H)
SIGMA = 1.0                         # known within-class sd
N_PER_CLASS = 30
TAU, M0 = 5.0, 0.0                  # prior of each centre coordinate
N_DRAWS = 5000                      # exact posterior draws S
GRID_POINTS = 50                    # per axis
TRANSECT_POINTS = 200               # along x1 = 0
MARGIN = 3.0                        # the grid extends this many sigma beyond the true centres
CHUNK = 500                         # grid points per benchmark batch (memory)
CACHE_VERSION = 2

K = 3
CLASS_COLORS = ("tab:red", "tab:green", "tab:blue")
METHODS = ("dirichlet", "probit")
METHOD_LABELS = {"dirichlet": "Dirichlet", "probit": "Probit (recentred)"}
METHOD_COLORS = {"dirichlet": "tab:orange", "probit": "tab:purple"}
TAB = "_tabpfn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dirichlet vs recentred probit U_e against an exact Bayesian "
                                                 "benchmark (three Gaussian clusters).")
    add_run_args(parser)
    parser.add_argument("--font-scale", type=float, default=1.0, help="multiply every font size")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# DGP, conjugate posterior and benchmark
# ---------------------------------------------------------------------------
def true_centers(a: float, h: float) -> np.ndarray:
    return np.array([[-a, 0.0], [a, 0.0], [0.0, h]])


def make_data(centers: np.ndarray, sigma: float, n_per_class: list[int], seed: int):
    """
    Stratified sample, rows ordered by class. Each class has its own noise
    stream, and its first n_k draws do not depend on n_k, so changing h or n_3
    moves / extends the same points rather than redrawing them.
    """
    xs, ys = [], []
    for k, (mu, nk) in enumerate(zip(centers, n_per_class)):
        z = np.random.default_rng([seed, k]).standard_normal((nk, 2))
        xs.append(mu + sigma * z)
        ys.append(np.full(nk, k))
    return np.vstack(xs), np.concatenate(ys)


def posterior(x: np.ndarray, y: np.ndarray, sigma: float, tau: float, m0: float):
    """mu_k | D ~ N(m_k, s_k^2 I). Returns m (K, 2) and s (K,)."""
    n_k = np.bincount(y, minlength=K).astype(float)
    sum_k = np.stack([x[y == k].sum(axis=0) for k in range(K)])
    s2 = 1.0 / (1.0 / tau ** 2 + n_k / sigma ** 2)
    m = s2[:, None] * (m0 / tau ** 2 + sum_k / sigma ** 2)
    return m, np.sqrt(s2)


def draw_centers(m: np.ndarray, s: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    """Exact iid posterior draws, (S, K, 2). Same seed -> same standard normals
    (and the first S' < S draws equal a run with S')."""
    eps = np.random.default_rng([seed, 99]).standard_normal((n_draws, K, 2))
    return m[None] + s[None, :, None] * eps


def log_class_probs(x_eval: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    """
    log p_k(x*; mu) for every draw, (S, n_eval, K). With pi uniform and sigma
    shared, ||x*||^2 and log pi cancel in the softmax, so the logits are linear
    in x*: (x* . mu_k - ||mu_k||^2 / 2) / sigma^2.
    """
    lin = np.matmul(x_eval, mu.transpose(0, 2, 1))          # (S, n_eval, K)
    logits = (lin - 0.5 * np.sum(mu ** 2, axis=-1)[:, None, :]) / sigma ** 2
    return log_softmax(logits, axis=-1)


def normalised_variance(g: np.ndarray, V: np.ndarray) -> np.ndarray:
    return V / np.maximum(g * (1.0 - g), np.finfo(float).tiny)


def reference(x_eval: np.ndarray, mu: np.ndarray, sigma: float, chunk: int) -> dict:
    """Exact targets from the posterior draws, in grid batches to bound memory."""
    n = len(x_eval)
    g, V, eta_mean = (np.empty((n, K)) for _ in range(3))
    for i0 in range(0, n, chunk):
        sl = slice(i0, i0 + chunk)
        logp = log_class_probs(x_eval[sl], mu, sigma)
        p = np.exp(logp)
        g[sl] = p.mean(axis=0)
        V[sl] = p.var(axis=0)
        eta_mean[sl] = (-p * logp).mean(axis=0)
    ue_k = asymptotic.neg_p_log_p(g) - eta_mean
    return {
        "g": g, "V": V,
        "r": normalised_variance(g, V),
        "ua": eta_mean.sum(axis=1),
        "ue": ue_k.sum(axis=1),
        "ue_k": ue_k,
    }




def design():
    """Data, true centres and the evaluation points (grid and transect along x1 = 0)."""
    centers = true_centers(A, H)
    x, y = make_data(centers, SIGMA, [N_PER_CLASS] * K, SEED)
    lo = centers.min(axis=0) - MARGIN * SIGMA
    hi = centers.max(axis=0) + MARGIN * SIGMA
    axes = (np.linspace(lo[0], hi[0], GRID_POINTS), np.linspace(lo[1], hi[1], GRID_POINTS))
    aa, bb = np.meshgrid(*axes)
    x_grid = np.column_stack([aa.ravel(), bb.ravel()])
    t = np.linspace(lo[1], hi[1], TRANSECT_POINTS)
    x_tr = np.column_stack([np.zeros_like(t), t])
    return x, y, axes, x_grid, x_tr


# ---------------------------------------------------------------------------
# Approximations: (g, V) -> U_e under the Dirichlet and the recentred probit
# ---------------------------------------------------------------------------
def approximate(g: np.ndarray, V: np.ndarray) -> dict:
    """
    Both families on the same (g, V), via uqpfn.asymptotic with n_train = 1.
    ``ue`` is the raw residual H(g) - U_a_hat (never clipped).
    """
    d = asymptotic.entropy_decomposition_multiclass(g, V, 1)
    gc, a0 = d["g_n"], d["alpha0"]
    ua_k = gc * (digamma(a0 + 1.0)[:, None] - digamma(d["alpha"] + 1.0))
    out = {"dirichlet": {"ue": d["epistemic_raw"],
                         "ue_k": asymptotic.neg_p_log_p(gc) - ua_k,
                         "alpha0": a0,
                         "rbar": 1.0 / (a0 + 1.0)}}

    d = asymptotic.entropy_decomposition_multiclass_probit(g, V, 1, recenter=True)
    out["probit"] = {"ue": d["epistemic_raw"],
                     "ue_k": asymptotic.neg_p_log_p(d["g_n"]) - d["aleatoric_per_class"],
                     "m": d["m"], "s": d["s"]}
    return out


# ---------------------------------------------------------------------------
# Fit: the TabPFN trajectory over grid and transect
# ---------------------------------------------------------------------------
def _cache(args) -> RunCache:
    config = {"a": A, "h": H, "sigma": SIGMA, "n_per_class": N_PER_CLASS, "seed": SEED, "grid_points": GRID_POINTS,
              "transect_points": TRANSECT_POINTS, "margin": MARGIN, "n_estimators": N_ESTIMATORS}
    return RunCache(args.outdir / "multiclass_blops", config, version=CACHE_VERSION)


def run_fit(args) -> None:
    set_threads(args.threads)
    cache = _cache(args)
    cache.prepare(args.overwrite)
    x, y, _, x_grid, x_tr = design()
    key = rep_key(len(y), 0)
    if cache.has(key):
        print(f"[fit] cached: {cache.path(key)}")
        return
    from uqpfn import tabpfn
    import torch
    torch.manual_seed(SEED)
    t0 = time.time()
    perm = np.random.default_rng([SEED, 7]).permutation(len(y))  # data are class-ordered
    model = tabpfn.make_classifier(n_estimators=N_ESTIMATORS)
    traj = tabpfn.build_multiclass_trajectory(x[perm], y[perm], np.vstack([x_grid, x_tr]), n_classes=K,
                                              n_estimators=N_ESTIMATORS, model=model)
    cache.save(key, trajectory=traj)
    print(f"[fit] TabPFN trajectory: {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# Plot: benchmark, approximations, summary and figures
# ---------------------------------------------------------------------------
def _input_cols(g: np.ndarray, V: np.ndarray, suf: str) -> dict:
    r = normalised_variance(g, V)
    out = {}
    for k in range(K):
        out[f"g{k + 1}{suf}"] = g[:, k]
        out[f"V{k + 1}{suf}"] = V[:, k]
        out[f"r{k + 1}{suf}"] = r[:, k]
    return out


def _approx_cols(ap: dict, suf: str) -> dict:
    out = {f"rbar{suf}": ap["dirichlet"]["rbar"], f"alpha0{suf}": ap["dirichlet"]["alpha0"]}
    for m in METHODS:
        out[f"ue_{m}{suf}"] = ap[m]["ue"]
        for k in range(K):
            out[f"ue{k + 1}_{m}{suf}"] = ap[m]["ue_k"][:, k]
    for k in range(K):
        out[f"m{k + 1}_probit{suf}"] = ap["probit"]["m"][:, k]
        out[f"s{k + 1}_probit{suf}"] = ap["probit"]["s"][:, k]
    return out


def results_frame(x_eval: np.ndarray, ref: dict, approx: dict, tab: dict | None = None) -> pd.DataFrame:
    cols = {"x1": x_eval[:, 0], "x2": x_eval[:, 1], "ue_ref": ref["ue"], "ua_ref": ref["ua"]}
    cols.update({f"ue{k + 1}_ref": ref["ue_k"][:, k] for k in range(K)})
    cols.update(_input_cols(ref["g"], ref["V"], ""))
    cols.update(_approx_cols(approx, ""))
    if tab is not None:
        cols.update(_input_cols(tab["g"], tab["V"], TAB))
        cols.update(_approx_cols(tab["approx"], TAB))
    return pd.DataFrame(cols)


def per_class(df: pd.DataFrame, pattern: str) -> np.ndarray:
    """Stack the K columns named pattern.format(1..K) into (n, K)."""
    return np.column_stack([df[pattern.format(k + 1)].to_numpy() for k in range(K)])


def has_tabpfn(df: pd.DataFrame) -> bool:
    return f"g1{TAB}" in df.columns


def error_summary(ue_ref: np.ndarray, ue_hat: np.ndarray) -> dict:
    e = ue_hat - ue_ref
    return {"mean_ue": ue_hat.mean(), "bias": e.mean(), "mae": np.abs(e).mean(),
            "rel_L1": np.abs(e).sum() / ue_ref.sum(), "max_abs": np.abs(e).max(),
            "frac_neg": (ue_hat < 0).mean()}


def sign_check(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """rbar - r_k and U_e,k^Dir - U_e,k^ref on exact inputs, and how often their signs agree."""
    top = df["rbar"].to_numpy()[:, None] - per_class(df, "r{}")
    bot = per_class(df, "ue{}_dirichlet") - per_class(df, "ue{}_ref")
    rows = []
    for k in range(K):
        agree = np.sign(top[:, k]) == np.sign(bot[:, k])
        w = np.abs(bot[:, k])
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(top[:, k], bot[:, k])[0, 1]
        rows.append({"class": k + 1, "frac_points": agree.mean(),
                     "frac_abs_error": w[agree].sum() / max(w.sum(), np.finfo(float).tiny),
                     "corr": corr})
    return top, bot, pd.DataFrame(rows)


FONT_SCALE = 1.0   # set from --font-scale in run_plot()


def fs(pt: float) -> float:
    """Font size in points, scaled by --font-scale."""
    return pt * FONT_SCALE


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.size"] = fs(10.0)   # everything without an explicit size
    return plt


def _heat(ax, field, axes, cmap="viridis", vmin=None, vmax=None, data=None, title=""):
    a, b = axes
    im = ax.imshow(np.asarray(field).reshape(len(b), len(a)), origin="lower", cmap=cmap,
                   extent=(a[0], a[-1], b[0], b[-1]), vmin=vmin, vmax=vmax, aspect="equal")
    if data is not None:
        x, y = data
        for k in range(K):
            ax.scatter(*x[y == k].T, s=7, c=CLASS_COLORS[k], edgecolors="k",
                       linewidths=0.3, zorder=3)
    ax.set_title(title, fontsize=fs(9))
    ax.tick_params(labelsize=fs(7))
    return im


def _sym(*fields) -> float:
    lim = max(float(np.nanmax(np.abs(f))) for f in fields)
    return lim if lim > 0 else 1.0


METHOD_SHORT = {"dirichlet": r"\mathrm{Dir}", "probit": r"\mathrm{Pr}"}


def ue_vmax(df) -> float:
    """One colour-scale maximum for figs 1a and 1b: over U_e^ref and every approximation present."""
    cols = ["ue_ref"] + [c for m in METHODS for c in (f"ue_{m}", f"ue_{m}{TAB}") if c in df.columns]
    vmax = max(float(np.nanmax(df[c].to_numpy())) for c in cols)
    return vmax if vmax > 0 else 1.0


def plot_ue_overview(df, axes, data, suffix, input_label, vmax, out):
    """Figs 1a / 1b: U_e^ref | Dirichlet | probit on a scale shared by both figures; negatives white."""
    plt = _plt()
    import matplotlib
    fields = [df["ue_ref"].to_numpy()] + [df[f"ue_{m}{suffix}"].to_numpy() for m in METHODS]
    titles = [r"$U_e^{\mathrm{ref}}$"] + [rf"$U_e^{{{METHOD_SHORT[m]}}}$ ({input_label})"
                                          for m in METHODS]
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_under("white")
    n_neg = int(sum((f < 0).sum() for f in fields))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    for axi, f, title in zip(ax, fields, titles):
        im = _heat(axi, f, axes, cmap, 0.0, vmax, data, title)
        axi.set_xlabel("$x_1$", fontsize=fs(8))
    ax[0].set_ylabel("$x_2$", fontsize=fs(8))
    cb = fig.colorbar(im, ax=list(ax), shrink=0.85, extend="min" if n_neg else "neither")
    cb.set_label(r"$U_e$" + ("; white: $U_e < 0$" if n_neg else ""), fontsize=fs(8))
    cb.ax.tick_params(labelsize=fs(7))
    fig.savefig(out, dpi=160)
    plt.close(fig)


SIGN_CMAPS = ("PuOr_r", "RdBu_r")   # top / bottom row: both warm = positive, cool = negative


def plot_sign_check(df, axes, data, out):
    """Fig 2: rbar - r_k (top) against U_e,k^Dir - U_e,k^ref (bottom), exact inputs.
    The rows have different units, so each gets its own diverging colormap and colour bar."""
    plt = _plt()
    top, bot, _ = sign_check(df)
    lt, lb = _sym(top), _sym(bot)
    fig, ax = plt.subplots(2, K, figsize=(13.5, 8.2), constrained_layout=True)
    for k in range(K):
        im_t = _heat(ax[0, k], top[:, k], axes, SIGN_CMAPS[0], -lt, lt, data,
                     rf"$\bar r - r_{k + 1}$  (class {k + 1})")
        im_b = _heat(ax[1, k], bot[:, k], axes, SIGN_CMAPS[1], -lb, lb, data,
                     rf"$U_{{e,{k + 1}}}^{{\mathrm{{Dir}}}} - U_{{e,{k + 1}}}^{{\mathrm{{ref}}}}$")
        ax[1, k].set_xlabel("$x_1$", fontsize=fs(8))
    for im, row in ((im_t, 0), (im_b, 1)):
        cb = fig.colorbar(im, ax=list(ax[row]), shrink=0.9)
        cb.ax.tick_params(labelsize=fs(7))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_transect(df, h, out):
    """Fig 3: along x1 = 0. Solid: exact inputs; dotted: TabPFN."""
    plt = _plt()
    tab = has_tabpfn(df)
    t = df["x2"].to_numpy()
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    for k in range(K):
        ax[0].plot(t, df[f"g{k + 1}"], color=CLASS_COLORS[k], label=f"$g_{k + 1}$")
        ax[1].plot(t, df[f"r{k + 1}"], color=CLASS_COLORS[k], label=f"$r_{k + 1}$")
        if tab:
            ax[0].plot(t, df[f"g{k + 1}{TAB}"], color=CLASS_COLORS[k], ls=":", lw=1.3)
            ax[1].plot(t, df[f"r{k + 1}{TAB}"], color=CLASS_COLORS[k], ls=":", lw=1.3)
    ax[1].plot(t, df["rbar"], "k--", label=r"$\bar r$ (Dirichlet)")
    if tab:
        ax[1].plot(t, df[f"rbar{TAB}"], "k:", lw=1.3)
    ax[2].plot(t, df["ue_ref"], "k-", lw=2.2, label="reference")
    for m in METHODS:
        ax[2].plot(t, df[f"ue_{m}"], color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        if tab:
            ax[2].plot(t, df[f"ue_{m}{TAB}"], color=METHOD_COLORS[m], ls=":", lw=1.3)
    if tab:
        for axi in ax:
            axi.plot([], [], color="0.35", ls=":", lw=1.3, label="TabPFN inputs")
    ax[1].set_ylim(bottom=0.0)
    titles = ["prediction $g_k$", r"normalised variance $r_k$ and $\bar r$", "$U_e$"]
    for i, axi in enumerate(ax):
        for x0 in (0.0, h):
            axi.axvline(x0, color="0.6", lw=0.7, ls=":")
        axi.set_xlabel("$x_2$ along $x_1 = 0$ (1-2 boundary)")
        axi.set_title(titles[i] + ("\n(solid: exact, dotted: TabPFN)" if tab else ""), fontsize=fs(9))
        axi.grid(alpha=0.25)
        axi.legend(fontsize=fs(7))
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def run_plot(args) -> None:
    global FONT_SCALE
    FONT_SCALE = args.font_scale
    cache = _cache(args)
    traj = cache.load(cache.require()[0])["trajectory"]
    x, y, axes, x_grid, x_tr = design()
    n, ng = len(y), len(x_grid)

    # benchmark from S exact posterior draws, and an independent second set for its Monte Carlo error
    m, s = posterior(x, y, SIGMA, TAU, M0)
    mu = draw_centers(m, s, N_DRAWS, SEED)
    ref, ref_tr = reference(x_grid, mu, SIGMA, CHUNK), reference(x_tr, mu, SIGMA, CHUNK)
    ue2 = reference(x_grid, draw_centers(m, s, N_DRAWS, SEED + 1), SIGMA, CHUNK)["ue"]

    # TabPFN's g_n and V_n / n
    g_tab, v_tab = traj[-1], asymptotic.compute_vn_multiclass(traj) / n
    tab = {"g": g_tab[:ng], "V": v_tab[:ng], "approx": approximate(g_tab[:ng], v_tab[:ng])}
    tab_tr = {"g": g_tab[ng:], "V": v_tab[ng:], "approx": approximate(g_tab[ng:], v_tab[ng:])}
    grid = results_frame(x_grid, ref, approximate(ref["g"], ref["V"]), tab)
    transect = results_frame(x_tr, ref_tr, approximate(ref_tr["g"], ref_tr["V"]), tab_tr)

    rows = {"MC: 2nd benchmark": error_summary(ref["ue"], ue2)}
    for inputs, suf in (("exact", ""), ("TabPFN", TAB)):
        rows.update({f"{METHOD_LABELS[mm]}, {inputs}": error_summary(ref["ue"], grid[f"ue_{mm}{suf}"].to_numpy())
                     for mm in METHODS})
    summary = pd.DataFrame(rows).T.rename_axis("comparison")
    summary.to_csv(cache.folder / "summary.csv", float_format="%.6g")
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(summary)

    data = (x, y)
    vmax = ue_vmax(grid)                    # shared by both U_e figures
    out = cache.folder
    plot_ue_overview(grid, axes, data, "", "exact input", vmax, out / "ue_exact.png")
    plot_transect(transect, H, out / "transect.png")
    plot_sign_check(grid, axes, data, out / "sign_check.png")
    plot_ue_overview(grid, axes, data, TAB, "TabPFN input", vmax, out / "ue_tabpfn.png")
    print(f"[plot] {out}: ue_exact.png, transect.png, sign_check.png, ue_tabpfn.png, summary.csv")


def main() -> None:
    args = parse_args()
    if args.run in ("fit", "all"):
        run_fit(args)
    if args.run in ("plot", "all"):
        run_plot(args)


if __name__ == "__main__":
    main()
