"""
Shared pieces of the regression coverage experiments (scripts/coverage/).

    DGPs           linear and dependent DGP of Fortini et al. (2026a), App. F.2 (thesis App. A.1.1),
                   used by linear_dependent.py and n_estimators_linear.py
    spline         spline DGP with its exact Gaussian posterior (thesis App. A.1.1),
                   used by spline.py and n_estimators_spline.py
    CDF helpers    response grid with the threshold inserted, CDF inversion, AMP quantile intervals

The Poisson DGP is only used by poisson_probit.py and lives there.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import SplineTransformer

# spline DGP (thesis App. A.1.1): defaults of the --spline-* arguments
SPLINE_DEFAULTS = {
    "sigma": 0.45,                      # noise sd
    "tau": 1.1,                         # prior sd of the coefficients
    "n_basis": 20,
    "degree": 3,
    "y_grid_size": 241,                 # response grid size
}
# fixed by the design of the experiments (cached arrays and the unbalanced design are built around them)
TARGETS = {"median": 0.5, "q90": 0.9}   # quantile level q of y | x
GAP = (-1.0, 1.0)                       # sparsely sampled x-region of the unbalanced design


def add_spline_args(parser: argparse.ArgumentParser) -> None:
    """--spline-sigma, --spline-tau, --spline-n-basis, --spline-degree, --spline-y-grid-size."""
    d = SPLINE_DEFAULTS
    parser.add_argument("--spline-sigma", type=float, default=d["sigma"], help="noise sd (default: %(default)s)")
    parser.add_argument("--spline-tau", type=float, default=d["tau"],
                        help="prior sd of the spline coefficients (default: %(default)s)")
    parser.add_argument("--spline-n-basis", type=int, default=d["n_basis"],
                        help="number of B-spline basis functions (default: %(default)s)")
    parser.add_argument("--spline-degree", type=int, default=d["degree"], help="B-spline degree (default: %(default)s)")
    parser.add_argument("--spline-y-grid-size", type=int, default=d["y_grid_size"],
                        help="size of the response grid (default: %(default)s)")


def spline_settings(args: argparse.Namespace) -> dict:
    """The spline DGP settings of a run, as keyword arguments of make_spline_data (and for the cache config)."""
    return {k: getattr(args, f"spline_{k}") for k in SPLINE_DEFAULTS}


# ---------------------------------------------------------------------------
# Linear and dependent DGP (Fortini et al. 2026a, App. F.2): X ~ U(-10, 10)
# ---------------------------------------------------------------------------
def generate_linear(n: int, rng: np.random.Generator):
    """Y = 0.2 X + N(0, 1)."""
    X = rng.uniform(-10, 10, size=(n, 1))
    Y = 0.2 * X[:, 0] + rng.standard_normal(n)
    return X, Y


def true_prob_linear(X_eval: np.ndarray, t: float = 0.0):
    """P(Y <= t | X = x) = Phi(t - 0.2 x)."""
    return norm.cdf(t - 0.2 * X_eval[:, 0])


def generate_dependent(n: int, rng: np.random.Generator):
    """Y | X = x ~ N(0.5 x + 1, (0.5 + 0.5 |x|)^2)."""
    X = rng.uniform(-10, 10, size=(n, 1))
    mu = 0.5 * X[:, 0] + 1
    sigma = 0.5 + 0.5 * np.abs(X[:, 0])
    Y = mu + sigma * rng.standard_normal(n)
    return X, Y


def true_prob_dependent(X_eval: np.ndarray, t: float = 0.0):
    mu = 0.5 * X_eval[:, 0] + 1
    sigma = 0.5 + 0.5 * np.abs(X_eval[:, 0])
    return norm.cdf((t - mu) / sigma)


DGPS = {
    "linear": (generate_linear, true_prob_linear),
    "dependent": (generate_dependent, true_prob_dependent),
}


def threshold_grid(y_train: np.ndarray, size: int, t: float) -> np.ndarray:
    """Response grid spanning the labels +- 4 sd, with t inserted so it is hit exactly (thesis App. A.3)."""
    y_pad = 4.0 * np.std(y_train)
    y_grid = np.linspace(y_train.min() - y_pad, y_train.max() + y_pad, size)
    return np.sort(np.append(y_grid, t))


# ---------------------------------------------------------------------------
# Spline DGP and its exact posterior
# ---------------------------------------------------------------------------
def build_design(
    x_fit: np.ndarray,
    x_eval: np.ndarray,
    n_basis: int,
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    d = x_fit.shape[1]
    blocks_f, blocks_e = [], []
    for j in range(d):
        st = SplineTransformer(
            degree=degree,
            n_knots=n_basis - degree + 1,
            include_bias=False,
        )
        st.fit(x_fit[:, [j]])
        bf = st.transform(x_fit[:, [j]])
        be = st.transform(x_eval[:, [j]])
        cm = bf.mean(0, keepdims=True)
        blocks_f.append(bf - cm)
        blocks_e.append(be - cm)
    z_f = np.c_[np.ones(len(x_fit)), np.hstack(blocks_f)]
    z_e = np.c_[np.ones(len(x_eval)), np.hstack(blocks_e)]
    return z_f, z_e


def gaussian_posterior(
    z: np.ndarray,
    y: np.ndarray,
    sigma: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    p = z.shape[1]
    lam = (sigma / tau) ** 2
    a = z.T @ z + lam * np.eye(p)
    mu = np.linalg.solve(a, z.T @ y)
    sigma_post = sigma ** 2 * np.linalg.inv(a)
    return mu, sigma_post


def analytic_ci(
    z_eval: np.ndarray,
    mu: np.ndarray,
    sigma_post: np.ndarray,
    sigma: float,
    q: float = 0.5,
    level: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    center = z_eval @ mu + norm.ppf(q) * sigma
    std_f = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", z_eval, sigma_post, z_eval), 0.0))  # posterior sd of f
    half = norm.ppf(0.5 + level / 2.0) * std_f
    return center - half, center + half


def make_spline_data(
    *,
    mode: str,
    seed: int,
    n_train: int,
    n_eval: int = 100,
    sigma: float = SPLINE_DEFAULTS["sigma"],
    tau: float = SPLINE_DEFAULTS["tau"],
    n_basis: int = SPLINE_DEFAULTS["n_basis"],
    degree: int = SPLINE_DEFAULTS["degree"],
    y_grid_size: int = SPLINE_DEFAULTS["y_grid_size"],
) -> dict:
    """
    One dataset of the spline DGP. mode 'unbalanced': 60% / 10% / 30% of the points left of, in and right
    of the gap [-1, 1]; 'balanced': x ~ U(-2.5, 2.5).
    """
    rng = np.random.default_rng(seed)

    if mode == "balanced":
        x_train = rng.uniform(-2.5, 2.5, size=n_train).reshape(-1, 1)
    else:
        n1 = int(n_train * 0.6)
        n2 = int(n_train * 0.1)
        n3 = n_train - n1 - n2
        x_train = np.concatenate([
            rng.uniform(-2.5, -1.0, size=n1),
            rng.uniform(-1.0, 1.0, size=n2),
            rng.uniform(1.0, 2.5, size=n3),
        ])
        rng.shuffle(x_train)
        x_train = x_train.reshape(-1, 1)
    x_eval = np.linspace(-2.5, 2.5, n_eval).reshape(-1, 1)

    # theta ~ N(0, tau^2 I), intercept included: exactly the prior of gaussian_posterior, so the analytic
    # posterior is the Bayes-optimal posterior of the DGP
    z_tr, z_ev = build_design(x_train, x_eval, n_basis=n_basis, degree=degree)
    theta_true = tau * rng.standard_normal(z_tr.shape[1])
    f_train = z_tr @ theta_true
    f_eval = z_ev @ theta_true
    y_train = f_train + sigma * rng.standard_normal(len(x_train))
    mu_train, sigma_train = gaussian_posterior(z_tr, y_train, sigma=sigma, tau=tau)

    y_pad = 4.0 * sigma
    y_grid = np.linspace(y_train.min() - y_pad, y_train.max() + y_pad, y_grid_size)

    out = {
        "x_train": x_train,
        "y_train": y_train,
        "x_eval": x_eval,
        "z_ev": z_ev,
        "y_grid": y_grid,
        "sigma": sigma,
        "mu_train": mu_train,
        "sigma_train": sigma_train,
    }
    # true quantile of y | x, and the one under the optimal posterior (posterior mean of f + z_q sigma)
    for t, q in TARGETS.items():
        out[f"true_{t}"] = f_eval + norm.ppf(q) * sigma
        out[f"posterior_{t}"] = z_ev @ mu_train + norm.ppf(q) * sigma
    return out


# ---------------------------------------------------------------------------
# CDF helpers
# ---------------------------------------------------------------------------
def invert_cdf(
    y_grid: np.ndarray,
    cdf: np.ndarray,
    probs: np.ndarray,
) -> np.ndarray:
    cdf_m = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0), axis=0)
    out = np.empty((len(probs), cdf.shape[1]), dtype=float)
    for j in range(cdf.shape[1]):
        for i, p in enumerate(probs):
            idx = int(np.searchsorted(cdf_m[:, j], p, side="left"))
            out[i, j] = y_grid[min(max(idx, 0), len(y_grid) - 1)]
    return out


def median_ci_from_resamples(
    res_list: list[dict],
    y_grid: np.ndarray,
    lo: float = 5.0,
    hi: float = 95.0,
    q: float = 0.5,
) -> np.ndarray:
    """Percentile interval over the AMP posterior draws of the q-quantile (q = 0.5: median)."""
    y1d = np.asarray(y_grid).ravel()
    out = []
    for entry in res_list:
        cdf = entry["cdf"]
        qb = [np.interp(q, cdf[:, b], y1d) for b in range(cdf.shape[1])]
        out.append(np.percentile(qb, [lo, hi]))
    return np.stack(out)
