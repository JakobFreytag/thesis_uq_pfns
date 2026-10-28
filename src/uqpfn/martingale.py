"""
Martingale posterior credible intervals.

The core algorithm (``get_posterior``) runs a vectorised martingale-posterior
update given *pre-computed* conditional CDF evaluations on a y-grid and a
copula correlation parameter rho.  It is model-agnostic: any predictor
can provide the CDF input.

Typical workflow
----------------
1. Compute ``cdf_conditionals`` (y_grid_size × n_eval) for your model of choice.
   For TabPFN, use the :mod:`uqpfn.tabpfn` helpers.
2. Call ``get_posterior(...)`` with either an explicit ``alpha_schedule`` or
   ``alpha_dimension=d``.
3. Extract intervals with ``median_intervals_from_resamples``.

Alpha schedules
---------------
``make_alpha_schedules(d)`` returns the two supported learning-rate choices:
- ``default``: ``beta = 0.5 + 2 / (1.1 * d + 4)``, ``C = 2 ** beta``
- ``fong``: ``C = 2``, ``beta = 1``
The default copula correlation is ``rho = 0.99``.
"""

from __future__ import annotations

import numpy as np
import pyvinecopulib as pv

DEFAULT_B_POSTSAMPLES = 50
DEFAULT_T_FWDSAMPLES = 500
DEFAULT_RHO = 0.99
FONG_ALPHA_C = 2.0
FONG_ALPHA_BETA = 1.0


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def get_posterior(
    y_grid: np.ndarray,
    n_train: int,
    B_postsamples: int,
    T_fwdsamples: int,
    cdf_conditionals: np.ndarray,
    rho: float = DEFAULT_RHO,
    seed: int = 100,
    alpha_schedule=None,
    alpha_dimension: int | None = None,
    n_eff=None,
    shared_uniform: bool = True,
    use_blowup: bool = True,
) -> list[dict]:
    """
    Run the martingale posterior update and return posterior samples.

    Parameters
    ----------
    y_grid : (y_grid_size,) or (y_grid_size, 1) array
        y-values at which the conditional CDF is evaluated (the support grid).
    n_train : int
        Number of observed training points (burn-in length).
    B_postsamples : int
        Number of independent posterior draws (repetitions).
    T_fwdsamples : int
        Number of forward steps beyond the training burn-in.
    cdf_conditionals : (y_grid_size, n_eval) array
        Pre-computed conditional CDF evaluations of the predictor on y_grid
        for each of the n_eval test points.
    rho : float
        Gaussian copula serial correlation parameter.
    seed : int
        Base RNG seed.
    alpha_schedule : callable(i) -> float, optional
        Learning-rate schedule. If omitted, callers must provide
        ``alpha_dimension`` so the dimension-adaptive default
        ``C * (i + 1)^(-beta)`` can be constructed with
        ``beta = 0.5 + 2 / (1.1 * d + 4)`` and ``C = 2 ** beta``.
        Use ``make_alpha_schedules`` to obtain the supported choices.
    alpha_dimension : int, optional
        Feature dimension ``d`` used by the default martingale schedule when
        ``alpha_schedule`` is omitted.
    n_eff : None, scalar, or (n_eval,) array, optional
        Effective sample size for each evaluation point. When provided,
        callers may use it inside ``alpha_schedule`` by defining a callable
        that accepts two arguments ``alpha_schedule(k, n_eff)`` where ``k``.  ### didnt overwrite k for y_grid_size in here, look at alpha schedula logic first! 
        is the forward-step index starting at 0. Existing one-argument
        schedules ``alpha_schedule(i)`` continue to work unchanged.
    shared_uniform : bool
        If True, a single uniform is drawn per step and shared across all
        test points (faster, matches the original implementation).
    use_blowup : bool
        If True, scale alpha by the finite-horizon blowup factor
        ``blowup_integral_inf(n_train, d, T_fwdsamples)`` when
        ``alpha_dimension`` is provided. If False, do not apply this scaling.

    Returns
    -------
    list of dicts, one per test point (length n_eval).
    Each dict has:
        ``"ysim"`` — (T_total, B_postsamples) array of simulated y values
        ``"cdf"``  — (y_grid_siz, B_postsamples) array of updated CDF vectors
    """
    cdf_conditionals = np.asarray(cdf_conditionals, dtype=float)
    y_grid_size, n_eval = cdf_conditionals.shape
    support = np.asarray(y_grid).ravel()
    if n_eff is None:
        n_eff_arr = None
    else:
        n_eff_arr = np.asarray(n_eff, dtype=float)
        if n_eff_arr.ndim == 0:
            n_eff_arr = np.full(n_eval, float(n_eff_arr))
        elif n_eff_arr.shape != (n_eval,):
            raise ValueError(
                f"n_eff must be scalar or shape ({n_eval},), got {n_eff_arr.shape}"
            )

    if alpha_schedule is None:
        if alpha_dimension is None:
            raise ValueError(
                "alpha_dimension is required when alpha_schedule is omitted; "
                "get_posterior no longer assumes d=1."
            )
        if alpha_dimension <= 0:
            raise ValueError(f"alpha_dimension must be positive, got {alpha_dimension}")

    rho_arr = np.array([[float(rho)]], dtype="float64")
    cop = pv.Bicop(family=pv.BicopFamily.gaussian, rotation=0, parameters=rho_arr)
    eps = 1e-6

    N = T_fwdsamples + n_train
    ysim_all = np.zeros((n_eval, N, B_postsamples))
    cdf_all = np.zeros((n_eval, y_grid_size, B_postsamples))

    if use_blowup and alpha_dimension is not None:
        alpha_blowup = blowup_integral(n_train, alpha_dimension, T_fwdsamples)
    else:
        alpha_blowup = 1.0

    for b in range(B_postsamples):
        np.random.seed(seed + b)
        cdf_b = np.clip(cdf_conditionals.T.copy(), eps, 1.0 - eps)  # (n_eval, y_grid_size)
        ysim = np.zeros((n_eval, N))

        for i in range(n_train, N):
            if alpha_schedule is None:
                alpha = default_alpha_schedule(i, alpha_dimension)
            else:
                alpha = alpha_schedule(i)
            alpha = np.asarray(alpha, dtype=float)
            if alpha.ndim == 0:
                alpha = np.full((n_eval, 1), float(alpha))
            elif alpha.shape == (n_eval,):
                alpha = alpha[:, None]
            elif alpha.shape != (n_eval, 1):
                raise ValueError(
                    "alpha_schedule must return a scalar, shape (n_eval,), or "
                    f"shape (n_eval, 1); got {alpha.shape}"
                )

            #alpha = alpha * alpha_blowup (this one doesnt )
            alpha = np.minimum(1.0, alpha * alpha_blowup)
            if shared_uniform:
                u_val = np.random.uniform()
                u2 = np.full((n_eval, y_grid_size), u_val)
            else:
                u_vals = np.random.uniform(size=n_eval)
                u2 = np.repeat(u_vals[:, None], y_grid_size, axis=1)

            u = np.stack([np.clip(cdf_b, eps, 1.0 - eps), np.clip(u2, eps, 1.0 - eps)], axis=-1).reshape(-1, 2)
            h = cop.hfunc2(u).reshape(n_eval, y_grid_size)
            cdf_b = (1 - alpha) * cdf_b + alpha * h
            cdf_b /= np.maximum(np.max(cdf_b, axis=1, keepdims=True), eps)
            cdf_b = np.clip(cdf_b, eps, 1.0 - eps)

            if shared_uniform:
                idx = np.array([np.searchsorted(cdf_b[j], u_val) for j in range(n_eval)])
                idx = np.clip(idx, 0, y_grid_size - 1)[:, None]
            else:
                idx = np.searchsorted(cdf_b, u_vals[:, None])
                idx = np.clip(idx, 0, y_grid_size - 1)

            ysim[:, i] = support[idx].squeeze()

        ysim_all[:, :, b] = ysim
        cdf_all[:, :, b] = cdf_b

    return [{"ysim": ysim_all[j], "cdf": cdf_all[j]} for j in range(n_eval)]


# ---------------------------------------------------------------------------
# Alpha schedules
# ---------------------------------------------------------------------------

def blowup_integral(n, d, T):
    gamma = 4.0 / (1.1 * d + 4.0)
    return 1.0 / np.sqrt(1.0 - (1.0 + T / n) ** (-gamma))

def _build_alpha_schedule(C: float, beta: float):
    return lambda i: C * (i + 1) ** (-beta)

def make_alpha_schedules(d: int) -> dict:
    """
    Return the supported alpha schedules for feature dimension *d*.

    Schedules
    ---------
    default : ``beta = 0.5 + 2 / (1.1 * d + 4)``, ``C = 2 ** beta``
    fong    : ``C = 2``, ``beta = 1``

    Returns
    -------
    dict mapping name -> {"C": float, "beta": float, "fn": callable(i) -> float}
    """
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")

    beta_default = 0.5 + 2 / (1.1 * d + 4.0)

    params = {
        "default": (2 ** beta_default, beta_default),
        "fong": (FONG_ALPHA_C, FONG_ALPHA_BETA),
    }

    return {
        name: {"C": C, "beta": beta, "fn": _build_alpha_schedule(C, beta)}
        for name, (C, beta) in params.items()
    }


def default_alpha_schedule(i: int, d: int) -> float:
    """
    Return the dimension-adaptive default martingale learning rate.
    """
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    beta = 0.5 + 2 / (1.1 * d + 4.0)
    C = 2 ** beta
    return C * (i + 1) ** (-beta)


def fong_alpha_schedule(i: int) -> float:
    """Return the Fong et al. learning rate ``2 * (i + 1)^(-1)``."""
    return FONG_ALPHA_C * (i + 1) ** (-FONG_ALPHA_BETA)


# ---------------------------------------------------------------------------
# Interval extraction
# ---------------------------------------------------------------------------

def median_intervals_from_resamples(
    res_list: list[dict],
    y_grid: np.ndarray,
    lo: float = 5.0,
    hi: float = 95.0,
) -> np.ndarray:
    """
    Extract credible intervals on the posterior median from ``get_posterior``
    output.

    For each test point the function interpolates the median from each
    posterior CDF draw, then returns the [lo, hi] percentile bounds across
    draws.

    Parameters
    ----------
    res_list : list of dicts — output of ``get_posterior``
    y_grid   : (y_grid_size,) or (y_grid_size, 1) array — same grid passed to ``get_posterior``
    lo, hi   : float — percentile bounds (default: 5th and 95th)

    Returns
    -------
    intervals : (n_eval, 2) array  — columns are [lower, upper]
    """
    y_grid_1d = np.asarray(y_grid).ravel()
    out = []
    for entry in res_list:
        # entry["cdf"] shape: (y_grid_size, B_postsamples)
        medians = [
            np.interp(0.5, entry["cdf"][:, b], y_grid_1d)
            for b in range(entry["cdf"].shape[1])
        ]
        out.append(np.percentile(medians, [lo, hi]))
    return np.stack(out)  # (n_eval, 2)
