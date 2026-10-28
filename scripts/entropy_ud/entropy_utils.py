"""
Shared pieces of the binary entropic uncertainty-decomposition experiments (scripts/entropy_ud/).

    trajectory   TabPFN classifier trajectory on randomly permuted rows, and (g_n, V_n) read off its
                 first n rows
    decompose    the binary entropy split under the moment-matched Beta or the recentred probit,
                 through uqpfn.asymptotic (clip_eps = 1e-12, 257-node Gauss-Legendre for the probit)

Used by replication_logistic.py, replication_moons.py, logistic_optimum.py and frequency.py. The multiclass
scripts call the multiclass functions of uqpfn.asymptotic directly.
"""
from __future__ import annotations

import numpy as np

from scripts.common import n_estimators
from uqpfn import asymptotic

N_ESTIMATORS = n_estimators(64)                     # all classification experiments (thesis App. A.1.2)
METHODS = ("beta", "probit_recentred")


def permuted_trajectory(x_train, y_train, x_eval, *, model, rng: np.random.Generator) -> np.ndarray:
    """(n_train, n_eval): g_k(x*) on the expanding prefixes of the randomly permuted rows."""
    from uqpfn import tabpfn
    idx = rng.permutation(len(y_train))
    return tabpfn.build_classifier_trajectory(x_train[idx], y_train[idx], x_eval, n_estimators=N_ESTIMATORS,
                                              model=model)


def gn_vn(traj: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """g_n and V_n read off the first n rows of a trajectory (a trajectory of the nested dataset D_n)."""
    sub = np.asarray(traj)[:n]
    return sub[-1], asymptotic.compute_vn_pointwise(sub)


def decompose(g_n: np.ndarray, v_n: np.ndarray, n: int, method: str) -> dict:
    """
    Split h(g_n) into aleatoric + epistemic with an approximating law for g_inf(x*) | z_1:n with mean g_n and
    variance v_n / n (n = 1 hands over a variance directly). Returns the uqpfn.asymptotic dict: ``total``,
    ``aleatoric``, ``epistemic`` and the unclipped ``epistemic_raw``, among others.
    """
    if method == "beta":
        return asymptotic.entropy_decomposition(g_n, v_n, n)
    if method == "probit_recentred":
        return asymptotic.entropy_decomposition_probit(g_n, v_n, n, recenter=True)
    raise ValueError(f"unknown method {method!r}")
