import numpy as np
from scipy.stats import norm
from scipy.special import digamma  # noqa: E402  (move to the module header)


def compute_vn(trajectory: np.ndarray, type="pointwise") -> np.ndarray:
    """
    Computes the V_n asymptotic variance estimator from a trajectory of predictive CDFs.
    
    Parameters
    ----------
    trajectory: (n_train, y_grid_size, n_eval) array
        The history of evaluated CDFs on the y_grid.
        n_train is the prefix length, y_grid_size is the size of the grid along the y-axis, n_eval is the test points.
    type: str
        The type of variance to compute: "pointwise" or "simultaneous".
        
    Returns
    -------
    V_n: array
        The estimated asymptotic variance for the CDF at each grid point and eval point[cite: 112].
        If type == "pointwise", shape is (y_grid_size, n_eval).
        If type == "simultaneous", shape is (y_grid_size, n_eval, n_eval).
    """
    n_prefix = trajectory.shape[0]
    
    # Delta_k = F_k - F_{k-1}
    # deltas will have shape (n_train - 1, y_grid_size, n_eval)
    # deltas[0] corresponds to F_2 - F_1 (so k = 2)
    deltas = np.diff(trajectory, axis=0)
    
    # array of k values starting from 2 up to n_prefix
    # Shape: (n_train - 1,)
    k_values = np.arange(2, n_prefix + 1) 
    
    # reshape, shape becomes (n_train - 1, 1, 1)
    k_weights = k_values[:, None, None]
    
    # k * Delta_k 
    # Shape: (n_train - 1, K, n_eval)
    kdiff = k_weights * deltas
    
    if type == "pointwise": # (only need variance, not covariance)
        # V_n = (1/n) * sum(k^2 * Delta_k^2)
        v_n = np.sum(kdiff ** 2, axis=0)
        
        # Divide by total prefix size n_k 
        v_n = v_n / n_prefix
        return v_n
        
    elif type == "simultaneous":
        # V_n = (1/n) * sum k^2 * Delta_k Delta_k^T # outer product across n_eval dimension (notice x only)
        outer = np.einsum("ijk,ijl->ijkl", kdiff, kdiff)
        
        # Vectorized sum over the prefix dimension (axis 0)
        v_n = np.sum(outer, axis=0)
        
        # Divide by total prefix size n_k
        v_n = v_n / n_prefix
        return v_n

def pointwise_cdf_bounds(
    prediction_cdf: np.ndarray, 
    vn_variance: np.ndarray, 
    n_train: int, 
    alpha: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constructs pointwise asymptotic credible bounds for the CDF using V_n, using asymptotic normality.
    
    Parameters
    ----------
    prediction_cdf : (y_grid_size, n_eval) array
        The final predicted CDF (usually F_n).
    vn_variance : (y_grid_size, n_eval) array
        The computed V_n variance.
    n_train : int
        The total number of training samples.
    alpha : float
        The significance level (default 0.1 for a 90% interval).
        
    Returns
    -------
    lower_cdf, upper_cdf : (y_grid_size, n_eval) arrays
        The bounded CDFs, clipped between 0 and 1.
    """

    z_val = norm.ppf(1 - alpha / 2)
    margin = z_val * np.sqrt(vn_variance / n_train)
    
    # Apply bounds and clip strictly to valid probability space [0, 1]
    lower_cdf = np.clip(prediction_cdf - margin, 0.0, 1.0)
    upper_cdf = np.clip(prediction_cdf + margin, 0.0, 1.0)
    
    return lower_cdf, upper_cdf


_TINY = np.finfo(float).tiny
 
 
# ---------------------------------------------------------------------------
# V_n for classification trajectories
# ---------------------------------------------------------------------------
 
def compute_vn_pointwise(trajectory: np.ndarray) -> np.ndarray:
    """
    V_n for a *classification* trajectory g_1(x*), ..., g_n(x*).
 
    Eq. (2) of the paper at gamma = 1:
        V_n = (1/n) * sum_k k^2 * Delta_k Delta_k^T,   Delta_k = g_k - g_{k-1}
 
    In binary classification there is no y-grid: the "event" is A = {1} and the
    predictive is the scalar g_k(x) = P(Y=1 | x, z_1:k). We therefore reuse
    ``compute_vn`` by inserting a singleton grid axis, so the estimator is
    literally the same code path used for the regression CDFs.
 
    Note on the summation range: eq. (2) sums from k = 1, i.e. it includes
    Delta_1 = g_1 - g_0 with g_0 the prior predictive on an empty context.
    Appendix B prescribes *non-empty* prefixes, so the observable trajectory
    starts at g_1 and the first available update is Delta_2. ``compute_vn``
    therefore sums k = 2..n while still dividing by n, which is the only
    feasible reading of eq. (2) under Appendix B.
 
    Parameters
    ----------
    trajectory : (n_train, n_eval) array
        History of g_k evaluated at each test point, k = 1, ..., n_train.
 
    Returns
    -------
    v_n : (n_eval,) array
        Diagonal entry v_n(x*) of V_n for each test point.
    """
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim != 2:
        raise ValueError(f"trajectory must be (n_train, n_eval); got {traj.shape}")
    # (n_train, 1, n_eval) -> compute_vn -> (1, n_eval) -> (n_eval,)
    return compute_vn(traj[:, None, :], type="pointwise")[0]
 
 
# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------
 
def binary_entropy(p: np.ndarray) -> np.ndarray:
    """h(p) = -p log p - (1-p) log(1-p), with 0 log 0 := 0."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    out = np.zeros_like(p)
    mask = (p > 0.0) & (p < 1.0)
    pm = p[mask]
    out[mask] = -pm * np.log(pm) - (1.0 - pm) * np.log1p(-pm)
    return out
 
 
def shrink_probability(g: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Map (0, 1) -> [eps, 1-eps] by the affine shrinkage g -> eps + (1-2 eps) g.
 
    Both decompositions need g strictly interior: the Beta path needs
    g(1-g) > 0 for T_n, and the probit path needs Phi^{-1}(g) finite. This is
    not a defect being papered over — the CLT itself is stated a.s. on
    {0 < P~(x,{1}) < 1}, so an interior g is an assumption of the theorem, and
    the shrinkage is only guarding against a *numerically* saturated g_n (a
    softmax underflowing in float32) rather than a genuine boundary value.
 
    A hard ``np.clip`` would also produce an interior g, but it is flat at the
    boundary, so it collapses the ordering of everything it touches and has
    Jacobian 0 exactly where the probit delta method needs one. The affine
    shrinkage is smooth, strictly monotone, has constant Jacobian (1 - 2 eps),
    and moves any interior g by less than 2 eps.
 
    Choose eps well below the smallest g_n you actually observe. It is NOT
    scale-free: at eps = 1e-6 a genuine g_n = 1e-7 is pulled up to ~1.1e-6,
    moving eta = Phi^{-1}(g) from -5.20 to -4.73 — a material distortion. The
    default 1e-12 only has to beat exact saturation, so it leaves any g_n above
    ~1e-10 untouched. ``probit_shrinkage_report`` quantifies the damage on real
    inputs.
    """
    return eps + (1.0 - 2.0 * eps) * np.asarray(g, dtype=float)
 
 
# ---------------------------------------------------------------------------
# Beta path (paper): Appendix L.1.1
# ---------------------------------------------------------------------------
 
def beta_moment_match(
    g_n: np.ndarray,
    v_n: np.ndarray,
    n_train: int,
    clip_eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match a Beta(alpha_n, beta_n) to the predictive-CLT mean and variance.
 
        sigma_n^2(x*) = min( v_n(x*) / n ,  (1-eps) * g_n (1 - g_n) )
        T_n(x*)       = g_n (1 - g_n) / sigma_n^2 - 1
        alpha_n       = g_n T_n,     beta_n = (1 - g_n) T_n
 
    The variance is clipped *strictly* below g_n(1-g_n) so that T_n > 0, as
    required in Appendix L.1.1. Note this clip binds whenever the CLT variance
    exceeds the largest variance any distribution on [0,1] with mean g_n can
    have — i.e. the Beta family cannot represent the CLT's uncertainty there,
    and the estimate is capped rather than matched.
 
    Returns
    -------
    alpha_n, beta_n : (n_eval,) arrays
    """
    g = shrink_probability(g_n, clip_eps)
    var_cap = g * (1.0 - g)
 
    sigma2 = np.minimum(np.asarray(v_n, dtype=float) / float(n_train),
                        (1.0 - clip_eps) * var_cap)
    sigma2 = np.maximum(sigma2, _TINY)
 
    T_n = var_cap / sigma2 - 1.0
    return g * T_n, (1.0 - g) * T_n
 
 
def entropy_decomposition(
    g_n: np.ndarray,
    v_n: np.ndarray,
    n_train: int,
    clip_eps: float = 1e-12,
) -> dict:
    """
    Entropic UD for binary classification, Beta path (Appendix L.1, eq. 13).
 
        Total      H  = h(g_n(x*))
        Aleatoric  Ua = -a/(a+b) psi(a+1) - b/(a+b) psi(b+1) + psi(a+b+1)
        Epistemic  Ue = H - Ua
 
    Because the Beta is moment-matched, E[G] = g_n exactly, so Jensen gives
    Ua <= h(E[G]) = H and hence Ue >= 0 by construction. ``epistemic_raw``
    is kept unclipped so that violations (which should only ever be
    floating-point noise here) stay visible.
 
    Returns
    -------
    dict with keys ``total``, ``aleatoric``, ``epistemic``, ``epistemic_raw``,
    ``mean_g``, ``alpha``, ``beta``, ``g_n``, ``v_n`` — each (n_eval,).
    """
    g = shrink_probability(g_n, clip_eps)
    a, b = beta_moment_match(g_n, v_n, n_train, clip_eps=clip_eps)
    ab = a + b
 
    total = binary_entropy(g)
    aleatoric = (
        -(a / ab) * digamma(a + 1.0)
        - (b / ab) * digamma(b + 1.0)
        + digamma(ab + 1.0)
    )
    epistemic_raw = total - aleatoric
 
    # The internally coherent total for a decomposition is the entropy of the
    # mean of ITS OWN approximating law, h(E[g~]) -- that is the quantity for
    # which total - aleatoric is a mutual information and is >= 0 by Jensen.
    # The Beta is moment-matched, so E[g~] = g_n and the two totals coincide
    # exactly; the field is kept for symmetry with the probit path, where they
    # do not.
    mean_g = a / ab
    total_selfconsistent = binary_entropy(mean_g)
 
    return {
        "total": total,
        "total_selfconsistent": total_selfconsistent,
        "aleatoric": aleatoric,
        "epistemic": np.maximum(epistemic_raw, 0.0),
        "epistemic_raw": epistemic_raw,
        "epistemic_selfconsistent": total_selfconsistent - aleatoric,
        "mean_g": mean_g,                     # exactly g by moment matching
        "alpha": a,
        "beta": b,
        "g_n": np.asarray(g_n, dtype=float),
        "v_n": np.asarray(v_n, dtype=float),
    }
 
 
# ---------------------------------------------------------------------------
# Probit-normal path: decompose on the real line
# ---------------------------------------------------------------------------
 
def probit_moment_match(
    g_n: np.ndarray,
    v_n: np.ndarray,
    n_train: int,
    recenter: bool = False,
    clip_eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Push the predictive CLT through the probit link and return the Gaussian
    (m, s) on R such that g~ = Phi(eta), eta ~ N(m, s^2).
 
    Delta method. With eta_n = Phi^{-1}(g_n) and d Phi^{-1}(p)/dp = 1/phi(eta),
 
        s^2 = (v_n / n) * (1 - 2 eps)^2 / phi(eta_n)^2
 
    (the (1-2 eps)^2 is the Jacobian of the shrinkage, negligible but exact).
 
    Unlike the Beta path, no cap on s is needed: *any* s > 0 gives a valid law
    on (0,1), so a CLT variance exceeding g_n(1-g_n) is represented faithfully
    instead of being clipped. That is the main structural advantage of working
    on R.
 
    Recentring. Phi is not affine, so E[Phi(eta)] != Phi(m). The probit-normal
    mean is available in closed form,
 
        E[Phi(m + sZ)] = Phi( m / sqrt(1 + s^2) ),
 
    which is shrunk towards 1/2 relative to Phi(m). Setting
 
        m* = eta_n * sqrt(1 + s^2)
 
    therefore makes E[g~] = g_n exactly, restoring the first-moment match the
    Beta path has by construction — and with it the Jensen guarantee Ue >= 0.
    With ``recenter=False`` the mean drifts towards 1/2, which inflates the
    aleatoric term and can drive the epistemic residual negative.
 
    Returns
    -------
    m, s : (n_eval,) arrays — mean and SD of eta on the real line
    """
    g = shrink_probability(g_n, clip_eps)
    eta = norm.ppf(g)
 
    jac = (1.0 - 2.0 * clip_eps) / np.maximum(norm.pdf(eta), _TINY)
    s = np.sqrt(np.maximum(np.asarray(v_n, dtype=float) / float(n_train), 0.0)) * jac
    s = np.maximum(s, _TINY)
 
    m = eta * np.sqrt(1.0 + s ** 2) if recenter else eta
    return m, s
 

def probit_shrinkage_report(g_n: np.ndarray, v_n: np.ndarray, n_train: int,
                            clip_eps: float = 1e-12) -> dict:
    """
    Diagnostics for the two knobs the probit path is sensitive to.
 
    ``max_eta_shift`` is how far the shrinkage moves eta = Phi^{-1}(g) for any
    non-saturated g_n; if it is not tiny, clip_eps is too large for the g_n
    this run actually produced. ``n_saturated`` counts g_n exactly at 0 or 1,
    where the shrinkage is doing real (and unavoidable) work.
 
    ``s_max`` and ``n_large_s`` say whether Gauss-Hermite would have been safe:
    it needs n_nodes >> 2 s^2.
    """
    g = np.asarray(g_n, dtype=float)
    interior = (g > 0.0) & (g < 1.0)
 
    eta_shift = np.zeros_like(g)
    eta_shift[interior] = np.abs(
        norm.ppf(shrink_probability(g[interior], clip_eps)) - norm.ppf(g[interior])
    )
 
    _, s = probit_moment_match(g, v_n, n_train, recenter=False, clip_eps=clip_eps)
    return {
        "g_min": float(g.min()),
        "g_max": float(g.max()),
        "n_saturated": int(np.sum(~interior)),
        "max_eta_shift": float(eta_shift.max(initial=0.0)),
        "s_min": float(s.min()),
        "s_max": float(s.max()),
        "n_large_s": int(np.sum(s > 2.0)),
        "n_total": int(g.size),
    }
 
 
def entropy_decomposition_probit(
    g_n: np.ndarray,
    v_n: np.ndarray,
    n_train: int,
    recenter: bool = False,
    clip_eps: float = 1e-12,
    n_nodes: int = 257,
    rule: str = "legendre",
) -> dict:
    """
    Entropic UD via the probit-normal approximation to g~(x*) | z_1:n.
 
    Total uncertainty is h(g_n) as in the Beta path, so the two decompositions
    split the *same* number and are directly comparable.
 
    ``rule`` selects the quadrature for the aleatoric integral; see
    ``_expected_binary_entropy_probit`` for why "hermite" is unsafe here.
 
    Returns
    -------
    dict with the same keys as ``entropy_decomposition`` plus ``m`` and ``s``
    (the Gaussian on R). ``mean_g`` is the induced E[g~]; it equals g_n only
    when ``recenter=True``.
    """
    g = shrink_probability(g_n, clip_eps)
    m, s = probit_moment_match(g_n, v_n, n_train, recenter=recenter, clip_eps=clip_eps)
 
    total = binary_entropy(g)
    aleatoric = _expected_binary_entropy_probit(m, s, n_nodes=n_nodes, rule=rule)
    epistemic_raw = total - aleatoric
 
    # Two defensible totals, and they differ here unless recentred:
    #   "total"                -- h(g_n), TabPFN's own predictive entropy. This
    #                             is the anchor that makes the three methods
    #                             comparable, since it does not depend on the
    #                             approximating family at all.
    #   "total_selfconsistent" -- h(E[g~]) under this method's own law, the
    #                             version for which total - aleatoric is a
    #                             mutual information and is >= 0 by Jensen.
    # With recenter=True, E[g~] = g_n and the two agree to machine precision.
    # With recenter=False the mean drifts towards 1/2, so h(E[g~]) > h(g_n):
    # the anchored residual can go negative while the self-consistent one
    # cannot. The gap between them IS the recentring bias.
    mean_g = norm.cdf(m / np.sqrt(1.0 + s ** 2))
    total_selfconsistent = binary_entropy(mean_g)
 
    return {
        "total": total,
        "total_selfconsistent": total_selfconsistent,
        "aleatoric": aleatoric,
        "epistemic": np.maximum(epistemic_raw, 0.0),
        "epistemic_raw": epistemic_raw,
        "epistemic_selfconsistent": total_selfconsistent - aleatoric,
        "mean_g": mean_g,
        "m": m,
        "s": s,
        "g_n": np.asarray(g_n, dtype=float),
        "v_n": np.asarray(v_n, dtype=float),
    }
 
 
# ---------------------------------------------------------------------------
# Multiclass Beta -> Dirichlet (Appendix L.1.3)
# ---------------------------------------------------------------------------
 
def entropy_decomposition_multiclass(
    g_n: np.ndarray,
    v_n: np.ndarray,
    n_train: int,
    clip_eps: float = 1e-12,
) -> dict:
    """
    Entropic UD for K-class classification (Appendix L.1.3, eqs. 15-18).
 
        alpha_0 = (1 - ||g_n||^2) / sum_k sigma^2_{n,k} - 1
        alpha_k = g_{n,k} alpha_0
        Ua      = psi(alpha_0 + 1) - sum_k g_{n,k} psi(alpha_k + 1)
 
    ``sum_k sigma^2_{n,k}`` is clipped strictly below ``1 - ||g_n||^2`` to keep
    alpha_0 > 0.
 
    Parameters
    ----------
    g_n : (n_eval, K) array — final class probabilities
    v_n : (n_eval, K) array — per-class diagonal of V_n
    n_train : int
    """
    g = np.clip(np.asarray(g_n, dtype=float), clip_eps, None)
    g = g / g.sum(axis=1, keepdims=True)
 
    cap = 1.0 - np.sum(g ** 2, axis=1)                      # 1 - ||g||^2
    s = np.sum(np.asarray(v_n, dtype=float) / float(n_train), axis=1)
    s = np.maximum(np.minimum(s, (1.0 - clip_eps) * cap), _TINY)
 
    alpha0 = cap / s - 1.0
    alpha = g * alpha0[:, None]
 
    total = -np.sum(g * np.log(g), axis=1)
    aleatoric = digamma(alpha0 + 1.0) - np.sum(g * digamma(alpha + 1.0), axis=1)
    epistemic_raw = total - aleatoric
 
    return {
        "total": total,
        "aleatoric": aleatoric,
        "epistemic": np.maximum(epistemic_raw, 0.0),
        "epistemic_raw": epistemic_raw,
        "alpha0": alpha0,
        "alpha": alpha,
        "g_n": g,
    }
 
 
def compute_vn_multiclass(trajectory: np.ndarray) -> np.ndarray:
    """
    Per-class variances sigma^2_{n,k}(x*) * n for a multiclass trajectory.
 
    Eq. (2) at gamma = 1, applied to each class coordinate. The trajectory
    layout (n_train, n_eval, K) is already what ``compute_vn`` consumes, so
    this is the same code path as the regression CDFs and the binary case.
 
    Parameters
    ----------
    trajectory : (n_train, n_eval, K) array
 
    Returns
    -------
    v_n : (n_eval, K) array — divide by n to get the sigma^2_{n,k} of eq. (16).
    """
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim != 3:
        raise ValueError(f"trajectory must be (n_train, n_eval, K); got {traj.shape}")
    return compute_vn(traj, type="pointwise")


def neg_p_log_p(p: np.ndarray) -> np.ndarray:
    """eta(p) = -p log p, with 0 log 0 := 0. The per-class Shannon summand."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    out = np.zeros_like(p)
    mask = p > 0.0
    pm = p[mask]
    out[mask] = -pm * np.log(pm)
    return out


def _expected_probit_functional(
    m: np.ndarray,
    s: np.ndarray,
    func=None,
    n_nodes: int = 257,
    eta_max: float = 12.0,
    rule: str = "legendre",
    s_floor: float = 1e-8,
) -> np.ndarray:
    """
    E[ func(Phi(eta)) ] for eta ~ N(m, s^2), func a bounded functional on [0,1].

    Generalises ``_expected_binary_entropy_probit`` from func = binary_entropy
    to arbitrary func; with func = neg_p_log_p this gives the per-class term
    E[eta(g~_k)] of the multiclass decomposition. The Legendre reasoning is
    unchanged: integrate in eta over the intersection of the Gaussian's
    [m - 9s, m + 9s] and the functional's effective support [-eta_max, eta_max],
    so the quadrature width is set by func rather than by s and does not
    degrade as s grows. For -p log p the integrand is below 1e-29 outside
    |eta| = 12 (it decays like phi(eta) eta^2 / 2 on the left tail and like
    1 - Phi(eta) on the right), so the same eta_max carries over.

    Small-s branch. The Legendre rule is written as ``half * sum(w f dens)``
    with ``dens = phi((eta-m)/s)/s``. Analytically the s -> 0 limit is
    func(Phi(m)), but numerically ``half`` underflows to 0 while ``dens``
    overflows to inf, giving NaN. Below ``s_floor`` the Gaussian is narrower
    than the curvature scale of func(Phi(.)) anyway, so we return the
    degenerate value directly. This matters whenever v_n = 0 at a grid point
    (a trajectory that never moved there), which does happen far from the data.
    """
    if func is None:
        func = binary_entropy  # noqa: F821  (defined in the host module)

    m = np.atleast_1d(np.asarray(m, dtype=float))
    s = np.atleast_1d(np.asarray(s, dtype=float))
    out = np.empty_like(m)

    tiny = s < s_floor
    if np.any(tiny):
        out[tiny] = func(norm.cdf(m[tiny]))
    if np.all(tiny):
        return out

    mm, ss = m[~tiny], s[~tiny]

    if rule == "hermite":
        if n_nodes > 300:
            raise ValueError(
                f"hermgauss is numerically unstable for n_nodes={n_nodes} (>300); "
                "use rule='legendre', which has no such limit."
            )
        x, w = np.polynomial.hermite.hermgauss(n_nodes)
        eta = mm[:, None] + np.sqrt(2.0) * ss[:, None] * x[None, :]
        out[~tiny] = (func(norm.cdf(eta)) @ w) / np.sqrt(np.pi)
        return out

    if rule != "legendre":
        raise ValueError(f"unknown quadrature rule {rule!r}")

    x, w = np.polynomial.legendre.leggauss(n_nodes)

    lo = np.maximum(mm - 9.0 * ss, -eta_max)
    hi = np.minimum(mm + 9.0 * ss, eta_max)
    valid = hi > lo

    mid = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    eta = mid[:, None] + half[:, None] * x[None, :]

    dens = norm.pdf((eta - mm[:, None]) / ss[:, None]) / ss[:, None]
    vals = half * ((func(norm.cdf(eta)) * dens) @ w)
    vals[~valid] = 0.0
    out[~tiny] = vals
    return out

def _expected_binary_entropy_probit(m, s, n_nodes=257, eta_max=12.0, rule="legendre"):
    """
    E[ h(Phi(eta)) ] for eta ~ N(m, s^2). No closed form, so quadrature.
 
    rule="hermite"
        The natural first choice: Gauss-Hermite integrates against the Gaussian
        weight directly, E[f] = (1/sqrt(pi)) sum w_i f(m + sqrt(2) s x_i), and
        it is machine-exact here for small s. It fails for large s, and the
        reason is structural rather than a tuning issue. In the GH variable
        x = (eta - m) / (sqrt(2) s), the factor h(Phi(eta)) is non-negligible
        only on |eta| <~ 4, i.e. on a window of width ~ 4/s in x. GH with K
        nodes has spacing ~ sqrt(2/K) near the origin, so resolving that window
        needs K >> 2 s^2. Measured against Monte Carlo at m = 0: exact to 1e-13
        at s = 1; 13% low at s = 5 with K = 32; and identically 0.0 at s = 20
        for every K that ``numpy.polynomial.hermite.hermgauss`` can even build
        (it overflows past K ~ 300). The failure is silent and one-directional
        — it under-reports the aleatoric term precisely in the tails.
 
    rule="legendre" (default)
        Integrate in eta instead, over the intersection of the two effective
        supports: the Gaussian's [m - 9s, m + 9s] and the entropy's
        [-eta_max, eta_max], outside which h(Phi(eta)) < 1e-30. This is bounded
        by the *entropy's* width rather than the Gaussian's, so it does not
        degrade as s grows. An empty intersection means all the mass sits where
        the entropy vanishes, i.e. the integral is 0. Verified against Monte
        Carlo to <1e-4 (MC error) for s in [1e-3, 200] and |m| up to 20.
 
    s reaches ~15 on the standard [-15, 15] grid at n = 15, because
    s = sqrt(v_n/n) / phi(eta_n) and phi(eta_n) collapses in the tails — so the
    large-s regime is the OOD region this experiment is about, not a corner
    case. Hence the default.
    """
    return _expected_probit_functional(
        m, s, func=binary_entropy, n_nodes=n_nodes, eta_max=eta_max, rule=rule
    )

def probit_moment_match_multiclass(
    g_n: np.ndarray,
    v_n: np.ndarray,
    n_train: int,
    recenter: bool = False,
    clip_eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-class probit-normal marginals for a K-class predictive.

    Shannon entropy is a sum of per-class terms, H(g~) = sum_k eta(g~_k), and
    eta(g~_k) depends only on the *marginal* of g~_k. So the multiclass case
    needs no law on the simplex: it is K independent copies of the binary
    construction, one per class coordinate.

        eta_{n,k} = Phi^{-1}(g_{n,k})
        s_k       = sqrt(v_{n,k} / n) * (1 - 2 eps) / phi(eta_{n,k})
        m_k       = eta_{n,k} * sqrt(1 + s_k^2)      (recenter=True)
                  = eta_{n,k}                        (recenter=False)

    Recentring uses the same closed form as the binary case, E[Phi(m + sZ)] =
    Phi(m / sqrt(1 + s^2)), so E[g~_k] = g_{n,k} for every k. Since the g_{n,k}
    sum to 1, the recentred marginal means also sum to 1 — the construction is
    consistent with the simplex in first moment even though it does not define
    a joint law on it.

    Note this matches strictly more than the Dirichlet of Appendix L.1.3: that
    has a single scalar alpha_0 and therefore matches the mean vector plus only
    the *sum* of the per-class variances, whereas this matches each
    (g_{n,k}, v_{n,k}/n) separately. It also needs no clip: any s_k > 0 gives a
    valid law on (0,1), whereas alpha_0 > 0 requires sum_k sigma^2_{n,k} to be
    capped below 1 - ||g_n||^2.

    At K = 2 this reduces exactly to the binary case: the two diagonal
    variances coincide and Phi^{-1}(1-g) = -Phi^{-1}(g), so s_2 = s_1,
    m_2 = -m_1, and the second marginal is the law of 1 - g~_1.

    Parameters
    ----------
    g_n : (n_eval, K) array — final class probabilities, rows summing to 1
    v_n : (n_eval, K) array — per-class diagonal of V_n (divide by n for sigma^2)

    Returns
    -------
    m, s : (n_eval, K) arrays — mean and SD of eta_k on the real line
    """
    g = shrink_probability(np.asarray(g_n, dtype=float), clip_eps)  # noqa: F821
    eta = norm.ppf(g)

    jac = (1.0 - 2.0 * clip_eps) / np.maximum(norm.pdf(eta), _TINY)  # noqa: F821
    s = np.sqrt(np.maximum(np.asarray(v_n, dtype=float) / float(n_train), 0.0)) * jac
    s = np.maximum(s, _TINY)  # noqa: F821

    m = eta * np.sqrt(1.0 + s ** 2) if recenter else eta
    return m, s


def entropy_decomposition_multiclass_probit(
    g_n: np.ndarray,
    v_n: np.ndarray,
    n_train: int,
    recenter: bool = False,
    clip_eps: float = 1e-12,
    n_nodes: int = 257,
    rule: str = "legendre",
) -> dict:
    """
    Entropic UD for K-class classification via per-class probit-normals.

        Total      H  = -sum_k g_{n,k} log g_{n,k}          (same anchor as the
                                                             Dirichlet path)
        Aleatoric  Ua = sum_k E[ -g~_k log g~_k ]
        Epistemic  Ue = H - Ua

    With ``recenter=True`` each term satisfies E[g~_k] = g_{n,k}, so concavity
    of eta(p) = -p log p gives E[eta(g~_k)] <= eta(g_{n,k}) *termwise*; summing
    over k gives Ue >= 0 with no clipping anywhere. With ``recenter=False`` the
    marginal means drift towards 1/2, which inflates Ua and can push the
    residual negative — ``epistemic_raw`` and ``mean_sum`` are returned
    unclipped so that drift stays visible.

    Drop-in for ``entropy_decomposition_multiclass``: same signature prefix,
    same ``total``/``aleatoric``/``epistemic`` keys.

    Parameters
    ----------
    g_n : (n_eval, K) array
    v_n : (n_eval, K) array
    n_train : int

    Returns
    -------
    dict with ``total``, ``aleatoric``, ``epistemic``, ``epistemic_raw``,
    ``total_selfconsistent``, ``epistemic_selfconsistent``, ``mean_g`` (n_eval, K),
    ``mean_sum``, ``m``, ``s``, ``g_n``.
    """
    g = np.clip(np.asarray(g_n, dtype=float), clip_eps, None)
    g = g / g.sum(axis=1, keepdims=True)

    m, s = probit_moment_match_multiclass(
        g, v_n, n_train, recenter=recenter, clip_eps=clip_eps
    )

    total = -np.sum(g * np.log(g), axis=1)

    per_class = _expected_probit_functional(
        m.ravel(), s.ravel(), func=neg_p_log_p, n_nodes=n_nodes, rule=rule
    ).reshape(g.shape)
    aleatoric = per_class.sum(axis=1)
    epistemic_raw = total - aleatoric

    # Induced marginal means. Under recenter=True these equal g exactly and
    # mean_sum == 1; under recenter=False they are each shrunk towards 1/2, so
    # mean_sum drifts away from 1 and the gap IS the recentring bias. The
    # self-consistent total normalises them so it remains a genuine entropy.
    mean_g = norm.cdf(m / np.sqrt(1.0 + s ** 2))
    mean_sum = mean_g.sum(axis=1)
    mean_norm = mean_g / mean_sum[:, None]
    total_selfconsistent = -np.sum(mean_norm * np.log(np.clip(mean_norm, _TINY, None)), axis=1)  # noqa: F821

    return {
        "total": total,
        "total_selfconsistent": total_selfconsistent,
        "aleatoric": aleatoric,
        "epistemic": np.maximum(epistemic_raw, 0.0),
        "epistemic_raw": epistemic_raw,
        "epistemic_selfconsistent": total_selfconsistent - aleatoric,
        "aleatoric_per_class": per_class,
        "mean_g": mean_g,
        "mean_sum": mean_sum,
        "m": m,
        "s": s,
        "g_n": g,
    }