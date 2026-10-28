"""
TabPFN helpers for martingale-posterior workflows.

Functions
---------
forward_samples               — draw n_forward samples from TabPFN's BAR distribution
cdf_from_samples              — convert raw forward samples to a CDF on a y-grid

These are separated from ``martingale.py`` so that ``get_posterior`` remains
model-agnostic.
"""

from __future__ import annotations

import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF


def _get_tabpfn_regressor_class():
    try:
        from tabpfn import TabPFNRegressor
    except ImportError:
        try:
            from tabpfn_extensions import TabPFNRegressor
        except ImportError as exc:
            raise ImportError(
                "Could not import TabPFNRegressor from `tabpfn` or `tabpfn_extensions`."
            ) from exc
    return TabPFNRegressor


# ---------------------------------------------------------------------------
# Forward samples from TabPFN's BAR distribution
# ---------------------------------------------------------------------------

def forward_samples(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    n_forward: int = 1000,
    n_estimators: int = 4
) -> np.ndarray:
    """
    Fit TabPFN on *(X_train, y_train)* and draw *n_forward* forward samples for each
    row of *X_eval* from the BAR (Bayesian Additive Regression) distribution.

    Parameters
    ----------
    X_train : (n_train, n_features) array
    y_train : (n_train,) array
    X_eval  : (n_eval, n_features) array
    n_forward       : int — number of samples per evaluation point

    Returns
    -------
    samples : (n_forward, n_eval) numpy array
    """
    import torch

    TabPFNRegressor = _get_tabpfn_regressor_class()

    # The sandbox tends to oversubscribe CPU threads with the default n_jobs=-1,
    # which slows down repeated hard-case evaluations more than it helps.
    model = TabPFNRegressor(n_jobs=1, n_estimators=n_estimators)
    model.fit(X_train, y_train)
    pred_output = model.predict(X_eval, output_type="full")
    logits = pred_output["logits"] 
    # these logits are sort of like probability mass assigned to lots of tiny slices along the y-axis (careful, not bounded or normalized!)
    n_eval = logits.shape[0]
    p_cdf = torch.rand(n_eval, n_forward, device=logits.device)

    rows = []
    for i in range(n_eval):
        row = torch.stack(
            [pred_output["criterion"].icdf(logits[i], p) for p in p_cdf[i]]
        )
        rows.append(row)

    return torch.stack(rows).T.detach().cpu().numpy()  # (n_forward, n_eval)


# ---------------------------------------------------------------------------
# CDF on a y-grid from raw samples
# ---------------------------------------------------------------------------

def cdf_from_samples(
    y_grid: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    """
    Compute the empirical CDF of each column of *samples* evaluated on
    *y_grid*.

    Parameters
    ----------
    y_grid  : (y_grid_size,) or (y_grid_size, 1) array — evaluation grid
    samples : (n_forward, n_eval) array — each column is an independent sample set for
              one test point

    Returns
    -------
    cdf : (y_grid_size, n_eval) array — CDF values on y_grid for each test point
    """
    y_grid_1d = np.asarray(y_grid).ravel()
    n_eval = samples.shape[1]
    cdf = np.zeros((len(y_grid_1d), n_eval))
    for i in range(n_eval):
        ecdf = ECDF(samples[:, i])
        cdf[:, i] = ecdf(y_grid_1d)
    return cdf


# ---------------------------------------------------------------------------
# Asymptotic Trajectory Helpers (Fortini Method)
# ---------------------------------------------------------------------------
def analytic_cdf_from_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_grid: np.ndarray,
    n_estimators: int = 4, 
    **tabpfn_kwargs,
) -> np.ndarray:
    """
    Fit TabPFN and compute the exact analytic CDF on the y_grid directly 
    from the BAR distribution criterion, used for trajectory for the asymptotic methods. 
    """
    import torch
    TabPFNRegressor = _get_tabpfn_regressor_class()

    # more estimators might be needed here to get smoother trajetory if using multiple features
    model = TabPFNRegressor(n_jobs=1, n_estimators=n_estimators, **tabpfn_kwargs)
    model.fit(X_train, y_train)
    
    pred_output = model.predict(X_eval, output_type="full")
    logits = pred_output["logits"]
    criterion = pred_output["criterion"]
    
    # Move  evaluation grid to  same device as the model output
    y_grid_t = torch.as_tensor(y_grid, dtype=torch.float32, device=logits.device)
    
    n_eval = logits.shape[0]
    cdf_rows = []
    
    with torch.no_grad():
        for i in range(n_eval):
            # criterion.cdf typically expects batched inputs.
            #  unsqueeze to make logits[i] (1, num_bins) and y_grid (1, y_grid_size)
            cdf_vals = criterion.cdf(logits[i].unsqueeze(0), y_grid_t.unsqueeze(0))
            
            # Squeeze back down to 1D and append
            cdf_rows.append(cdf_vals.squeeze(0))
            
    # return shape (y_grid_size, n_eval)
    return torch.stack(cdf_rows).T.cpu().numpy()


def build_predictive_trajectory(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_eval: np.ndarray, 
    y_grid: np.ndarray,
    n_estimators: int = 4,
    **tabpfn_kwargs
) -> np.ndarray:
    """
    Iteratively evaluates TabPFN on expanding prefixes of the training data.
    
    Parameters
    ---------- 
    X_train : (n_train, n_features) array
        Shuffled training features.
    y_train : (n_train,) array
        Shuffled training labels.
    X_eval  : (n_eval, n_features) array
        Evaluation points.
    y_grid  : (y_grid_size,) array
        The discrete grid of thresholds to evaluate the CDF on.
    n_estimators : int
        Number of estimators for TabPFN 
        
    Returns
    -------
    trajectory : (n_train, y_grid_size, n_eval) array
        The history of evaluated CDFs on the y_grid.
    """
    n_train = len(X_train) 
    n_eval = len(X_eval) # Covariate values to evaluate the CDF at 
    y_grid_size = len(y_grid) # y_grid_size different thresholds to evaluate the CDF at 
    
    trajectory = np.zeros((n_train, y_grid_size, n_eval))
    
    
    for i in range(1, n_train + 1):
        x_prefix = X_train[:i]
        y_prefix = y_train[:i]

        # Degeneracy check, if only one label present in prefix, substitute with marginal empirical CDF 
        # (Regression setting) 
        label_degenerate = len(np.unique(y_prefix)) < 2
        
        # further added degeneracy check for covariates as well 
        covariate_degenerate = (i == 1) or np.all(x_prefix == x_prefix[0])

        if label_degenerate or covariate_degenerate:
            y_prefix_1d = np.asarray(y_prefix).ravel()
            ecdf = ECDF(y_prefix_1d)
            cdf_vals = ecdf(y_grid)
            # Broadcast the empirical CDF across all evaluation points
            trajectory[i-1] = np.tile(cdf_vals[:, None], (1, n_eval))
        else:
            # exact CDF
            trajectory[i-1] = analytic_cdf_from_tabpfn(x_prefix, y_prefix, X_eval, y_grid, n_estimators, **tabpfn_kwargs)
            
    return trajectory


def _get_tabpfn_classifier_class():
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        try:
            from tabpfn_extensions import TabPFNClassifier
        except ImportError as exc:
            raise ImportError(
                "Could not import TabPFNClassifier from `tabpfn` or `tabpfn_extensions`."
            ) from exc
    return TabPFNClassifier
 
 
def make_classifier(
    n_estimators: int = 64,
    device: str | None = None,
    fit_mode: str | None = "low_memory",
    softmax_temperature: float = 1.0,
):
    """
    Instantiate a TabPFNClassifier with the paper's default settings
    (Appendix A: n_estimators=64, softmax_temperature=1.0, fit_mode="low_memory").
 
    The instance is meant to be *reused* across prefixes: ``fit`` only swaps the
    in-context table, so building the trajectory does not reload the checkpoint
    n times.
 
    Unknown keyword arguments are dropped for older tabpfn versions.
    """
    TabPFNClassifier = _get_tabpfn_classifier_class()
 
    kwargs = {"n_jobs": 1, "n_estimators": n_estimators}
    if device is not None:
        kwargs["device"] = device
    if fit_mode is not None:
        kwargs["fit_mode"] = fit_mode
    if softmax_temperature is not None:
        kwargs["softmax_temperature"] = softmax_temperature
 
    while True:
        try:
            return TabPFNClassifier(**kwargs)
        except TypeError as exc:
            # drop whichever optional kwarg this version does not know about
            dropped = None
            for key in ("softmax_temperature", "fit_mode", "device", "n_jobs"):
                if key in kwargs and key in str(exc):
                    dropped = key
                    break
            if dropped is None:
                raise
            kwargs.pop(dropped)
 
 
def class1_proba_from_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    n_estimators: int = 64,
    model=None,
) -> np.ndarray:
    """
    g_k(x*) = P(Y = 1 | x*, z_1:k) from TabPFNClassifier.
 
    Softmax (temperature 1) over the output logits, i.e. exactly the predictive
    probability the framework needs (Appendix B).
 
    Returns
    -------
    (n_eval,) array
    """
    if model is None:
        model = make_classifier(n_estimators=n_estimators)
 
    y_int = np.asarray(y_train).astype(int).ravel()
    model.fit(np.asarray(X_train), y_int)
 
    proba = np.asarray(model.predict_proba(np.asarray(X_eval)), dtype=float)
    classes = np.asarray(model.classes_).astype(int)
    col = int(np.flatnonzero(classes == 1)[0])
    return proba[:, col]
 
 
def build_classifier_trajectory(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    n_estimators: int = 64,
    model=None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Iteratively evaluate TabPFNClassifier on expanding (non-empty) prefixes of
    the — already permuted — context, giving g_1, ..., g_n at each test point.
 
    Degeneracy rule (Appendix B): for single-class prefixes the predictive is
    replaced by the observed statistic, g_k(x) = 1 if the prefix contains only
    class 1 and 0 if it contains only class 0.
 
    Parameters
    ----------
    X_train : (n_train, n_features) array — *randomly permuted* rows
    y_train : (n_train,) array of {0, 1} labels, in the same order
    X_eval  : (n_eval, n_features) array — test covariates x*
    n_estimators : int
    model : optional pre-built TabPFNClassifier to reuse across prefixes
 
    Returns
    -------
    trajectory : (n_train, n_eval) array — row k-1 holds g_k(x*)
    """
    X_train = np.asarray(X_train)
    y_int = np.asarray(y_train).astype(int).ravel()
    X_eval = np.asarray(X_eval)
 
    n_train = len(X_train)
    n_eval = len(X_eval)
    trajectory = np.zeros((n_train, n_eval))
 
    if model is None:
        model = make_classifier(n_estimators=n_estimators)
 
    for k in range(1, n_train + 1):
        y_prefix = y_int[:k]
        uniq = np.unique(y_prefix)
 
        if uniq.size < 2:
            # single-class prefix -> observed constant class
            trajectory[k - 1] = float(uniq[0])

        elif np.all(X_train[:k] == X_train[0]):
            # constant-covariate prefix (TabPFN refuses it) -> observed class
            # frequency, the classification analogue of the marginal ECDF used
            # in build_predictive_trajectory
            trajectory[k - 1] = float(y_prefix.mean())
        else:
            trajectory[k - 1] = class1_proba_from_tabpfn(
                X_train[:k], y_prefix, X_eval, n_estimators=n_estimators, model=model
            )

 
        if verbose and (k % 25 == 0 or k == n_train):
            print(f"    prefix {k}/{n_train}", flush=True)
 
    return trajectory
 
 
def class_proba_from_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    n_classes: int,
    n_estimators: int = 64,
    model=None,
) -> np.ndarray:
    """
    Full K-vector g_k(x*) = P(Y = . | x*, z_1:k) from TabPFNClassifier.
 
    TabPFN only knows the classes present in the prefix, so its predict_proba
    has one column per *observed* class. Those are scattered back into the full
    K-vector and unobserved classes receive probability 0 — the multiclass
    reading of Appendix B's rule that the predictive is replaced by the
    observed statistics. Note that a class first appearing at step k therefore
    produces a large Delta_k, which V_n weights by k^2; that is the same
    phenomenon as the first class flip in the binary case, not an artefact.
 
    Returns
    -------
    (n_eval, n_classes) array, rows summing to 1
    """
    if model is None:
        model = make_classifier(n_estimators=n_estimators)
 
    y_int = np.asarray(y_train).astype(int).ravel()
    model.fit(np.asarray(X_train), y_int)
 
    proba = np.asarray(model.predict_proba(np.asarray(X_eval)), dtype=float)
    classes = np.asarray(model.classes_).astype(int)
 
    out = np.zeros((proba.shape[0], n_classes))
    out[:, classes] = proba
    return out
 
 
def build_multiclass_trajectory(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    n_classes: int,
    n_estimators: int = 64,
    model=None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Multiclass analogue of ``build_classifier_trajectory``.
 
    Degeneracy rule (Appendix B): for single-class prefixes the predictive is
    replaced by the observed statistic — here a point mass on the one class
    seen so far.
 
    Returns
    -------
    trajectory : (n_train, n_eval, n_classes) array — entry k-1 holds g_k(x*).
        This is exactly the layout ``compute_vn(..., type="pointwise")``
        consumes, so the per-class variances come straight out of it.
    """
    X_train = np.asarray(X_train)
    y_int = np.asarray(y_train).astype(int).ravel()
    X_eval = np.asarray(X_eval)
 
    n_train = len(X_train)
    trajectory = np.zeros((n_train, len(X_eval), n_classes))
 
    if model is None:
        model = make_classifier(n_estimators=n_estimators)
 
    for k in range(1, n_train + 1):
        y_prefix = y_int[:k]
        uniq = np.unique(y_prefix)
 
        if uniq.size < 2:
            trajectory[k - 1, :, int(uniq[0])] = 1.0
        else:
            trajectory[k - 1] = class_proba_from_tabpfn(
                X_train[:k], y_prefix, X_eval, n_classes,
                n_estimators=n_estimators, model=model,
            )
 
        if verbose and (k % 25 == 0 or k == n_train):
            print(f"    prefix {k}/{n_train}", flush=True)
 
    return trajectory