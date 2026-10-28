"""
uqpfn: Uncertainty Quantification for Prior-Data Fitted Networks.

Public modules
--------------
martingale  - model-agnostic approximate martingale posterior (AMP) credible intervals
asymptotic  - predictive CLT: V_n, CLT / probit-CLT bounds, entropy decompositions
tabpfn      - TabPFN predictive CDFs and predictive / classifier trajectories
"""

def _optional_import(name: str):
    try:
        mod = __import__(f"{__name__}.{name}", fromlist=[name])
    except Exception:  # pragma: no cover - dependency availability varies by env
        mod = None
    return mod


martingale = _optional_import("martingale")
asymptotic = _optional_import("asymptotic")
tabpfn = _optional_import("tabpfn")

__all__ = ["martingale", "asymptotic", "tabpfn"]
