"""
Shared pieces of the experiment scripts: the fit / plot run cache, CLI flags and the thesis plot style.

Every experiment follows the same pattern:

    results/artifacts/<script>[_<mode>]/
        config.json          the settings of the fit (+ cache version); a cache made with other settings
                             is refused
        cache/<key>.npz      one file per unit of work, usually one (n, repetition), written atomically
        *.csv, *.pdf, *.png  the thesis tables and figures, rebuilt from the whole cache by the plot step

    --run fit    compute and cache only what is missing (resumable; a larger --n-reps only adds reps)
    --run plot   tables and figures from the cache, no TabPFN
    --run all    both (default)
    --overwrite  delete the cache and config.json before fitting
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = Path("results") / "artifacts"


# ---------------------------------------------------------------------------
# CLI and environment
# ---------------------------------------------------------------------------
def add_run_args(parser: argparse.ArgumentParser) -> None:
    """--run, --overwrite, --threads and --outdir, shared by every script."""
    parser.add_argument("--run", choices=("fit", "plot", "all"), default="all",
                        help="fit: compute and cache missing runs; plot: tables and figures from the cache; "
                             "all: both")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete the cache and config.json before fitting")
    parser.add_argument("--threads", type=int, default=8, help="torch CPU threads used by TabPFN")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="parent folder of the outputs")


def add_sample_args(parser: argparse.ArgumentParser, n_list, n_reps: int) -> None:
    """--n-train-list / --n-reps: what the fit step computes (the plot step uses everything cached)."""
    parser.add_argument("--n-train-list", type=int, nargs="+", default=list(n_list), help="sample sizes n")
    parser.add_argument("--n-reps", type=int, default=n_reps, help="repetitions R")


def n_estimators(thesis_value: int) -> int:
    """TabPFN ensemble size: the thesis value, or $UQPFN_N_ESTIMATORS if set (quick test runs)."""
    return int(os.environ.get("UQPFN_N_ESTIMATORS", thesis_value))


def add_seed_arg(parser: argparse.ArgumentParser, default: int) -> None:
    parser.add_argument("--seed", type=int, default=default, help="base random seed (default: %(default)s)")


def add_n_estimators_arg(parser: argparse.ArgumentParser, thesis_value: int) -> None:
    """--n-estimators, defaulting to the thesis value ($UQPFN_N_ESTIMATORS overrides the default)."""
    parser.add_argument("--n-estimators", type=int, default=n_estimators(thesis_value),
                        help=f"TabPFN ensemble size (thesis: {thesis_value}; default: %(default)s)")


def ci_label(alpha: float, usetex: bool = False) -> str:
    """Legend label of a (1 - alpha) interval, e.g. '95% CI'."""
    pct = r"\%" if usetex else "%"
    return f"{round(100 * (1 - alpha), 6):g}{pct} CI"


def quiet_warnings() -> None:
    """Warnings from the sklearn functions TabPFN uses, and harmless overflow in the AMP copula."""
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", message="overflow encountered in power", category=RuntimeWarning)


def set_threads(n: int | None) -> None:
    """Called at the start of a fit only, so plot-only runs never import torch."""
    if not n:
        return
    import torch
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:  # can only be set once per process
        pass


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def se(values) -> float:
    """Standard error of the mean (NaN for a single value)."""
    v = np.asarray(values, dtype=float)
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


# ---------------------------------------------------------------------------
# Run cache
# ---------------------------------------------------------------------------
class RunCache:
    """
    config.json + one .npz per key under <folder>/cache.

    ``config`` holds every setting that changes the fitted results. ``version`` is appended to it
    (bump it when the cached quantities change); ``None`` leaves it out.
    """

    def __init__(self, folder: Path, config: dict, version: int | None = 1):
        self.folder = Path(folder)
        self.dir = self.folder / "cache"
        self.config = dict(config) if version is None else dict(config) | {"cache_version": version}
        self.config_path = self.folder / "config.json"

    def prepare(self, overwrite: bool = False) -> None:
        """Create the folder, check (or write) config.json; with overwrite, start from scratch."""
        self.folder.mkdir(parents=True, exist_ok=True)
        if overwrite:
            shutil.rmtree(self.dir, ignore_errors=True)
            self.config_path.unlink(missing_ok=True)
            print(f"[fit] --overwrite: removed the cache in {self.folder}")
        self.check()
        self.config_path.write_text(json.dumps(_jsonable(self.config), indent=2))

    def check(self) -> None:
        """Exit if config.json exists and holds other settings than this run."""
        if self.config_path.exists():
            old = json.loads(self.config_path.read_text())
            if old != _jsonable(self.config):
                raise SystemExit(f"{self.config_path} holds other settings {old} than this run "
                                 f"{_jsonable(self.config)}; pass the same arguments, rerun the fit with "
                                 "--overwrite to replace them, or use another --outdir to keep both.")

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.npz"

    def has(self, key: str) -> bool:
        return self.path(key).exists()

    def save(self, key: str, **arrays) -> Path:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)   # an interrupted run never leaves a broken file
        return path

    def load(self, key: str) -> dict:
        return load_npz(self.path(key))

    def keys(self, pattern: str = "**/*") -> list[str]:
        """Cached keys (paths relative to cache/, without .npz), sorted."""
        if not self.dir.exists():
            return []
        files = sorted(p for p in self.dir.glob(f"{pattern}.npz") if not p.name.endswith(".tmp.npz"))
        return [str(p.relative_to(self.dir).with_suffix("")) for p in files]

    def require(self) -> list[str]:
        """All cached keys; exits if there are none, or if the cache was made with other settings."""
        self.check()
        keys = self.keys()
        if not keys:
            raise SystemExit(f"No cached runs under {self.dir}; run with --run fit first.")
        return keys


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def n_tag(n: int | None) -> str:
    """Cache folder of a sample size (``npool`` for designs whose size is not set by n)."""
    return "npool" if n is None else f"n{n:04d}"


def rep_key(n: int | None, rep: int, suffix: str = "") -> str:
    return f"{n_tag(n)}/rep{rep:03d}{suffix}"


def parse_rep_key(key: str) -> tuple[int | None, int, str]:
    """'n0200/rep003_est064' -> (200, 3, '_est064')."""
    tag, stem = key.split("/")
    n = None if tag == "npool" else int(tag[1:])
    rep, _, rest = stem[3:].partition("_")
    return n, int(rep), ("_" + rest) if rest else ""


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(u) for k, u in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(u) for u in v]
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


# ---------------------------------------------------------------------------
# Thesis plot style (fixed panel geometry, Computer-Modern-like serif)
# ---------------------------------------------------------------------------
def setup_style(font_size: float, usetex: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import logging
    # silence harmless fontTools messages ("'created' timestamp seems very low") when embedding fonts in the PDF
    logging.getLogger("fontTools").setLevel(logging.ERROR)

    # pick a Computer-Modern-like serif font if one is installed, otherwise fall back
    available = {f.name for f in font_manager.fontManager.ttflist}
    serif = [f for f in ["CMU Serif", "Latin Modern Roman", "Computer Modern Roman"] if f in available]
    serif.append("DejaVu Serif")

    plt.rcParams.update({
        "text.usetex": usetex,
        "font.family": "serif",
        "font.serif": serif,
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "lines.linewidth": 1.3,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    })
    if usetex:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    return plt


def add_style_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fig-width", type=float, default=6.3,
                        help="Width (inches) of a full row of panels, the LaTeX \\textwidth (6.3in ~ 16cm). "
                             "Include figures at natural size so fonts stay true size.")
    parser.add_argument("--font-size", type=float, default=9.0, help="Base font size in pt at the final printed size.")
    parser.add_argument("--usetex", action="store_true", help="Render all text with LaTeX (needs a LaTeX installation).")


def panel_geometry(fig_width: float, n_cols: int = 3) -> dict:
    """Fixed margins in inches, so every panel (in every figure file) has the exact same axes size."""
    left, right, wspace = 0.50, 0.08, 0.12
    panel_w = (fig_width - left - right - (n_cols - 1) * wspace) / n_cols
    return {
        "left": left,
        "right": right,
        "wspace": wspace,
        "bottom": 0.42,  # room for x tick labels + x label
        "top": 0.28,     # room for the title
        "panel_w": panel_w,
        "panel_h": 0.9 * panel_w,
    }


def fixed_axes_figure(plt, n_panels: int, geo: dict, extra_bottom: float = 0.0, sharey: bool = True):
    bottom = geo["bottom"] + extra_bottom
    fig_w = geo["left"] + n_panels * geo["panel_w"] + (n_panels - 1) * geo["wspace"] + geo["right"]
    fig_h = bottom + geo["panel_h"] + geo["top"]
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = []
    for i in range(n_panels):
        x0 = geo["left"] + i * (geo["panel_w"] + geo["wspace"])
        rect = [x0 / fig_w, bottom / fig_h, geo["panel_w"] / fig_w, geo["panel_h"] / fig_h]
        share = axes[0] if (sharey and axes) else None
        axes.append(fig.add_axes(rect, sharey=share))
    return fig, axes


def save_fig(fig, stem: Path) -> list[Path]:
    paths = []
    for ext in ("pdf", "png"):
        p = stem.parent / f"{stem.name}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
        paths.append(p)
    return paths
