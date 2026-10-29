#!/usr/bin/env python3
"""
Quasi-martingale diagnostics: TabPFN rollouts and the conditional moments of their updates.

Thesis Sec. 4.1 and App. A.2, following Fortini et al. (2026a), App. B.3:
    rollout     initial prefix of n0 = 25 points, x ~ U{-1, 0, 1, 2}, y ~ Bern(sigmoid(2 x - 0.5)),
                redrawn until both labels occur; then extended to N = 1025 under TabPFN's own predictive
    moments     at 125 geometrically spaced prefix lengths n in [n0 + 100, N], the conditional drift
                b_n = E[Delta_n | Z_1:n-1] and b'_n = E[Delta_n^2 | Z_1:n-1] at x* = 1, computed exactly by
                refitting TabPFN on the prefix augmented with every (support point, label)

The figures (Fig. 4.1 - 4.4, App. A.4, A.5) are made by tabpfn_diagnostics.ipynb from the rollout CSVs.

Cache: results/tabpfn_diagnostics_rollouts/tabpfn_drift_diagnostics_raw_r<k>.csv, one file per rollout
(k = 1, 2, ...), seed = 123 + k - 1, written atomically; the 20 rollouts used in the thesis are committed.
Rollouts that already exist are skipped, so a run can be stopped and resumed.

Usage
    python scripts/quasi_martingale.py                     # rollouts 1..20 (missing ones only)
    python scripts/quasi_martingale.py --n-rollouts 25     # adds rollouts 21..25
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [p for p in (str(ROOT), str(ROOT / "src")) if p not in sys.path]

from scripts.common import RunCache, add_n_estimators_arg, add_seed_arg, quiet_warnings, set_threads  # noqa: E402

quiet_warnings()
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "true"

# the support of x is fixed: the CSV columns (counts, g and drift per support point) are built around it
SUPPORT = np.array([-1, 0, 1, 2])
FIELDNAMES = [
    'rollout_id', 'step_n', 'query_x',
    'prefix_y_ratio', 'prefix_count_x_minus1', 'prefix_count_x_0', 'prefix_count_x_1', 'prefix_count_x_2',
    'g_target', 'entropy_target', 'g_x_minus1', 'g_x_0', 'g_x_1', 'g_x_2',
    'expected_drift_bn', 'expected_squared_drift_bn_prime',
    'drift_given_x_minus1', 'drift_given_x_0', 'drift_given_x_1', 'drift_given_x_2'
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TabPFN quasi-martingale drift diagnostics (rollouts).")
    parser.add_argument("--n-rollouts", type=int, default=20, help="rollouts R")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete the rollout CSVs and config.json in --outdir before running")
    parser.add_argument("--threads", type=int, default=8, help="torch CPU threads used by TabPFN")
    parser.add_argument("--outdir", type=Path, default=Path("results") / "tabpfn_diagnostics_rollouts")
    add_seed_arg(parser, 123)
    add_n_estimators_arg(parser, 64)
    parser.add_argument("--n0", type=int, default=25, help="initial prefix n0 (default: %(default)s)")
    parser.add_argument("--total-n", type=int, default=1025, help="length N of each rollout (default: %(default)s)")
    parser.add_argument("--grid-size", type=int, default=125,
                        help="geometrically spaced evaluation points on [n0 + eval-offset, N] (default: %(default)s)")
    parser.add_argument("--eval-offset", type=int, default=100,
                        help="first evaluation point is n0 + this (default: %(default)s)")
    parser.add_argument("--query-x", type=float, default=1, help="test point x* (default: %(default)s)")
    parser.add_argument("--beta", type=float, default=2.0, help="slope of the prefix logit (default: %(default)s)")
    parser.add_argument("--beta0", type=float, default=-0.5, help="intercept of the prefix logit (default: %(default)s)")
    parser.add_argument("--device", default="cpu", help="torch device of TabPFN (default: %(default)s)")
    return parser.parse_args()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _classifier(n_estimators: int, device: str):
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(n_estimators=n_estimators, device=device)


def generate_tabpfn_rollout(total_N, n_estimators=8, n_0=25, beta=2.0, beta0=-0.5, support=SUPPORT,
                            device="cpu"):
    """
    Generates a sequence exactly following Appendix B.3 of Fortini et al. 2026.
    """

    # initial prefix of n_0 points, redrawn until both labels occur
    while True:
        X_init = np.random.choice(support, size=n_0)
        probs = sigmoid(beta * X_init + beta0)
        y_init = np.random.binomial(1, probs)
        if len(np.unique(y_init)) > 1:
            break

    X_seq = list(X_init)
    y_seq = list(y_init)

    # extend the sequence under TabPFN's own predictive
    for t in range(n_0 + 1, total_N + 1):
        X_t = np.random.choice(support)
        X_t_2d = np.array([[X_t]])

        X_prefix = np.array(X_seq).reshape(-1, 1)
        y_prefix = np.array(y_seq)

        model = _classifier(n_estimators, device)
        model.fit(X_prefix, y_prefix)

        # draw the next label from the predicted probability at the new covariate
        g_t_minus_1 = model.predict_proba(X_t_2d)[0, 1]
        Y_t = np.random.binomial(1, g_t_minus_1)

        X_seq.append(X_t)
        y_seq.append(Y_t)

    return np.array(X_seq).reshape(-1, 1), np.array(y_seq)


def compute_and_log_expected_drift(X_seq, y_seq, eval_points, rollout_id, csv_filename, n_estimators=8,
                                   query_x=1, support=SUPPORT, device="cpu"):
    """
    Computes expected drift, variance, and context state, writing every evaluation step to the CSV.
    """
    support = support.reshape(-1, 1)
    x_star = np.array([[query_x]])

    file_exists = os.path.isfile(csv_filename) and os.path.getsize(csv_filename) > 0
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for n in eval_points:
            if n > len(X_seq):
                continue

            X_prefix = X_seq[:n-1]
            y_prefix = y_seq[:n-1]

            if len(np.unique(y_prefix)) < 2:
                continue  # degenerate prefix

            # state of the prefix
            unique_x, counts_x = np.unique(X_prefix, return_counts=True)
            counts_dict = dict(zip(unique_x.flatten(), counts_x))

            y_ratio = np.mean(y_prefix)
            count_m1 = counts_dict.get(-1, 0)
            count_0 = counts_dict.get(0, 0)
            count_1 = counts_dict.get(1, 0)
            count_2 = counts_dict.get(2, 0)

            # g_{n-1} at the support points and at x*
            model = _classifier(n_estimators, device)
            model.fit(X_prefix, y_prefix)

            g_n_minus_1_support = model.predict_proba(support)[:, 1]
            g_m1, g_0, g_1, g_2 = g_n_minus_1_support

            g_n_minus_1_x_star = model.predict_proba(x_star)[0, 1]

            # entropy at x* (clipped to avoid log(0))
            eps = 1e-9
            g_safe = np.clip(g_n_minus_1_x_star, eps, 1-eps)
            entropy = -g_safe * np.log(g_safe) - (1 - g_safe) * np.log(1 - g_safe)

            # conditional first and second moment of the update: refit with every (support point, label)
            expected_delta_n = 0.0
            expected_squared_delta_n = 0.0
            drift_given_x = {}

            for j, x_j in enumerate(support):
                x_j_2d = x_j.reshape(1, -1)

                model_0 = _classifier(n_estimators, device)
                model_0.fit(np.vstack([X_prefix, x_j_2d]), np.append(y_prefix, 0))
                d_0 = model_0.predict_proba(x_star)[0, 1] - g_n_minus_1_x_star

                model_1 = _classifier(n_estimators, device)
                model_1.fit(np.vstack([X_prefix, x_j_2d]), np.append(y_prefix, 1))
                d_1 = model_1.predict_proba(x_star)[0, 1] - g_n_minus_1_x_star

                prob_y_is_1 = g_n_minus_1_support[j]
                prob_y_is_0 = 1.0 - prob_y_is_1

                e_delta_given_x_j = (prob_y_is_0 * d_0) + (prob_y_is_1 * d_1)
                e_sq_delta_given_x_j = (prob_y_is_0 * (d_0**2)) + (prob_y_is_1 * (d_1**2))

                drift_given_x[x_j[0]] = e_delta_given_x_j

                expected_delta_n += 0.25 * e_delta_given_x_j
                expected_squared_delta_n += 0.25 * e_sq_delta_given_x_j

            row = {
                'rollout_id': rollout_id,
                'step_n': n,
                'query_x': query_x,
                'prefix_y_ratio': y_ratio,
                'prefix_count_x_minus1': count_m1,
                'prefix_count_x_0': count_0,
                'prefix_count_x_1': count_1,
                'prefix_count_x_2': count_2,
                'g_target': g_n_minus_1_x_star,
                'entropy_target': entropy,
                'g_x_minus1': g_m1,
                'g_x_0': g_0,
                'g_x_1': g_1,
                'g_x_2': g_2,
                'expected_drift_bn': expected_delta_n,
                'expected_squared_drift_bn_prime': expected_squared_delta_n,
                'drift_given_x_minus1': drift_given_x[-1],
                'drift_given_x_0': drift_given_x[0],
                'drift_given_x_1': drift_given_x[1],
                'drift_given_x_2': drift_given_x[2]
            }

            writer.writerow(row)
            file.flush()  # a long rollout keeps its progress in the .tmp file

            print(f"  Eval step n={n:<3d} | Expected Drift b_n: {expected_delta_n:.8f}")


def rollout_path(outdir: Path, rollout_id: int) -> Path:
    return outdir / f"tabpfn_drift_diagnostics_raw_r{rollout_id}.csv"


def main() -> None:
    args = parse_args()
    set_threads(args.threads)
    # the CSVs are the cache here; config.json guards against mixing settings in one folder
    config = {"n_estimators": args.n_estimators, "total_n": args.total_n, "grid_size": args.grid_size,
              "seed": args.seed, "n0": args.n0, "eval_offset": args.eval_offset, "query_x": args.query_x,
              "beta": args.beta, "beta0": args.beta0}
    cache = RunCache(args.outdir, config, version=None)
    if args.overwrite:
        for f in args.outdir.glob("tabpfn_drift_diagnostics_raw_r*.csv"):
            f.unlink()
    cache.prepare(args.overwrite)

    eval_points = np.unique(np.geomspace(args.n0 + args.eval_offset, args.total_n, num=args.grid_size, dtype=int))
    todo = [r for r in range(1, args.n_rollouts + 1) if not rollout_path(args.outdir, r).exists()]
    print(f"{len(todo)} of {args.n_rollouts} rollouts to do in {args.outdir}")

    for r in todo:
        print(f"=== rollout {r} of {args.n_rollouts} ===")
        np.random.seed(args.seed + r - 1)
        X_seq, y_seq = generate_tabpfn_rollout(total_N=args.total_n, n_estimators=args.n_estimators, n_0=args.n0,
                                               beta=args.beta, beta0=args.beta0, device=args.device)
        path = rollout_path(args.outdir, r)
        tmp = path.with_suffix(".csv.tmp")
        tmp.unlink(missing_ok=True)
        compute_and_log_expected_drift(X_seq, y_seq, eval_points=eval_points, rollout_id=r, csv_filename=str(tmp),
                                       n_estimators=args.n_estimators, query_x=args.query_x, device=args.device)
        os.replace(tmp, path)   # only complete rollouts count as done
        print(f"[rollout {r}] {path}")


if __name__ == "__main__":
    main()
