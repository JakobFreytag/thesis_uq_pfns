#!/usr/bin/env bash
# Reproduces the tables and figures of the thesis into results/artifacts/ (see the README, "Thesis <-> code").
#
#   ./reproduce.sh          fit everything that is not cached yet, then write all tables and figures
#   ./reproduce.sh plot     only rebuild the tables and figures from the existing caches (no TabPFN)
#   ./reproduce.sh quick    test run of the whole pipeline with n_estimators = 1, into results/artifacts/quick/
#
# Not included: the quasi-martingale diagnostics (Sec. 4.1, App. A.2, A.4, A.5). Their 20 rollouts are committed
# in results/tabpfn_diagnostics_rollouts/ and the figures come from tabpfn_diagnostics.ipynb; recomputing the
# rollouts (scripts/quasi_martingale.py) takes far longer than everything below together.
#
# The fits are expensive (TabPFN on every prefix of every dataset). They are cached per repetition, so an
# interrupted run resumes where it stopped. Run from an environment with the dependencies installed
# (e.g. `source .venv/bin/activate`), or set PYTHON=/path/to/python.
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-all}"
PY="${PYTHON:-python}"
OUTDIR=results/artifacts

case "$MODE" in
    all | plot) RUN="$MODE" ;;
    quick) RUN=all; OUTDIR=results/artifacts/quick; export UQPFN_N_ESTIMATORS=1 ;;
    *) echo "usage: $0 [all | plot | quick]" >&2; exit 1 ;;
esac

run() {
    echo
    echo ">>> $*"
    "$PY" "$@" --run "$RUN" --outdir "$OUTDIR"
}

# Sec. 4.3: Tab. 4.1, Fig. 4.5; App. A.6: Fig. A.4 (n = 500, 1000), Fig. A.5
run scripts/coverage/n_estimators_linear.py
run scripts/coverage/n_estimators_linear.py --n-train-list 500 1000 --n-reps 1 --plot-n 500
run scripts/coverage/n_estimators_linear.py --n-train-list 500 1000 --n-reps 1 --plot-n 1000
run scripts/coverage/n_estimators_spline.py

# Sec. 4.4.1: Tab. 4.2, Fig. 4.6; App. A.7: Fig. A.6
run scripts/coverage/linear_dependent.py

# Sec. 4.4.2: Tab. 4.3, Fig. 4.7, 4.8; App. A.7: Tab. A.1, A.2
run scripts/coverage/spline.py
run scripts/coverage/spline.py --mode balanced

# Sec. 4.4.3: Tab. 4.4, 4.5; App. A.7: Tab. A.3-A.5
run scripts/coverage/poisson_probit.py

# Sec. 4.5.1: Fig. 4.9; App. A.8: Fig. A.7, A.8
run scripts/entropy_ud/replication_logistic.py
run scripts/entropy_ud/replication_moons.py

# Sec. 4.5.2: Fig. 4.10, 4.11; App. A.8: Tab. A.6
run scripts/entropy_ud/frequency.py
run scripts/entropy_ud/logistic_optimum.py

# Sec. 4.5.3: Fig. 4.12, 4.13; Sec. 4.5.4: Fig. 4.14-4.17
run scripts/entropy_ud/replication_spirals.py
run scripts/entropy_ud/multiclass_blops.py

echo
echo "Done: tables and figures in $OUTDIR/"
