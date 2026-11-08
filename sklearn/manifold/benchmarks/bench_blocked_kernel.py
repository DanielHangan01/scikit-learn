"""
Microbenchmark: cost per pair update of the random and blocked kernels, by D.

Calls the Cython kernels directly -- one epoch per measurement, no estimator,
no scheduler, no scoring -- so the numbers are pure kernel cost. `random` is
plain pair sampling; `blocked m=2` is the same algorithm (bit-identical output)
through the blocked kernel, so their difference is implementation overhead.

Each arm reports ns per PAIR update, and a least-squares fit
ns/pair = a + b*D across the D grid.

Usage:
    python bench_blocked_kernel.py                 # N=50,000, full D grid
    python bench_blocked_kernel.py --quick         # smoke test, seconds
    python bench_blocked_kernel.py --n 20000 --dims 8,784 --reps 3
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.manifold._sgd_mds_cython import (
    run_sgd_epoch_blocked,
    run_sgd_epoch_random,
)

OUT_DIR = (
    Path(os.environ.get("SGDMDS_RESULTS_ROOT", Path(__file__).parent / "results"))
    / "bench_kernels"
)

N_DEFAULT = 50_000
DIMS_DEFAULT = [3, 8, 64, 384, 784, 3072]
K_DEFAULT = 20        # pair updates per point in each timed epoch
REPS_DEFAULT = 5
LR = 0.1              # small, so the embedding barely moves between reps
SEED = 12345

ARMS = ["random", "blocked m=2", "blocked m=4", "blocked m=8", "blocked m=16"]


def time_arm(arm: str, X: np.ndarray, E0: np.ndarray, n_updates: int,
             reps: int) -> float:
    """Median seconds for one epoch of ``n_updates`` pair updates."""
    times = []
    for rep in range(reps + 1):          # rep 0 is a warm-up, discarded
        E = E0.copy()
        t0 = perf_counter()
        if arm == "random":
            run_sgd_epoch_random(E, X, n_updates, LR, 0, SEED)
        else:
            m = int(arm.split("=")[1])
            run_sgd_epoch_blocked(E, X, n_updates, m, LR, 0, SEED)
        if rep:
            times.append(perf_counter() - t0)
    return float(np.median(times))


def main():
    args = sys.argv[1:]

    def take(flag, cast, default):
        if flag in args:
            i = args.index(flag)
            val = cast(args[i + 1])
            del args[i:i + 2]
            return val
        return default

    quick = "--quick" in args
    n = take("--n", int, 5_000 if quick else N_DEFAULT)
    dims = take("--dims", lambda s: [int(v) for v in s.split(",")],
                [3, 64] if quick else DIMS_DEFAULT)
    reps = take("--reps", int, 1 if quick else REPS_DEFAULT)
    k = take("--k", int, K_DEFAULT)
    n_updates = k * n

    out_csv = OUT_DIR / ("results_quick.csv" if quick else "results.csv")
    if out_csv.exists():
        raise SystemExit(f"{out_csv} already exists; move it first.")

    print(f"N={n:,}  D={dims}  {n_updates:,} pair updates per timed epoch  "
          f"median of {reps}")

    rows = []
    for D in dims:
        rs = np.random.RandomState(0)
        X = np.ascontiguousarray(rs.standard_normal((n, D)))
        E0 = np.ascontiguousarray(rs.standard_normal((n, 2)))
        print(f"\n--- D={D}  (X is {X.nbytes / 2**20:,.0f} MiB) ---")
        for arm in ARMS:
            t = time_arm(arm, X, E0, n_updates, reps)
            ns = 1e9 * t / n_updates
            rows.append(dict(arm=arm, D=D, n_samples=n, n_updates=n_updates,
                             seconds=t, ns_per_pair=ns))
            print(f"  {arm:<14} {ns:8.1f} ns/pair")
        del X

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if len(dims) >= 2:
        print("\n=== Cost model ns/pair = a + b*D (least squares over D) ===")
        for arm in ARMS:
            sub = df[df["arm"] == arm]
            b, a0 = np.polyfit(sub["D"], sub["ns_per_pair"], 1)
            print(f"  {arm:<14} a={a0:7.1f} ns  b={b:6.3f} ns/dim")

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
