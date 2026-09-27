"""
Microbenchmark: cost per pair update of point-blocked SGD, by block size and D.

Calls the Cython kernels directly -- one epoch per measurement, no estimator,
no scheduler, no scoring -- so the numbers are pure kernel cost. Run this
before Experiment N: it is minutes, not hours, and it answers three
questions the experiment's wall times cannot separate.

1. DOES THE BLOCKED KERNEL COST ANYTHING AT m=2? `blocked m=2 random
   sequential` is the same algorithm as `legacy` (bit-identical output, see
   tests/test_blocked_sgdmds.py). Any time difference is pure implementation
   overhead of the generalised kernel.

2. IS THE FUSED 4-POINT PASS WORTH IT? m=4 runs twice: `fused` (one pass over
   the features, six sums in registers, as SQuaD-MDS's kernel does) and
   `generic` (pairs outer, features inner; rows re-read from cache). If
   fused wins clearly, an unrolled m=8 kernel is worth writing; if not, the
   saving is all in memory traffic and the generic path is enough.

3. IS THE PARTITION SHUFFLE NEGLIGIBLE? partition vs random at the same m and
   D differs only by the Fisher-Yates shuffle (O(N) per N/m blocks).

Each arm reports ns per PAIR update, and a least-squares fit
ns/pair = a + b*D across the D grid -- the same form as the K-optimized
cost model (budget: ~52 ns + 0.6 ns x D). The model predicts b to fall as
2/(m-1) relative to legacy while rows are fetched from memory.

Usage:
    python bench_blocked_kernel.py                 # N=50,000, full D grid
    python bench_blocked_kernel.py --quick         # smoke test, seconds
    python bench_blocked_kernel.py --n 20000 --dims 8,784 --reps 3
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.manifold._sgd_mds_cython import (
    run_sgd_epoch_lazy_blocked,
    run_sgd_epoch_lazy_random_native,
)

OUT_DIR = Path(__file__).parent / "results" / "bench_blocked_kernel"

N_DEFAULT = 50_000
DIMS_DEFAULT = [3, 8, 64, 384, 784, 3072]
K_DEFAULT = 20        # pair updates per point in each timed epoch
REPS_DEFAULT = 5
LR = 0.1              # small, so the embedding barely moves between reps
SEED = 12345

UPDATE_CODES = {"sequential": 0, "sum": 1, "mean": 2}


def arms() -> List[Dict]:
    """Every (kernel, m, sampling, update, fused) combination measured."""
    out = [dict(kernel="legacy", m=2, sampling="random",
                update="sequential", fused=False)]
    for m in (2, 4, 8, 16):
        for sampling in ("random", "partition"):
            for update in ("sequential", "sum", "mean"):
                out.append(dict(kernel="blocked", m=m, sampling=sampling,
                                update=update, fused=(m == 4)))
                if m == 4:
                    out.append(dict(kernel="blocked", m=m,
                                    sampling=sampling, update=update,
                                    fused=False))
    return out


def label(a: Dict) -> str:
    if a["kernel"] == "legacy":
        return "legacy"
    tag = f"m{a['m']} {a['sampling']:<9} {a['update']:<10}"
    if a["m"] == 4:
        tag += " fused" if a["fused"] else " generic"
    return tag


def time_arm(a: Dict, X: np.ndarray, E0: np.ndarray, n_updates: int,
             reps: int) -> float:
    """Median seconds for one epoch of ``n_updates`` pair updates."""
    perm = np.arange(X.shape[0], dtype=np.int32)
    times = []
    for rep in range(reps + 1):          # rep 0 is a warm-up, discarded
        E = E0.copy()
        t0 = perf_counter()
        if a["kernel"] == "legacy":
            run_sgd_epoch_lazy_random_native(E, X, n_updates, LR, 0, SEED)
        else:
            run_sgd_epoch_lazy_blocked(
                E, X, n_updates, a["m"], a["sampling"] == "partition",
                UPDATE_CODES[a["update"]], LR, 0, SEED, perm,
                fused=a["fused"],
            )
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

    all_arms = arms()
    print(f"N={n:,}  D={dims}  {n_updates:,} pair updates per timed epoch  "
          f"median of {reps}  ({len(all_arms)} arms)")

    rows = []
    for D in dims:
        rs = np.random.RandomState(0)
        X = np.ascontiguousarray(rs.standard_normal((n, D)))
        E0 = np.ascontiguousarray(rs.standard_normal((n, 2)))
        print(f"\n--- D={D}  (X is {X.nbytes / 2**20:,.0f} MiB) ---")
        legacy_ns = None
        for a in all_arms:
            t = time_arm(a, X, E0, n_updates, reps)
            ns = 1e9 * t / n_updates
            if a["kernel"] == "legacy":
                legacy_ns = ns
            rows.append(dict(a, D=D, n_samples=n, n_updates=n_updates,
                             seconds=t, ns_per_pair=ns))
            print(f"  {label(a):<38} {ns:8.1f} ns/pair   "
                  f"{legacy_ns / ns:5.2f}x legacy")
        del X

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / ("results_quick.csv" if quick else "results.csv")
    df.to_csv(out_csv, index=False)

    # ---- cost model per arm: ns/pair = a + b*D ----
    if len(dims) >= 2:
        print("\n=== Cost model ns/pair = a + b*D (least squares over D) ===")
        leg = df[df["kernel"] == "legacy"]
        b_leg = np.polyfit(leg["D"], leg["ns_per_pair"], 1)[0]
        for a in all_arms:
            sub = df[(df["kernel"] == a["kernel"]) & (df["m"] == a["m"])
                     & (df["sampling"] == a["sampling"])
                     & (df["update"] == a["update"])
                     & (df["fused"] == a["fused"])]
            b, a0 = np.polyfit(sub["D"], sub["ns_per_pair"], 1)
            model = "" if a["kernel"] == "legacy" else (
                f"   b/b_legacy={b / b_leg:.2f}  "
                f"(memory model predicts 1/(m-1) = {1 / (a['m'] - 1):.2f})")
            print(f"  {label(a):<38} a={a0:7.1f} ns  b={b:6.3f} ns/dim"
                  + model)

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
