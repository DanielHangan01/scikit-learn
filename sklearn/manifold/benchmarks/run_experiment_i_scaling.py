"""
Experiment I-scaling: how must the budget k scale with N to hold stress
overhead fixed?

Experiment I established that at *fixed* k the overhead grows with N, but
it could not isolate the scaling law because dataset *size* and dataset
*difficulty* were confounded across the seven datasets. This follow-up
removes the confound: it takes a SINGLE source dataset and subsamples it
to a geometric ladder of N values, so intrinsic dimensionality is held
(approximately) fixed and only N changes. Regressing the interpolated
k*(target overhead) against N then gives a clean exponent p in

    k*  ~  N^p          (p=0  -> constant k suffices;
                          p=1  -> k must grow linearly, budget ~ N^2).

We run it on more than one source so the exponent itself can be compared
across intrinsic dimensionalities (hypothesis: higher intrinsic dim ->
larger p, closer to linear).

For each (source, N, seed):
  - draw a row-subsample of size N (seeded),
  - SGD-cycle baseline on that subsample (the overhead denominator),
  - SGD-random at n_updates_per_epoch = k*N for each k.

Overhead at (N, k) = median_seed( random_stress / cycle_stress ).

Saves results.csv per source under results/experiment_i_scaling/<source>/.
Skip-if-exists (matching the other experiments).

Usage:
    python run_experiment_i_scaling.py                     # default sources
    python run_experiment_i_scaling.py fashion_mnist       # subset
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

from bench_utils import (
    load_local_dataset,
    run_sgd_benchmark,
    run_random_budget_benchmark,
)

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_i_scaling"

# One moderate-dim source (steep overhead-vs-k in I) and one high-dim
# source (nearly floored in I). If the exponent tracks intrinsic
# dimensionality, hiva should regress steeper (larger p) than fashion_mnist.
DEFAULT_SOURCES = ["fashion_mnist", "hiva"]

# Geometric N ladder (ratio ~1.5), capped at the source's full N.
N_LADDER = [400, 600, 900, 1350, 2000, 3000]

# Budget grid; k is filtered per-N so the budget never exceeds the cycle
# budget N(N-1)/2 (i.e. k < (N-1)/2).
K_VALUES = [10, 25, 50, 100, 200]

MAX_ITER = 30
N_REPEATS = 5
SEED_BASE = 42

SGD_SWITCH = 0.5
SGD_EPSILON = 0.001
SGD_LR = 0.5

# Targets for the k* interpolation reported in the console summary.
TARGETS = [1.05, 1.10, 1.15]


def _valid_k(N: int) -> List[int]:
    cap = (N - 1) // 2
    return [k for k in K_VALUES if k < cap]


def _k_star(ks, overheads, target):
    """Smallest k (log-log interpolated) reaching overhead <= target."""
    ks = np.asarray(ks, float)
    excess = np.asarray(overheads, float) - 1.0
    tgt = target - 1.0
    order = np.argsort(ks)
    ks, excess = ks[order], excess[order]
    if excess.min() > tgt:
        return None              # never reaches target in range
    if excess.max() <= tgt:
        return float(ks[0])      # already there at smallest k
    lk = np.log(ks)
    lexc = np.log(np.clip(excess, 1e-9, None))
    # excess decreasing in k; invert (lexc -> lk) at log(tgt)
    return float(np.exp(np.interp(np.log(tgt), lexc[::-1], lk[::-1])))


def run_source(source: str) -> None:
    out_dir = RESULTS_DIR / source
    out_csv = out_dir / "results.csv"
    if out_csv.exists():
        print(f"Skipping {source} (results already exist).")
        return

    print(f"\n=== Experiment I-scaling: source={source} ===")
    X_full, _ = load_local_dataset(source)
    N_full, D = X_full.shape
    ladder = [n for n in N_LADDER if n <= N_full]
    if N_full not in ladder and N_full <= max(N_LADDER):
        ladder.append(N_full)
    print(f"  full N={N_full}, D={D}")
    print(f"  N ladder: {ladder}")

    rows: List[Dict[str, Any]] = []
    run_id = 0

    def add_row(phase, N, k, n_updates, seed, time_solver, stress):
        nonlocal run_id
        rid = f"run_{run_id:06d}"; run_id += 1
        rows.append({
            "run_id": rid, "source": source,
            "n_samples": N, "n_features": D,
            "phase": phase, "k": k,
            "n_updates_per_epoch": n_updates,
            "max_iter": MAX_ITER, "seed": seed,
            "time_solver": float(time_solver),
            "stress": float(stress),
        })

    common = dict(max_iter=MAX_ITER, switch_ratio=SGD_SWITCH,
                  epsilon=SGD_EPSILON, lr=SGD_LR)

    for N in ladder:
        ks = _valid_k(N)
        print(f"\n  --- N={N}  (k in {ks}) ---")
        for rep in range(N_REPEATS):
            seed = SEED_BASE + rep
            # Seeded row subsample: same subset for cycle and all k within
            # this (N, seed) so overhead is measured on identical data.
            sub_rng = np.random.RandomState(seed)
            idx = sub_rng.choice(N_full, size=N, replace=False)
            X = np.ascontiguousarray(X_full[idx], dtype=np.float64)

            rc = run_sgd_benchmark(X, dissimilarity="euclidean",
                                   random_state=seed, **common)
            add_row("cycle", N, None, N * (N - 1) // 2, seed,
                    rc["time"], rc["stress"])

            for k in ks:
                rb = run_random_budget_benchmark(
                    X, n_updates_per_epoch=k * N,
                    random_state=seed, **common)
                add_row("budget", N, k, k * N, seed,
                        rb["time"], rb["stress"])

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # ---- summary: overhead(k) per N, k*(target), exponent fit ----
    df = pd.DataFrame(rows)
    print(f"\n  === Summary: {source} (median over {N_REPEATS} seeds) ===")
    star = {t: ([], []) for t in TARGETS}   # target -> (Ns, k*s)
    for N in ladder:
        cyc = df[(df["phase"] == "cycle") & (df["n_samples"] == N)]["stress"].median()
        bud = df[(df["phase"] == "budget") & (df["n_samples"] == N)]
        ks, ov = [], []
        for k, grp in bud.groupby("k"):
            ks.append(int(k)); ov.append(grp["stress"].median() / cyc)
        order = np.argsort(ks)
        ks = list(np.array(ks)[order]); ov = list(np.array(ov)[order])
        ov_str = "  ".join(f"k{kk}={oo:.3f}x" for kk, oo in zip(ks, ov))
        print(f"    N={N:5d} | {ov_str}")
        for t in TARGETS:
            ks_star = _k_star(ks, ov, t)
            if ks_star is not None:
                star[t][0].append(N); star[t][1].append(ks_star)

    print(f"\n  -- k*(N) and exponent fit  (k* ~ N^p) --")
    for t in TARGETS:
        Ns, kstars = star[t]
        line = "  ".join(f"N{N}:k*={ks:.0f}" for N, ks in zip(Ns, kstars))
        if len(Ns) >= 3:
            p = np.polyfit(np.log(Ns), np.log(kstars), 1)[0]
            print(f"    target {t:.2f}x:  {line}   => p={p:.2f}")
        else:
            print(f"    target {t:.2f}x:  {line}   (too few points to fit)")

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    targets = sys.argv[1:] or DEFAULT_SOURCES
    for src in targets:
        run_source(src)
