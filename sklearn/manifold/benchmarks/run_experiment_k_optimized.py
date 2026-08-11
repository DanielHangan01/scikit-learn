"""
Experiment K-optimized: the large-N budget-scaling protocol, restructured so
cluster time goes into the SGD fits being benchmarked -- not into scoring.

Same scientific design as Experiment K (run_experiment_k.py): per (dataset,
N): SGD-cycle baseline vs SGD-random at k*N updates/epoch, overhead =
budget_stress / cycle_stress. Three protocol changes, each removing a
dominant non-science cost that made plain K infeasible at cluster scale:

1. NO IN-FIT STRESS EVALUATION (compute_stress=False; timing is pure fit).
   Stress is observational -- nothing in the optimizer depends on it -- so
   the embedding is bit-identical. Plain K's fits scored themselves: an
   O(N^2) (lazy mode: O(N^2 D), via a full N x N distance matrix -- 80 GB at
   N=100k) evaluation *inside* the timed region, which both dominated
   measured time (up to ~97% at low D; the whole measured-vs-deployable
   gap of EXPERIMENT_K_RESULTS.md section 5) and re-introduced the O(N^2)
   memory the fast mode exists to avoid. Here each fit is scored ONCE,
   after it finishes, outside the timed region, with a row-blocked exact
   scorer (bench_utils.stress_exact_chunked: O(N * block) memory, no 20k
   cap, BLAS-parallel). Scoring time is recorded separately
   (time_scoring). => measured speedup IS the deployable speedup; the
   model-based a+b*k decomposition is no longer needed.

2. ONE CYCLE RUN, FIVE BUDGET RUNS (per (dataset, N, max_iter)). All arms
   fit the SAME seeded subsample; budget arms vary only the SGD seed.
   Justification from plain-K data at N=10k: cycle stress CV across 5 seeds
   is 0.07-0.90% per dataset (and that *includes* subsample variance --
   each K seed drew a different subsample; here the subsample is fixed, so
   the single-cycle denominator uncertainty is smaller still). Against
   overheads of +1..10% that is acceptable; raise N_CYCLE_REPEATS for
   tighter error bars at a specific rung. Note the tradeoff explicitly:
   overhead error bars now reflect budget-SGD noise only, not
   subsample-to-subsample variation (J/K already characterized that).

3. PER-FIT CHECKPOINT / RESUME. Plain K checkpointed per rung; at cluster
   scale a single fit is minutes-to-hours, so this runner appends to
   results.csv after EVERY fit and resumes at (max_iter, N, phase, k, seed)
   granularity. A killed job loses at most the one fit in flight.

Ladder: starts at 10,000 -- an OVERLAP ANCHOR with plain K (its top rung),
to validate the protocol change: anchor stress overheads should be
statistically consistent with K's (not identical: K paired each seed with
its own subsample; the anchor comparison is across that difference).
2k-7k rungs are NOT re-run -- plain K already covers them on all 8 sources.

MEMORY: the cycle FIT is the only large-memory phase (~36 N^2 bytes: matrix
+ explicit pair list + per-pair arrays + shuffle copies; calibrated to
N=20k, extrapolated above -- see run_experiment_k.py). Scoring no longer
adds an N x N matrix, and budget fits are O(N*D). MAX_MEM_GB gates cycle
fits only; a gated rung is skipped whole (no baseline -> no overhead).

Usage:
    python run_experiment_k_optimized.py                      # all DATASETS
    python run_experiment_k_optimized.py fashion_mnist_full   # subset
    python run_experiment_k_optimized.py --list

Results: results/experiment_k_optimized/<dataset>/{results.csv, meta.json}.
Columns as in K plus: time_scoring (untimed exact-stress evaluation),
subsample_seed. `seed` is the SGD seed (init + sampling); the subsample is
fixed per (dataset, N).
"""

from __future__ import annotations

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional

from bench_utils import (
    load_local_dataset,
    run_sgd_benchmark,
    run_random_budget_benchmark,
    stress_exact_chunked,
)
from run_experiment_j import intrinsic_dim_measures

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_k_optimized"

# Same 8 sources as plain K (see run_experiment_k.py for provenance/roles).
DATASETS = [
    "fashion_mnist_full",  # D=784   strain 0.37  N_full= 70,000
    "feynman_iii_15_12",   # D=3     strain 0.69  N_full=100,000
    "california_housing",  # D=8     strain 0.65  N_full= 20,640
    "patchcamelyon",       # D=512   strain 0.31  N_full=100,000
    "cifar10_raw",         # D=3072  strain 0.38  N_full= 50,000
    "ag_news",             # D=384   strain 0.08  N_full=100,000
    "tiny_imagenet",       # D=512   strain 0.07  N_full=100,000
    "mnist_digits",        # D=784   strain 0.12  N_full= 70,000
]

# 10,000 = overlap anchor with plain K; the rest are new territory.
# Capped at 50,000: the cycle baseline's fit needs ~36*N^2 bytes, so 50k is
# ~84 GB peak -- a realistic cluster-node budget -- while 70k (~164 GB) and
# 100k (~335 GB) are not. Sources larger than 50k (fashion_mnist_full,
# mnist_digits at 70k; feynman/patchcamelyon/ag_news/tiny_imagenet at 100k)
# are subsampled to 50k at the top rung.
N_LADDER = [10000, 20000, 35000, 50000]

K_VALUES = [50, 100, 200, 400]

MAX_ITER_VALUES = [30]

N_CYCLE_REPEATS = 1    # see protocol change 2 in the module docstring
N_BUDGET_REPEATS = 5

SEED_BASE = 42          # SGD seeds: SEED_BASE .. SEED_BASE + repeats - 1
SUBSAMPLE_SEED = 42     # one fixed subsample per (dataset, N)

SGD_SWITCH = 0.5
SGD_EPSILON = 0.001
SGD_LR = 0.5

SCORING_BLOCK = 2048    # row-block size for stress_exact_chunked

# Skip (don't attempt) any rung whose estimated cycle-fit peak memory
# exceeds this many GB. None = no limit. Set to your node's usable RAM.
MAX_MEM_GB: Optional[float] = None

TARGETS = [1.05, 1.10, 1.15]


def make_ladder(N_full: int) -> List[int]:
    """Ladder rungs <= full N, capped at N_LADDER[-1] (the 50k cap).

    A source whose full N fits under the cap gets its true full N as the top
    rung: replacing the nearest rung when within 5% (california_housing's
    20,640 replaces the 20,000 rung -- same cost, whole dataset) or appended
    otherwise. Sources above the cap are subsampled to the capped ladder.
    """
    rungs = [n for n in N_LADDER if n <= N_full]
    if not rungs:
        return [N_full]
    if N_full <= N_LADDER[-1] and N_full not in rungs:
        if N_full < 1.05 * rungs[-1]:
            rungs[-1] = N_full
        else:
            rungs.append(N_full)
    return sorted(set(rungs))


def valid_k(N: int) -> List[int]:
    cap = (N - 1) // 2
    return [k for k in K_VALUES if k < cap]


def _estimate_cycle_peak_gb(N: int) -> float:
    """~36*N^2 bytes; calibrated to N=20k, extrapolated above (see K)."""
    return 36.0 * (N ** 2) / (1024 ** 3)


def _k_star(ks, overheads, target):
    ks = np.asarray(ks, float)
    excess = np.asarray(overheads, float) - 1.0
    tgt = target - 1.0
    order = np.argsort(ks)
    ks, excess = ks[order], excess[order]
    if excess.min() > tgt:
        return None
    if excess.max() <= tgt:
        return float(ks[0])
    lk = np.log(ks)
    lexc = np.log(np.clip(excess, 1e-9, None))
    return float(np.exp(np.interp(np.log(tgt), lexc[::-1], lk[::-1])))


def run_dataset(dataset: str) -> None:
    out_dir = RESULTS_DIR / dataset
    out_csv = out_dir / "results.csv"
    out_meta = out_dir / "meta.json"
    if out_csv.exists() and out_meta.exists():
        print(f"Skipping {dataset} (results already exist).")
        return

    print(f"\n=== Experiment K-optimized: {dataset} ===")
    X_full, _ = load_local_dataset(dataset)
    N_full, D = X_full.shape
    ladder = make_ladder(N_full)
    print(f"  full N={N_full}, D={D}")
    print(f"  N ladder: {ladder}")
    if MAX_MEM_GB is not None:
        print(f"  MAX_MEM_GB={MAX_MEM_GB:g} -- cycle rungs exceeding this "
              f"estimated peak will be skipped")
    for N in ladder:
        est = _estimate_cycle_peak_gb(N)
        flag = "  <-- exceeds MAX_MEM_GB, rung will be SKIPPED" \
            if (MAX_MEM_GB is not None and est > MAX_MEM_GB) else ""
        print(f"    N={N:>7,}  est. cycle-fit peak ~{est:>7.1f} GB{flag}")

    idm = intrinsic_dim_measures(X_full)
    print(f"  intrinsic dim: PR={idm['pca_participation_ratio']:.1f}  "
          f"strain_top2={idm['strain_top2_frac']:.3f}")

    # ---- per-fit resume ----------------------------------------------------
    rows: List[Dict[str, Any]] = []
    done = set()  # (max_iter, N, phase, k_or_-1, seed)
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        for r in rows:
            k_key = -1 if pd.isna(r["k"]) else int(r["k"])
            done.add((int(r["max_iter"]), int(r["n_samples"]),
                      r["phase"], k_key, int(r["seed"])))
        print(f"  Resuming: {len(done)} completed fit(s) on disk.")
    run_id = len(rows)

    def checkpoint():
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    def add_and_checkpoint(phase, N, k, n_updates, max_iter, seed,
                           time_fit, time_scoring, stress):
        nonlocal run_id
        rid = f"run_{run_id:06d}"; run_id += 1
        rows.append({
            "run_id": rid, "dataset": dataset,
            "n_samples": N, "n_features": D,
            "phase": phase, "k": k,
            "n_updates_per_epoch": n_updates, "max_iter": max_iter,
            "seed": seed, "subsample_seed": SUBSAMPLE_SEED,
            "time_solver": float(time_fit),
            "time_scoring": float(time_scoring),
            "stress": float(stress),
        })
        checkpoint()

    common = dict(switch_ratio=SGD_SWITCH, epsilon=SGD_EPSILON, lr=SGD_LR,
                  compute_stress=False)

    any_mem_skipped = False

    for max_iter in MAX_ITER_VALUES:
        for N in ladder:
            est_gb = _estimate_cycle_peak_gb(N)
            if MAX_MEM_GB is not None and est_gb > MAX_MEM_GB:
                print(f"\n  --- max_iter={max_iter}  N={N:,}  SKIPPED: "
                      f"est. cycle peak ~{est_gb:.1f} GB > "
                      f"MAX_MEM_GB={MAX_MEM_GB:g} ---")
                any_mem_skipped = True
                continue
            ks = valid_k(N)
            print(f"\n  --- max_iter={max_iter}  N={N:,}  (k in {ks}) ---")

            # One fixed subsample per (dataset, N), shared by every arm.
            if N < N_full:
                sub_rng = np.random.RandomState(SUBSAMPLE_SEED)
                idx = sub_rng.choice(N_full, size=N, replace=False)
                X = np.ascontiguousarray(X_full[idx], dtype=np.float64)
            else:
                X = np.ascontiguousarray(X_full, dtype=np.float64)

            def fit_and_score(phase, k, seed):
                """Run one fit (untimed scoring afterwards) + checkpoint."""
                if phase == "cycle":
                    r = run_sgd_benchmark(X, dissimilarity="euclidean",
                                          random_state=seed,
                                          max_iter=max_iter, **common)
                    n_upd = N * (N - 1) // 2
                else:
                    r = run_random_budget_benchmark(
                        X, n_updates_per_epoch=k * N, random_state=seed,
                        max_iter=max_iter, **common)
                    n_upd = k * N
                t0 = time()
                stress = stress_exact_chunked(X, r["embedding"],
                                              block_size=SCORING_BLOCK)
                t_score = time() - t0
                add_and_checkpoint(phase, N, k, n_upd, max_iter, seed,
                                   r["time"], t_score, stress)

            for rep in range(N_CYCLE_REPEATS):
                seed = SEED_BASE + rep
                if (max_iter, N, "cycle", -1, seed) in done:
                    print(f"    cycle seed={seed}: already done, skip")
                    continue
                fit_and_score("cycle", None, seed)

            for k in ks:
                for rep in range(N_BUDGET_REPEATS):
                    seed = SEED_BASE + rep
                    if (max_iter, N, "budget", k, seed) in done:
                        print(f"    budget k={k} seed={seed}: already done, skip")
                        continue
                    fit_and_score("budget", k, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    if any_mem_skipped:
        print(f"\n  NOTE: rung(s) skipped due to MAX_MEM_GB -- meta.json NOT "
              f"written; re-running (e.g. with a higher limit) resumes "
              f"per-fit from results.csv.")
    else:
        meta = {
            "dataset": dataset, "n_samples": N_full, "n_features": D,
            "ladder": ladder, "k_values": K_VALUES,
            "max_iter_values": MAX_ITER_VALUES,
            "n_cycle_repeats": N_CYCLE_REPEATS,
            "n_budget_repeats": N_BUDGET_REPEATS,
            "subsample_seed": SUBSAMPLE_SEED,
            "protocol": "optimized: compute_stress=False in fits; exact "
                        "chunked scoring outside timing; single shared "
                        "subsample per (dataset, N)",
            "max_mem_gb": MAX_MEM_GB,
            **idm,
        }
        out_meta.write_text(json.dumps(meta, indent=2))

    # ---- summary: overhead + (now uncontaminated) speedup ------------------
    df = pd.DataFrame(rows)
    mi = 30 if 30 in MAX_ITER_VALUES else MAX_ITER_VALUES[0]
    print(f"\n  === Summary: {dataset} (budget median over "
          f"{N_BUDGET_REPEATS} SGD seeds vs {N_CYCLE_REPEATS} cycle run(s), "
          f"max_iter={mi}) ===")
    if df.empty:
        print("    (no fits completed)")
        return
    star = {t: ([], []) for t in TARGETS}
    for N in ladder:
        rung = df[(df["n_samples"] == N) & (df["max_iter"] == mi)]
        if rung.empty:
            print(f"    N={N:>7,} | (skipped)")
            continue
        cyc = rung[rung["phase"] == "cycle"]
        cyc_stress = cyc["stress"].median()
        cyc_time = cyc["time_solver"].median()
        bud = rung[rung["phase"] == "budget"]
        ks, ov, sp = [], [], []
        for k, grp in bud.groupby("k"):
            ks.append(int(k))
            ov.append(grp["stress"].median() / cyc_stress)
            sp.append(cyc_time / grp["time_solver"].median())
        order = np.argsort(ks)
        ks = list(np.array(ks)[order])
        ov = list(np.array(ov)[order])
        sp = list(np.array(sp)[order])
        cells = "  ".join(f"k{kk}={oo:.3f}x/{ss:.1f}x"
                          for kk, oo, ss in zip(ks, ov, sp))
        print(f"    N={N:>7,} | cycle {cyc_time:7.1f}s | "
              f"overhead/speedup: {cells}")
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


def main():
    args = sys.argv[1:]
    if args and args[0] == "--list":
        print("Experiment K-optimized datasets:")
        for ds in DATASETS:
            print(f"  {ds}")
        print(f"\nN ladder: {N_LADDER}")
        print(f"k grid:   {K_VALUES}")
        print(f"max_iter: {MAX_ITER_VALUES}")
        print(f"repeats:  cycle={N_CYCLE_REPEATS}, budget={N_BUDGET_REPEATS}")
        print(f"MAX_MEM_GB: {MAX_MEM_GB}")
        print("\nEstimated cycle-fit peak memory per rung (~36*N^2 bytes; "
              "extrapolated above N=20,000):")
        for N in N_LADDER:
            print(f"    N={N:>7,}  ~{_estimate_cycle_peak_gb(N):>7.1f} GB")
        return
    targets = args or DATASETS
    for ds in targets:
        run_dataset(ds)


if __name__ == "__main__":
    main()
