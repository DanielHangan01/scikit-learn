"""
Experiment K: does the budgeted-random fast mode hold up at LARGE N?

Experiments I / I-scaling / J characterised the budget knob up to N~=5750
(epileptic). J's headline for *responsive* data (high strain_top2, e.g.
fashion_mnist) is that the budget k* needed to hold a fixed stress overhead
is near-constant in N (p ~= 0.1). Experiment K stress-tests that prediction
by pushing large sources up a taller N-ladder.

This is structurally Experiment I-scaling (run_experiment_i_scaling.py): one
source, subsampled to a geometric N-ladder so intrinsic dimensionality is
held fixed and only N varies; SGD-cycle baseline + SGD-random at k*N per
(N, seed); overhead(N, k) = median_seed(random_stress / cycle_stress).

    k*  ~  N^p       (p=0 -> a fixed k holds quality at any N -- the win)

WHAT LIMITS N: two O(N^2) costs, both from the CYCLE BASELINE, not the fast
mode itself (which is O(N*D)):
  1. Exact stress SCORING (SGDMDS._safe_stress) -- both arms are scored
     exactly so the overhead ratio is meaningful, but that means every fit
     pays an O(N^2) evaluation. `_safe_stress` skips scoring (returns -1.0)
     above `sklearn.manifold._sgd_mds.EXACT_STRESS_N_CAP` (default 20,000) --
     a *safety guard*, not a hard limit. This runner raises it (see
     `_configure_stress_cap()` below) so results.STRESS_CAP is the effective
     ceiling on a machine that can afford the memory.
  2. RAM for the cycle baseline itself: the N x N distance matrix, the
     explicit N^2/2 pair list, and per-pair target/weight arrays -- peak
     ~36*N^2 bytes (calibrated against observed OOM behaviour up to N=20k;
     see MEMORY below for the table and the large-N caveat).
On a laptop/workstation (~16 GB RAM) the practical ceiling is ~10k. On a
cluster node with substantially more RAM, N_LADDER and STRESS_CAP below can
be raised to each dataset's own full N (up to 100,000 for several sources
here) -- see MEMORY for what that costs and MAX_MEM_GB for a safety net.

MULTIPLE DATASETS: DATASETS is a plain list -- add any large source here
(digit MNIST, a large real dataset, etc.). Each is subsampled independently
to the shared ladder (rungs <= its full N, and <= STRESS_CAP), so the
per-dataset exponent p and overhead curves are directly comparable, and the
per-dataset intrinsic-dim scalars (reused from Experiment J) place each new
source in the responsive / hard-middle / degenerate taxonomy.

Prep for a new source: drop datasets/<name>/X.npy (+ optional y.npy) in the
usual format; pool any train/test split (the split is meaningless for MDS)
and subsample with a fixed seed. See the fashion_mnist_full note below.

MEMORY -- the real ceiling is RAM, not STRESS_CAP. The cycle baseline
(euclidean mode) does not just store the N x N distance matrix (8 N^2); it
also materialises the full explicit pair list (N^2/2 pairs) plus per-pair
target-distance and weight arrays, plus transient buffers during pair
construction and scoring -- peak is ~36*N^2 bytes, calibrated against
observed OOM behaviour up to N=20,000:
    N=  7000 -> ~1.8 GB     N= 10000 -> ~3.6 GB     N= 20000 -> ~14.4 GB
    N= 35000 -> ~44.1 GB    N= 50000 -> ~90.0 GB    N= 70000 -> ~176.4 GB
    N=100000 -> ~360.0 GB
The 36*N^2 coefficient is EXTRAPOLATED beyond N=20,000 (numpy internals used
by the pair-construction path, e.g. `np.triu_indices`, may add further
transient overhead at very large N that was not exercised by the calibration
run) -- treat the >=35k figures as an estimate, not a guarantee. Recommended
practice on a new / unfamiliar node: run one dataset up to an intermediate
rung first (e.g. 35k) and watch peak RSS before committing the full ladder to
a large batch allocation.

On a 16 GB machine (~7 GB free) the safe ceiling is ~10k -- 15k gets
OOM-killed. On a large-memory cluster node, raise N_LADDER and STRESS_CAP to
match your node's RAM (each dataset's own full N is the natural ceiling --
see the DATASETS comments for source sizes up to 100,000). Set MAX_MEM_GB to
your node's usable RAM and the runner will SKIP (not attempt, not crash) any
rung whose estimated peak would exceed it -- safer than an OOM kill mid-batch,
since it burns no allocation time on a doomed rung and leaves the rest of the
ladder to run. The runner CHECKPOINTS results.csv after every rung and
resumes per-rung, so a kill loses at most the rung in flight (meta.json is
written only when the run completes).

Saves results.csv + meta.json per dataset under results/experiment_k/<name>/.

Usage:
    python run_experiment_k.py                      # all DATASETS
    python run_experiment_k.py fashion_mnist_full   # subset
    python run_experiment_k.py --list
"""

from __future__ import annotations

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench_utils import (
    load_local_dataset,
    run_sgd_benchmark,
    run_random_budget_benchmark,
)
# Reuse J's intrinsic-dimensionality measurement verbatim so each large
# source is placed in the same regime taxonomy (strain_top2, pca_PR, ...).
from run_experiment_j import intrinsic_dim_measures

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_k"

# Large sources to test. fashion_mnist_full is the full 70k Fashion-MNIST
# (pool train+test, /255 to [0,1]); the rest are wired from the bulk
# collections by prep_shortlist_datasets.py (symlinked into datasets/<name>/).
# Ordered cheapest-first (cost ~ N^2 * D), so low-D results land early.
#
# strain_top2 (standardized) and role annotated; N_full = the source's own
# size -- the natural ceiling for N_LADDER/STRESS_CAP on a large-RAM node
# (see MEMORY in the module docstring for what running at N_full costs).
DATASETS = [
    # --- responsive, spanning D: tests the D-vs-speed axis (H-K3) ---
    "fashion_mnist_full",  # D=784   strain 0.37  N_full= 70,000  (anchor)
    "feynman_iii_15_12",   # D=3     strain 0.69  N_full=100,000  extreme low-D  -> biggest speedup
    "california_housing",  # D=8     strain 0.65  N_full= 20,640  low-D, real
    "patchcamelyon",       # D=512   strain 0.31  N_full=100,000  mid-D image
    "cifar10_raw",         # D=3072  strain 0.38  N_full= 50,000  very high-D    -> smallest speedup
    # --- hard-middle, large N: should NOT reach constant-k (H-K4) ---
    "ag_news",             # D=384   strain 0.08  N_full=100,000  text
    "tiny_imagenet",       # D=512   strain 0.07  N_full=100,000  image
    "mnist_digits",        # D=784   strain 0.12  N_full= 70,000  digit MNIST
]

# Ladder bridges from J's range (max N=5750) up through each dataset's own
# full N (make_ladder() caps per-source at min(full N, STRESS_CAP)). On a
# workstation, lower this back to [2000, 4000, 7000, 10000] (the original
# 16 GB-safe ladder) or set MAX_MEM_GB below to let the runner self-limit.
N_LADDER = [2000, 4000, 7000, 10000, 20000, 35000, 50000, 70000, 100000]

# Estimator ceiling: SGDMDS._safe_stress returns -1.0 above this many samples
# (stress not measured -> overhead undefined). Raised here (and pushed into
# the estimator via _configure_stress_cap() below) to cover every DATASETS
# source's full N -- the runner is the thing deciding "how large a fit are we
# prepared to score exactly", not the estimator's conservative laptop default.
STRESS_CAP = 120_000

# Optional safety net for cluster runs: if set (GB), any rung whose estimated
# peak memory (see _estimate_cycle_peak_gb) exceeds this is SKIPPED with a
# clear message instead of attempted -- avoids an OOM kill burning allocation
# time mid-batch. None = no limit (attempt every rung the ladder produces).
# Set this to your node's usable RAM (leave headroom below the hard limit).
MAX_MEM_GB: Optional[float] = None


def _configure_stress_cap() -> None:
    """Raise the estimator's exact-stress safety guard to match STRESS_CAP.

    SGDMDS._safe_stress defaults to skipping scoring above 20,000 samples --
    a conservative guard for typical (non-cluster) use, not a technical
    limit. This runner explicitly opts in to scoring up to STRESS_CAP.
    """
    import sklearn.manifold._sgd_mds as _sgd_mds_mod
    if _sgd_mds_mod.EXACT_STRESS_N_CAP < STRESS_CAP:
        print(f"  [config] raising SGDMDS exact-stress cap "
              f"{_sgd_mds_mod.EXACT_STRESS_N_CAP:,} -> {STRESS_CAP:,}")
        _sgd_mds_mod.EXACT_STRESS_N_CAP = STRESS_CAP


def _estimate_cycle_peak_gb(N: int) -> float:
    """Calibrated peak-memory estimate for one cycle-baseline fit at size N.

    ~36*N^2 bytes (distance matrix + explicit pair list + target/weight
    arrays + transient scoring/pair-construction buffers), calibrated against
    observed OOM behaviour up to N=20,000. EXTRAPOLATED above that -- see the
    module docstring's MEMORY section for the caveat.
    """
    return 36.0 * (N ** 2) / (1024 ** 3)

# Budget grid. Dropped the cheap low-k end (10, 25) that dominated the small-N
# experiments; added 400 to pin the responsive-regime floor at large N.
# Filtered per-N so k < (N-1)/2 (budget below the full-cycle pair count).
K_VALUES = [50, 100, 200, 400]

# max_iter=30 is the scaling reference (fashion is responsive -> needs the
# full iteration budget; see J's mutually-exclusive-levers result). Append 15
# to also probe the iteration lever at scale -- doubles the expensive 20k
# baseline cost.
MAX_ITER_VALUES = [30]

N_REPEATS = 5
SEED_BASE = 42

SGD_SWITCH = 0.5
SGD_EPSILON = 0.001
SGD_LR = 0.5

# Targets for the k* interpolation reported in the console summary.
TARGETS = [1.05, 1.10, 1.15]


def make_ladder(N_full: int) -> List[int]:
    """Rungs <= full N and <= STRESS_CAP, plus full N when it fits."""
    rungs = [n for n in N_LADDER if n <= min(N_full, STRESS_CAP)]
    # Include the source's own full N as a top rung when it is small enough to
    # (a) not already be represented and (b) still be scoreable.
    if N_full <= STRESS_CAP and N_full not in rungs:
        rungs.append(N_full)
    if not rungs:
        rungs = [min(N_full, STRESS_CAP)]
    return sorted(set(rungs))


def valid_k(N: int) -> List[int]:
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
        return None
    if excess.max() <= tgt:
        return float(ks[0])
    lk = np.log(ks)
    lexc = np.log(np.clip(excess, 1e-9, None))
    return float(np.exp(np.interp(np.log(tgt), lexc[::-1], lk[::-1])))


def run_dataset(dataset: str) -> None:
    _configure_stress_cap()

    out_dir = RESULTS_DIR / dataset
    out_csv = out_dir / "results.csv"
    out_meta = out_dir / "meta.json"
    if out_csv.exists() and out_meta.exists():
        print(f"Skipping {dataset} (results already exist).")
        return

    print(f"\n=== Experiment K: {dataset} ===")
    X_full, _ = load_local_dataset(dataset)
    N_full, D = X_full.shape
    ladder = make_ladder(N_full)
    print(f"  full N={N_full}, D={D}")
    print(f"  N ladder: {ladder}  (capped at STRESS_CAP={STRESS_CAP:,})")
    if MAX_MEM_GB is not None:
        print(f"  MAX_MEM_GB={MAX_MEM_GB:g} -- rungs exceeding this "
              f"estimated peak will be skipped")
    for N in ladder:
        est = _estimate_cycle_peak_gb(N)
        flag = "  <-- exceeds MAX_MEM_GB, will be SKIPPED" \
            if (MAX_MEM_GB is not None and est > MAX_MEM_GB) else ""
        print(f"    N={N:>7,}  est. peak ~{est:>7.1f} GB{flag}")

    idm = intrinsic_dim_measures(X_full)
    print(f"  intrinsic dim: PR={idm['pca_participation_ratio']:.1f}  "
          f"strain_top2={idm['strain_top2_frac']:.3f}")

    # Resume: a partial results.csv without meta.json means a prior run was
    # killed. Reuse its completed (max_iter, N) rungs so we don't recompute the
    # expensive baselines. A rung only reaches disk via checkpoint() *after*
    # all its seeds finish, so any rung present in the CSV is fully done.
    rows: List[Dict[str, Any]] = []
    done_rungs = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done_rungs = {(int(r["max_iter"]), int(r["n_samples"])) for r in rows}
        print(f"  Resuming: {len(done_rungs)} rung(s) already on disk "
              f"-> {sorted(done_rungs)}")
    run_id = len(rows)

    def add_row(phase, N, k, n_updates, max_iter, seed, time_solver, stress):
        nonlocal run_id
        rid = f"run_{run_id:06d}"; run_id += 1
        rows.append({
            "run_id": rid, "dataset": dataset,
            "n_samples": N, "n_features": D,
            "phase": phase, "k": k,
            "n_updates_per_epoch": n_updates, "max_iter": max_iter,
            "seed": seed, "time_solver": float(time_solver),
            "stress": float(stress),
        })

    def checkpoint():
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    common = dict(switch_ratio=SGD_SWITCH, epsilon=SGD_EPSILON, lr=SGD_LR)

    # Tracked so meta.json (== "this dataset is fully done") is only written
    # when every configured rung was actually attempted. A memory-skip is a
    # deliberate *incomplete* run -- writing meta.json anyway would make the
    # top-of-function "already exists, skip" check permanently hide the
    # skipped rungs, even after re-running on a bigger node with a higher
    # MAX_MEM_GB. The CSV itself is unaffected: whatever *did* complete is
    # still checkpointed and will be resumed via done_rungs next time.
    any_mem_skipped = False

    for max_iter in MAX_ITER_VALUES:
        for N in ladder:
            if (max_iter, N) in done_rungs:
                print(f"\n  --- max_iter={max_iter}  N={N}  (already done, skip) ---")
                continue
            est_gb = _estimate_cycle_peak_gb(N)
            if MAX_MEM_GB is not None and est_gb > MAX_MEM_GB:
                print(f"\n  --- max_iter={max_iter}  N={N}  SKIPPED: "
                      f"est. peak ~{est_gb:.1f} GB > MAX_MEM_GB={MAX_MEM_GB:g} "
                      f"---")
                any_mem_skipped = True
                continue
            ks = valid_k(N)
            print(f"\n  --- max_iter={max_iter}  N={N}  (k in {ks}) ---")
            for rep in range(N_REPEATS):
                seed = SEED_BASE + rep
                # Seeded row subsample: identical subset for cycle and every k
                # within this (N, seed) so overhead is a paired ratio.
                sub_rng = np.random.RandomState(seed)
                idx = sub_rng.choice(N_full, size=N, replace=False)
                X = np.ascontiguousarray(X_full[idx], dtype=np.float64)

                rc = run_sgd_benchmark(X, dissimilarity="euclidean",
                                       random_state=seed, max_iter=max_iter,
                                       **common)
                add_row("cycle", N, None, N * (N - 1) // 2, max_iter, seed,
                        rc["time"], rc["stress"])

                for k in ks:
                    rb = run_random_budget_benchmark(
                        X, n_updates_per_epoch=k * N,
                        random_state=seed, max_iter=max_iter, **common)
                    add_row("budget", N, k, k * N, max_iter, seed,
                            rb["time"], rb["stress"])
            # Persist after each completed rung so a kill loses at most this
            # rung, never the whole run.
            checkpoint()

    out_dir.mkdir(parents=True, exist_ok=True)
    if any_mem_skipped:
        print(f"\n  NOTE: one or more rungs were skipped due to MAX_MEM_GB "
              f"-- meta.json NOT written, so this dataset will be retried "
              f"(resuming completed rungs from results.csv) on the next run.")
    else:
        meta = {
            "dataset": dataset, "n_samples": N_full, "n_features": D,
            "ladder": ladder, "k_values": K_VALUES,
            "max_iter_values": MAX_ITER_VALUES, "n_repeats": N_REPEATS,
            "stress_cap": STRESS_CAP, "max_mem_gb": MAX_MEM_GB,
            **idm,
        }
        out_meta.write_text(json.dumps(meta, indent=2))

    # ---- summary: overhead(k) per N, k*(target), exponent fit ----
    df = pd.DataFrame(rows)
    mi = 30 if 30 in MAX_ITER_VALUES else MAX_ITER_VALUES[0]
    print(f"\n  === Summary: {dataset} (median over {N_REPEATS} seeds, "
          f"max_iter={mi}) ===")
    if df.empty:
        print("    (no rungs completed -- everything skipped or nothing run)")
        return
    star = {t: ([], []) for t in TARGETS}
    for N in ladder:
        rung_df = df[(df["n_samples"] == N) & (df["max_iter"] == mi)]
        if rung_df.empty:
            print(f"    N={N:6d} | (skipped -- exceeded MAX_MEM_GB)")
            continue
        cyc = rung_df[rung_df["phase"] == "cycle"]["stress"].median()
        bud = rung_df[rung_df["phase"] == "budget"]
        ks, ov = [], []
        for k, grp in bud.groupby("k"):
            ks.append(int(k)); ov.append(grp["stress"].median() / cyc)
        order = np.argsort(ks)
        ks = list(np.array(ks)[order]); ov = list(np.array(ov)[order])
        ov_str = "  ".join(f"k{kk}={oo:.3f}x" for kk, oo in zip(ks, ov))
        print(f"    N={N:6d} | {ov_str}")
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
        print("Experiment K datasets:")
        for ds in DATASETS:
            print(f"  {ds}")
        print(f"\nN ladder: {N_LADDER}  (capped at STRESS_CAP={STRESS_CAP:,})")
        print(f"k grid:   {K_VALUES}")
        print(f"max_iter: {MAX_ITER_VALUES}")
        print(f"MAX_MEM_GB: {MAX_MEM_GB}")
        print("\nEstimated peak memory per rung (cycle baseline, ~36*N^2 "
              "bytes; extrapolated above N=20,000 -- see module docstring):")
        for N in N_LADDER:
            print(f"    N={N:>7,}  ~{_estimate_cycle_peak_gb(N):>7.1f} GB")
        return
    targets = args or DATASETS
    for ds in targets:
        run_dataset(ds)


if __name__ == "__main__":
    main()
