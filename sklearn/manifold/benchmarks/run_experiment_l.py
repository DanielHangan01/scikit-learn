"""
Experiment L: standalone Pivot MDS (Brandes & Pich, 2007) on every dataset.

A *spectral* baseline for the whole benchmark collection. Unlike everything
in Experiments F-K, this is not SGD at all: select k pivots, build the N-by-k
squared-distance matrix, rectangular double-centre, one SVD -> embedding.
No epochs, no learning rate, no iteration budget.

WHY: Experiment H ran this on 6 datasets only, and found (after the scale
correction, see below) that it lands 1.4-3.4x above SGD-cycle stress on 5 of
6 -- i.e. the published small-k claim replicates. This experiment extends
that baseline to the full collection so the paper can state how a classical
spectral approximation compares with budgeted SGD across all 26 datasets and
up to N=50,000.

TWO SUITES (choose with --suite):

  j18  the 18-dataset benchmark collection, at Experiment J's top N-rung,
       reusing J's exact subsample convention (seed 42+rep -> the same 5
       subsamples J used). Pivot-MDS stress is therefore directly comparable
       to J's SGD-cycle and budgeted numbers at the same (N, seed).

  k8   the 8 large sources on the Experiment-K-optimized ladder
       {10k, 20k, 35k, 50k}, reusing that experiment's single fixed
       subsample per (dataset, N) (SUBSAMPLE_SEED=42). Directly comparable
       to the K-optimized cycle/budget results without re-running any SGD.

TWO THINGS THAT MATTER FOR CORRECTNESS

1. STRESS IS SCORED ONCE, AFTER THE SOLVE, OUTSIDE THE TIMED REGION.
   Pivot MDS is one-shot, so there are no epochs to score during -- but the
   same discipline as Experiment K-optimized applies: `time_solver` is pure
   solve time and `time_scoring` is recorded separately. Scoring uses the
   row-blocked O(N*block) scorer, so it works at N=50,000 (the old
   full-matrix path needed two N x N matrices, ~40 GB there).

2. THE EMBEDDING MUST BE RESCALED BEFORE ITS STRESS IS COMPARABLE.
   Classical-MDS-family methods fit inner products (strain), not distances,
   so they land at a stress-suboptimal global scale; SGD optimises scale
   implicitly. We apply the closed-form optimum
   alpha = <D_emb,D>/<D_emb,D_emb> before scoring. Skipping this overstated
   the spectral gap by up to ~6x in the first pass of Experiment H.
   `stress` is the rescaled (comparable) number; `stress_raw` and
   `scale_alpha` are also recorded.

NOTE ON `k`: here k = NUMBER OF PIVOTS (cost ~ O(N*k*D) + SVD). In the
budgeted-SGD experiments k = updates-per-point multiplier (cost
~ O(epochs*k*N*D)). Same letter, different quantity -- do not compare
"k=50" across the two without saying which.

MEMORY: Pivot MDS is cheap -- an N x k distance matrix, not N x N. The
whole experiment is O(N*k + N*block) and has no large-N memory wall.

Per-fit checkpoint/resume; skip-if-exists per dataset.

Usage:
    python run_experiment_l.py --suite j18
    python run_experiment_l.py --suite k8
    python run_experiment_l.py --suite k8 cifar10_raw     # subset
    python run_experiment_l.py --list
"""

from __future__ import annotations

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

from time import time

from bench_utils import (
    load_local_dataset,
    run_pivot_mds_benchmark,
    stress_and_optimal_scale_chunked_multi,
)
from run_experiment_j import (
    intrinsic_dim_measures,
    make_ladder as j_make_ladder,
    ALL_DATASETS as J_DATASETS,
)
import run_experiment_k_optimized as KO

RESULTS_ROOT = Path(__file__).parent / "results"

# Pivot counts. Matches Experiment H's grid so the 6 overlapping datasets
# can be checked against it directly.
K_VALUES = [10, 25, 50, 100, 200]

# Both selection strategies from H: raw farthest-point, and farthest-point
# in a PCA-10 space (H found PCA-10 the better selector for SGD-pivot; for
# the spectral solver we let the data decide).
STRATEGIES = [("maxmin", None), ("maxmin_pca", 10)]

SEED_BASE = 42
SCORING_BLOCK = 2048

SUITES = {
    # suite -> (datasets, n_seeds, results dirname)
    #
    # 5 seeds in both suites, matching the SGD arms they are compared
    # against (Experiment J and Experiment K-optimized both use 5 seeds for
    # the budgeted runs), so the two methods carry the same replication.
    # Note the cycle BASELINE in K-optimized is a single run, so Suite B
    # ratios still share a single-sample denominator for both methods.
    "j18": (J_DATASETS, 5, "experiment_l_j18"),
    "k8": (KO.DATASETS, 5, "experiment_l_k8"),
}


def valid_k(N: int) -> List[int]:
    """Pivot count must satisfy n_components <= k < N."""
    return [k for k in K_VALUES if 2 <= k < N]


def _reference_cycle_stress(suite: str, dataset: str, N: int,
                            seed: Optional[int]) -> Optional[float]:
    """SGD-cycle stress for the matching (dataset, N[, seed]), if on disk.

    Lets the console summary report Pivot-MDS stress as a ratio to
    full-cycle SGD without re-running any SGD. Returns None when the
    reference experiment has not been run for this cell.
    """
    if suite == "j18":
        path = RESULTS_ROOT / "experiment_j" / dataset / "results.csv"
    else:
        path = RESULTS_ROOT / "experiment_k_optimized" / dataset / "results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[(df["phase"] == "cycle") & (df["n_samples"] == N)
            & (df["max_iter"] == 30)]
    if suite == "j18" and seed is not None and "seed" in df.columns:
        seeded = df[df["seed"] == seed]
        if len(seeded):
            df = seeded
    return float(df["stress"].median()) if len(df) else None


def run_dataset(dataset: str, suite: str) -> None:
    datasets, n_seeds, dirname = SUITES[suite]
    out_dir = RESULTS_ROOT / dirname / dataset
    out_csv = out_dir / "results.csv"
    out_meta = out_dir / "meta.json"
    if out_csv.exists() and out_meta.exists():
        print(f"Skipping {dataset} (results already exist).")
        return

    print(f"\n=== Experiment L [{suite}]: {dataset} ===")
    X_full, _ = load_local_dataset(dataset)
    X_full = np.ascontiguousarray(X_full, dtype=np.float64)
    N_full, D = X_full.shape

    if suite == "j18":
        # Experiment J's top rung for this dataset (NOT necessarily full N:
        # J only appends full N when the top ladder rung is < 0.95*N_full).
        rungs = [j_make_ladder(N_full)[-1]]
    else:
        rungs = KO.make_ladder(N_full)

    idm = intrinsic_dim_measures(X_full)
    print(f"  full N={N_full}, D={D}, rungs={rungs}")
    print(f"  strain_top2={idm['strain_top2_frac']:.3f}  "
          f"pca_PR={idm['pca_participation_ratio']:.1f}")

    rows: List[Dict[str, Any]] = []
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        for r in rows:
            done.add((int(r["n_samples"]), r["pivot_strategy"],
                      int(r["n_pivots"]), int(r["seed"])))
        print(f"  Resuming: {len(done)} completed fit(s) on disk.")
    run_id = len(rows)

    def checkpoint():
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    for N in rungs:
        ks = valid_k(N)
        print(f"\n  --- N={N:,}  (pivot counts {ks}) ---")

        # Scoring groups share one X, so their input-distance blocks can be
        # computed once and reused across every embedding in the group (see
        # stress_and_optimal_scale_chunked_multi). j18 draws a fresh
        # subsample per seed (matching J), so each seed is its own group;
        # k8 uses one fixed subsample per (dataset, N) (matching
        # K-optimized), so all seeds score together.
        if suite == "k8" or N == N_full:
            sub_seed = None if N == N_full else KO.SUBSAMPLE_SEED
            groups = [(sub_seed, [SEED_BASE + r for r in range(n_seeds)])]
        else:
            groups = [(SEED_BASE + r, [SEED_BASE + r]) for r in range(n_seeds)]

        for sub_seed, group_seeds in groups:
            if N < N_full:
                idx = np.random.RandomState(sub_seed).choice(
                    N_full, size=N, replace=False)
                X = np.ascontiguousarray(X_full[idx], dtype=np.float64)
            else:
                X = X_full

            # --- solve every configuration in this group (cheap: ~0.1-0.7s
            # each even at N=50,000, since Pivot MDS is a single SVD on an
            # N x k matrix) ---
            pending, embeddings = [], []
            for seed in group_seeds:
                for strategy, pca_dim in STRATEGIES:
                    tag = strategy if pca_dim is None else f"{strategy}{pca_dim}"
                    for k in ks:
                        if (N, tag, k, seed) in done:
                            continue
                        r = run_pivot_mds_benchmark(
                            X, n_pivots=k, random_state=seed,
                            pivot_strategy=strategy,
                            pivot_pca_dim=(pca_dim or 30),
                            scoring_block_size=SCORING_BLOCK,
                            score=False,          # scored in the batch below
                        )
                        pending.append((tag, k, seed, r["time"],
                                        r["time_pivot_selection"],
                                        r["time_spectral"]))
                        embeddings.append(r["embedding"])
            if not pending:
                continue

            # --- score the whole group in ONE blocked pass ---
            t0 = time()
            scored = stress_and_optimal_scale_chunked_multi(
                X, embeddings, block_size=SCORING_BLOCK)
            batch_scoring_time = time() - t0
            per_fit_scoring = batch_scoring_time / len(pending)
            print(f"    scored {len(pending)} embeddings in "
                  f"{batch_scoring_time:.1f}s "
                  f"({per_fit_scoring:.2f}s/fit amortised)")

            for (tag, k, seed, t_solve, t_sel, t_spec), (
                    stress, stress_raw, alpha) in zip(pending, scored):
                rows.append({
                    "run_id": f"run_{run_id:06d}",
                    "dataset": dataset, "suite": suite,
                    "n_samples": N, "n_features": D,
                    "algo": "pivot_mds",
                    "pivot_strategy": tag, "n_pivots": k,
                    "seed": seed,
                    # time_solver = selection + spectral solve (the full cost
                    # of producing an embedding). Selection dominates and is
                    # single-threaded; see bench_utils.run_pivot_mds_benchmark.
                    "time_solver": float(t_solve),
                    "time_pivot_selection": float(t_sel),
                    "time_spectral": float(t_spec),
                    "time_scoring": float(per_fit_scoring),
                    "stress": float(stress),
                    "stress_raw": float(stress_raw),
                    "scale_alpha": float(alpha),
                })
                run_id += 1
            checkpoint()

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": dataset, "suite": suite,
        "n_samples": N_full, "n_features": D,
        "rungs": rungs, "k_values": K_VALUES,
        "strategies": [s for s, _ in STRATEGIES],
        "n_seeds": n_seeds, "seed_base": SEED_BASE,
        "algo": "pivot_mds (Brandes & Pich 2007, spectral)",
        "notes": "stress is scale-corrected (comparable to SGD); "
                 "stress_raw is unscaled; scoring outside timed region",
        **idm,
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    # ---- summary: Pivot-MDS stress vs SGD-cycle, per rung ----
    df = pd.DataFrame(rows)
    print(f"\n  === Summary: {dataset} "
          f"(median over {n_seeds} seeds; ratio vs SGD-cycle) ===")
    for N in rungs:
        ref = _reference_cycle_stress(suite, dataset, N, SEED_BASE)
        sub = df[df["n_samples"] == N]
        if sub.empty:
            continue
        for tag in sorted(sub["pivot_strategy"].unique()):
            cells = []
            for k, g in sub[sub["pivot_strategy"] == tag].groupby("n_pivots"):
                s = g["stress"].median()
                cells.append(f"k{int(k)}=" +
                             (f"{s / ref:.2f}x" if ref else f"{s:.3e}"))
            label = "vs cycle" if ref else "abs stress (no cycle ref on disk)"
            print(f"    N={N:>6,} {tag:<12} [{label}] " + "  ".join(cells))

    print(f"\nSaved: {out_csv}")


def main():
    args = sys.argv[1:]
    suite = "j18"
    if "--suite" in args:
        i = args.index("--suite")
        suite = args[i + 1]
        del args[i:i + 2]
    if args and args[0] == "--list":
        for name, (ds, seeds, dirname) in SUITES.items():
            print(f"\nsuite '{name}' -> results/{dirname}/  "
                  f"({len(ds)} datasets, {seeds} seeds)")
            print("  " + ", ".join(ds))
        print(f"\npivot counts: {K_VALUES}")
        print(f"strategies:   {[s for s, _ in STRATEGIES]}")
        return
    if suite not in SUITES:
        raise SystemExit(f"unknown suite {suite!r}; choose from {list(SUITES)}")
    targets = args or SUITES[suite][0]
    for ds in targets:
        run_dataset(ds, suite)


if __name__ == "__main__":
    main()
