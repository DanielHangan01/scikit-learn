"""
Experiment M: standalone SQuaD-MDS (Lambert et al., 2022) on every dataset.

A *stochastic quartet* baseline for the whole benchmark collection. Like
Experiment L (Pivot MDS), this is not our SGD: each iteration partitions the
points into N/4 random disjoint quartets and follows the gradient of a
per-quartet cost over sum-normalised distances. O(N*D) per iteration, no
distance matrix.

WHY: the paper's abstract and section 1 claim that the stochastic MDS methods
that scale "optimize a relaxation of stress rather than stress itself",
naming SQuaD-MDS. That claim is currently supported only by reading the
method's definition. This measures it.

WHAT IS ACTUALLY OPTIMISED (verified, see sklearn/manifold/_squad_mds.py and
tests/test_squad_mds.py): per quartet, both the high-dimensional and the
embedding distances are divided by their own sum before the squared error is
taken. The cost is therefore scale-free -- the relaxation claim is correct
about the objective. Whether that costs anything *in stress* is the question.

TWO SUITES (choose with --suite), exactly as Experiment L:

  j18  the 18-dataset benchmark collection, each at its FULL N (which is
       also Experiment J's top rung), reusing J's subsample convention
       (seed 42+rep). Comparable to J's SGD-cycle and budgeted numbers at
       the same (N, seed) without re-running any SGD. At full N there is no
       subsample to draw, so all 5 seeds share one X and score together.

  k8   the 8 large sources on the Experiment-K-optimized ladder
       {10k, 20k, 35k, 50k}, reusing that experiment's single fixed
       subsample per (dataset, N) (SUBSAMPLE_SEED=42).

THREE THINGS THAT MATTER FOR CORRECTNESS

1. THE EMBEDDING HAS NO MEANINGFUL GLOBAL SCALE, and must be rescaled before
   its stress means anything. We apply the closed-form optimum
   alpha = <D_emb,D>/<D_emb,D_emb>. `stress` is the rescaled (comparable)
   number; `stress_raw` and `scale_alpha` are also recorded. This is the same
   correction Experiment L applies to Pivot MDS, but it is a WEAKER repair
   here: SQuaD-MDS normalises per quartet, not globally, so one global alpha
   is not guaranteed to undo it. Watch alpha's spread -- see section 3 of
   EXPERIMENT_M_SPEC.md.

2. STRESS IS SCORED AFTER THE SOLVE, OUTSIDE THE TIMED REGION, in one
   row-blocked O(N*block) pass batched over every embedding sharing an X.
   In-fit O(N^2) scoring consumed up to 97% of a budgeted run's wall time in
   plain Experiment K.

3. THE GRID IS SWEPT IN SQuaD-MDS'S FAVOUR. The authors' default
   (exaggerate_d=True, lr=550, n_iter=1000) is tuned for RNX, and our
   validation found it is NOT the stress-optimal setting -- exaggeration
   costs stress while helping RNX. Reporting only the authors' default would
   understate the baseline, so both settings are swept and the results doc
   reports the best, as Experiment L did by giving Pivot MDS its best k and
   best selection strategy.

NOTE ON THE COMPARISON TO SGD: SQuaD-MDS gets a PCA initialisation (the
authors' choice and part of their published method), while our SGD arms use
their own init. `init=random` is swept on j18 to quantify what that is
worth; it is not a confound to be silently absorbed.

Per-fit checkpoint/resume; skip-if-exists per dataset.

Usage:
    python run_experiment_m.py --suite j18
    python run_experiment_m.py --suite k8
    python run_experiment_m.py --suite k8 cifar10_raw        # subset
    python run_experiment_m.py --suite k8 --n-iter 100,250   # override grid
    python run_experiment_m.py --list
"""

from __future__ import annotations

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple

from bench_utils import (
    load_local_dataset,
    run_squad_mds_benchmark,
    stress_and_optimal_scale_chunked_multi,
)
from run_experiment_j import (
    intrinsic_dim_measures,
    make_ladder as j_make_ladder,
    ALL_DATASETS as J_DATASETS,
)
import run_experiment_k_optimized as KO

RESULTS_ROOT = Path(__file__).parent / "results"

SEED_BASE = 42
SCORING_BLOCK = 2048

# ---------------------------------------------------------------------------
# The grid.
#
# n_iter is the method's one real cost/quality knob (cost is exactly linear in
# it), so the ladder doubles as a Pareto curve against budgeted SGD rather
# than just a search for the best value.
#
# lr: the reference reports 50-1500 as reasonable when the initial embedding
# has std 10, which _squad_mds.py reproduces; 550 is its default.
#
# exaggerate_d: squared HD distances for the first 60% of iterations. The
# reference's main.py turns this ON; validation found it helps RNX and hurts
# stress, so it is swept, never assumed.
#
# j18 is cheap (N <= 5,750) so it carries the full cross, including the
# init control. k8 is ~100x more expensive per fit, so it runs the lean grid
# at the reference learning rate; use --lr / --n-iter / --exaggerate to widen
# it once j18 has identified the winner.
# ---------------------------------------------------------------------------

# max_step_frac: the trust region that guards SQuaD-MDS's O(lr/S) gradient
# blow-up on duplicate points (see _squad_mds.py). NOT a free choice -- it
# changes results on the 7 collection datasets carrying duplicates (orl is
# 439x cycle unguarded, 27x guarded), so it is swept like any other
# hyperparameter rather than picked by hand, and `clip_frac` in the output
# records when it was active. `None` is the reference's exact behaviour.

GRID_J18 = {
    "n_iter": [100, 250, 500, 1000, 2000],
    "lr": [150.0, 550.0, 1500.0],
    "exaggerate_d": [False, True],
    "init": ["pca", "random"],
    "max_step_frac": [None, 4.0, 1.0],
}

GRID_K8 = {
    "n_iter": [100, 250, 500, 1000],
    "lr": [550.0],
    "exaggerate_d": [False, True],
    "init": ["pca"],
    # Only patchcamelyon (4.1%) carries meaningful duplicates among the 8
    # large sources, so one cap suffices here; clip_frac proves inertness.
    "max_step_frac": [4.0],
}

SUITES = {
    # suite -> (datasets, n_seeds, results dirname, grid)
    #
    # 5 seeds in both suites, matching the SGD arms they are compared against
    # (Experiment J and Experiment K-optimized both use 5 seeds for the
    # budgeted runs) and matching Experiment L, so all three baselines carry
    # the same replication. The cycle BASELINE in K-optimized is a single
    # run, so Suite B ratios still share a single-sample denominator.
    "j18": (J_DATASETS, 5, "experiment_m_j18", GRID_J18),
    "k8": (KO.DATASETS, 5, "experiment_m_k8", GRID_K8),
}


def configs(grid: Dict[str, list]) -> List[Dict[str, Any]]:
    """Full cross of the grid, as a flat list of kwargs dicts."""
    out: List[Dict[str, Any]] = [{}]
    for key in ("n_iter", "lr", "exaggerate_d", "init", "max_step_frac"):
        out = [dict(c, **{key: v}) for c in out for v in grid[key]]
    return out


def _cap(v) -> float:
    """max_step_frac as a float; None (guard off) stores as inf."""
    return float("inf") if v is None else float(v)


def config_key(cfg: Dict[str, Any]) -> Tuple:
    return (int(cfg["n_iter"]), float(cfg["lr"]),
            bool(cfg["exaggerate_d"]), str(cfg["init"]),
            _cap(cfg["max_step_frac"]))


def _reference_cycle_stress(suite: str, dataset: str, N: int,
                            seed: Optional[int]) -> Optional[float]:
    """SGD-cycle stress for the matching (dataset, N[, seed]), if on disk.

    Lets the console summary report SQuaD-MDS stress as a ratio to full-cycle
    SGD without re-running any SGD. Returns None when the reference
    experiment has not been run for this cell. Identical to Experiment L's
    helper, so the two baselines quote the same denominator.
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


def run_dataset(dataset: str, suite: str, grid: Dict[str, list]) -> None:
    datasets, n_seeds, dirname, _ = SUITES[suite]
    out_dir = RESULTS_ROOT / dirname / dataset
    out_csv = out_dir / "results.csv"
    out_meta = out_dir / "meta.json"
    if out_csv.exists() and out_meta.exists():
        print(f"Skipping {dataset} (results already exist).")
        return

    print(f"\n=== Experiment M [{suite}]: {dataset} ===")
    X_full, _ = load_local_dataset(dataset)
    X_full = np.ascontiguousarray(X_full, dtype=np.float64)
    N_full, D = X_full.shape

    if suite == "j18":
        # The dataset's FULL N -- which is also Experiment J's top rung, since
        # j_make_ladder now always ends there. Stated directly rather than
        # derived, so the two cannot drift apart silently.
        rungs = [N_full]
        assert j_make_ladder(N_full)[-1] == N_full, (
            f"{dataset}: Experiment J's top rung "
            f"({j_make_ladder(N_full)[-1]}) is not full N ({N_full}); the two "
            f"experiments would no longer join."
        )
    else:
        rungs = KO.make_ladder(N_full)

    idm = intrinsic_dim_measures(X_full)
    cfgs = configs(grid)
    print(f"  full N={N_full}, D={D}, rungs={rungs}")
    print(f"  strain_top2={idm['strain_top2_frac']:.3f}  "
          f"pca_PR={idm['pca_participation_ratio']:.1f}")
    print(f"  {len(cfgs)} configs x {n_seeds} seeds "
          f"= {len(cfgs) * n_seeds} fits per rung")

    rows: List[Dict[str, Any]] = []
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        for r in rows:
            done.add((int(r["n_samples"]), int(r["n_iter"]), float(r["lr"]),
                      bool(r["exaggerate_d"]), str(r["init"]),
                      float(r["max_step_frac"]), int(r["seed"])))
        print(f"  Resuming: {len(done)} completed fit(s) on disk.")
    run_id = len(rows)

    def checkpoint():
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    for N in rungs:
        print(f"\n  --- N={N:,} ---")

        # Scoring groups share one X, so their input-distance blocks can be
        # computed once and reused across every embedding in the group (see
        # stress_and_optimal_scale_chunked_multi) -- the difference between
        # ~1 minute and ~26 minutes for 30 embeddings at N=50k, D=3072.
        # j18 draws a fresh subsample per seed (matching J), so each seed is
        # its own group; k8 uses one fixed subsample per (dataset, N)
        # (matching K-optimized), so all seeds score together.
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

            pending, embeddings = [], []
            for seed in group_seeds:
                for cfg in cfgs:
                    key = (N, *config_key(cfg), seed)
                    if key in done:
                        continue
                    r = run_squad_mds_benchmark(
                        X, random_state=seed,
                        scoring_block_size=SCORING_BLOCK,
                        score=False,          # scored in the batch below
                        **cfg,
                    )
                    pending.append((cfg, seed, r["time"], r["clip_frac"]))
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

            for (cfg, seed, t_solve, clip_frac), (
                    stress, stress_raw, alpha) in zip(pending, scored):
                rows.append({
                    "run_id": f"run_{run_id:06d}",
                    "dataset": dataset, "suite": suite,
                    "n_samples": N, "n_features": D,
                    "algo": "squad_mds",
                    "n_iter": int(cfg["n_iter"]),
                    "lr": float(cfg["lr"]),
                    "exaggerate_d": bool(cfg["exaggerate_d"]),
                    "init": str(cfg["init"]),
                    "max_step_frac": _cap(cfg["max_step_frac"]),
                    # fraction of point-steps the trust region shortened;
                    # 0.0 == the guard never fired, i.e. this fit is exactly
                    # the reference algorithm
                    "clip_frac": float(clip_frac),
                    "seed": seed,
                    # time_solver = PCA init + all quartet iterations, i.e.
                    # the full cost of producing an embedding, scoring
                    # excluded. Matches Experiment L's convention.
                    "time_solver": float(t_solve),
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
        "rungs": rungs, "grid": grid,
        "n_seeds": n_seeds, "seed_base": SEED_BASE,
        "algo": "squad_mds (Lambert et al. 2022, stochastic quartet)",
        "notes": "stress is scale-corrected (comparable to SGD); stress_raw "
                 "is unscaled; scoring outside timed region. The optimised "
                 "cost is per-quartet RELATIVE distance error, hence "
                 "scale-free -- see _squad_mds.py.",
        **idm,
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    # ---- summary: SQuaD-MDS stress vs SGD-cycle, per rung ----
    df = pd.DataFrame(rows)
    print(f"\n  === Summary: {dataset} "
          f"(median over {n_seeds} seeds; ratio vs SGD-cycle) ===")
    for N in rungs:
        ref = _reference_cycle_stress(suite, dataset, N, SEED_BASE)
        sub = df[df["n_samples"] == N]
        if sub.empty:
            continue
        for (exa, init_, cap), g0 in sub.groupby(
                ["exaggerate_d", "init", "max_step_frac"]):
            cells = []
            for n_iter, g in g0.groupby("n_iter"):
                # best lr at this n_iter, median over seeds
                s = g.groupby("lr")["stress"].median().min()
                cells.append(f"i{int(n_iter)}=" +
                             (f"{s / ref:.2f}x" if ref else f"{s:.3e}"))
            label = "vs cycle" if ref else "abs stress (no cycle ref on disk)"
            tag = f"exa={exa!s:<5} init={init_:<6} cap={cap:<4g}"
            print(f"    N={N:>6,} {tag} [{label}] " + "  ".join(cells))
        keys = ["n_iter", "lr", "exaggerate_d", "init", "max_step_frac"]
        med = sub.groupby(keys)["stress"].median()
        best, bstress = med.idxmin(), med.min()
        print(f"      best: n_iter={best[0]} lr={best[1]:g} "
              f"exaggerate_d={best[2]} init={best[3]} cap={best[4]:g} -> "
              + (f"{bstress / ref:.3f}x cycle" if ref else f"{bstress:.3e}"))
        print(f"      scale_alpha across all configs: "
              f"[{sub['scale_alpha'].min():.3f}, "
              f"{sub['scale_alpha'].max():.3f}]")
        unguarded = sub[sub["max_step_frac"] == float("inf")]
        print(f"      trust region fired on "
              f"{100 * (sub['clip_frac'] > 0).mean():.0f}% of guarded fits; "
              f"max clip_frac {sub['clip_frac'].max():.2e}"
              + ("" if unguarded.empty else
                 f"   (unguarded best: "
                 + (f"{med.xs(float('inf'), level='max_step_frac').min() / ref:.2f}x)"
                    if ref else "n/a)")))

    print(f"\nSaved: {out_csv}")


def _parse_list(raw: str, cast):
    return [cast(v) for v in raw.split(",") if v != ""]


def main():
    args = sys.argv[1:]
    suite = "j18"
    overrides: Dict[str, list] = {}

    def take(flag):
        if flag in args:
            i = args.index(flag)
            val = args[i + 1]
            del args[i:i + 2]
            return val
        return None

    if (v := take("--suite")) is not None:
        suite = v
    if (v := take("--n-iter")) is not None:
        overrides["n_iter"] = _parse_list(v, int)
    if (v := take("--lr")) is not None:
        overrides["lr"] = _parse_list(v, float)
    if (v := take("--exaggerate")) is not None:
        overrides["exaggerate_d"] = _parse_list(
            v, lambda s: s.strip().lower() in ("1", "true", "yes"))
    if (v := take("--init")) is not None:
        overrides["init"] = _parse_list(v, str)
    if (v := take("--max-step")) is not None:
        overrides["max_step_frac"] = _parse_list(
            v, lambda s: None if s.strip().lower() in ("none", "inf") else float(s))

    if args and args[0] == "--list":
        for name, (ds, seeds, dirname, grid) in SUITES.items():
            n_cfg = len(configs(grid))
            print(f"\nsuite '{name}' -> results/{dirname}/  "
                  f"({len(ds)} datasets, {seeds} seeds, {n_cfg} configs "
                  f"= {n_cfg * seeds} fits/rung)")
            print("  " + ", ".join(ds))
            for k, vals in grid.items():
                print(f"    {k:<13} {vals}")
        return
    if suite not in SUITES:
        raise SystemExit(f"unknown suite {suite!r}; choose from {list(SUITES)}")

    grid = dict(SUITES[suite][3], **overrides)
    if overrides:
        print(f"Grid overrides applied: {overrides}")
    targets = args or SUITES[suite][0]
    for ds in targets:
        run_dataset(ds, suite, grid)


if __name__ == "__main__":
    main()
