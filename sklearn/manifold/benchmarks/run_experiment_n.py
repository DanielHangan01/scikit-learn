"""
Experiment N: point-blocked sampling for budgeted SGD-MDS.

Budgeted SGD draws one random pair per update and reads two feature rows for
it (2*D values per pair). SQuaD-MDS reads four rows and gets six pairs out of
them (0.67*D per pair) -- one of the two reasons it is 7-14x faster than
k=200 (Experiment M). Point-blocking borrows that and nothing else: draw m
points, compute all m(m-1)/2 feature-space distances from one read of their
rows, then apply our ordinary raw-stress step to each pair. Same objective,
same per-pair update, same budget counted in PAIRS; only the grouping of
pairs changes. See _sgd_mds_cython.run_sgd_epoch_lazy_blocked.

QUESTIONS
  1. Speed: how much does time per pair fall with m, and where does it stop
     (register spill, cache)? The memory model predicts row reads per pair
     fall as 1/(m-1) relative to m=2.
  2. Quality: pairs inside a block are correlated (in a block of 8, every
     point is in 7 of the 28 pairs). At EQUAL pair count, what does that
     cost in stress -- and does it depend on the dataset's regime?
  3. The two switches:
       sampling  random    each block's points drawn independently
                 partition shuffle all points, cut into disjoint blocks
                           (SQuaD-MDS's scheme)
       update    sequential  each pair's step applied immediately (our rule)
                 sum         steps from the block's starting positions,
                             summed per point, applied once (SQuaD's rule)
                 mean        as sum, divided by pairs per point (m-1)
  The primary summary is the time to reach a stress target (e.g. <= 1.01x
  the legacy k=200 stress), which folds 1 and 2 together.

TWO BASELINES, AND WHY BOTH
  m2-random-sequential-legacy   the original random-pair kernel -- the exact
                                code every earlier experiment was timed with
  m2-random-sequential-blocked  the SAME algorithm (bit-identical embedding,
                                checked per fit in `matches_legacy`) run
                                through the blocked kernel
  The original kernel pays a GIL round-trip per pair update (Cython's
  exception check after every `_lazy_sgd_step` call, which is not declared
  noexcept); the blocked kernel does not. That alone is worth ~3x at low D
  with no change to the algorithm. So:
      legacy  / blocked-m2   = what the implementation fix buys
      blocked-m2 / blocked-m = what blocking buys
  Quoting any blocked arm against legacy without this split would credit
  blocking with the fix.

SAME DATA, SEEDS AND HYPERPARAMETERS AS THE REFERENCE EXPERIMENTS
  j18  every dataset at FULL N, rows permuted by RandomState(seed) exactly as
       Experiment J does even at full N, so the legacy arm reproduces J's
       budget runs; seeds 42.., one X per seed.
  k8   Experiment K-optimized's fixed subsample (SUBSAMPLE_SEED=42) at
       N=10,000 and at the largest K-optimized rung <= 50,000 (20,640 for
       california_housing); seeds 42...
  Both: max_iter=30, lr=0.5, hybrid scheduler, switch 0.5, eps 0.001.
  REPRODUCTION CHECK: every legacy fit's raw stress is compared with J's /
  K-optimized's stored value for the same (N, k, seed); rel error is stored
  in `repro_rel_err` and summarised in meta.json. (Different scorer, so
  ~1e-10, not bitwise. J's bank/hiva were run at 2,000/3,076-truncated N
  until their queued re-run, so they have nothing to compare against.)

NO IN-FIT SCORING. compute_stress=False; stress is scored after the fits,
outside the timed region, batched over embeddings sharing an X
(stress_and_optimal_scale_chunked_multi), in chunks of SCORE_CHUNK so a crash
loses at most one chunk. `stress` is the stress-optimal rescale (sum-mode
blocks could plausibly land off scale); `stress_raw` and `scale_alpha` are
kept. A diverged fit (non-finite embedding) is recorded as stress=inf.

Per-fit checkpoint/resume; skip-if-exists per dataset (delete meta.json to
resume after widening the grid).

Usage:
    python run_experiment_n.py --suite j18
    python run_experiment_n.py --suite k8
    python run_experiment_n.py --suite j18 coil20 --k 50 --seeds 1   # subset
    python run_experiment_n.py --suite k8 --m 2,4,8,16 --update sequential,sum,mean
    python run_experiment_n.py --list
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bench_utils import (
    load_local_dataset,
    run_random_budget_benchmark,
    stress_and_optimal_scale_chunked_multi,
)
from run_experiment_j import (
    intrinsic_dim_measures,
    make_ladder as j_make_ladder,
    ALL_DATASETS as J_DATASETS,
)
from run_experiment_m import _reference_cycle_stress
import run_experiment_k_optimized as KO

RESULTS_ROOT = Path(__file__).parent / "results"

SEED_BASE = 42
SCORING_BLOCK = 2048
SCORE_CHUNK = 16

# Same SGD hyperparameters as Experiments J and K-optimized.
SGD_COMMON = dict(max_iter=30, switch_ratio=0.5, epsilon=0.001, lr=0.5)

# k8 rungs: the 10k overlap anchor and the largest K-optimized rung <= this.
K8_TOP_CAP = 50_000

# ---------------------------------------------------------------------------
# Grids. j18 is cheap (N <= 5,750), so it carries the full cross and acts as
# the filter; k8 runs a lean default -- widen it with --m/--sampling/--update
# once j18 has shown which arms are worth it (the full cross on k8 is ~87 h).
#
# At m=2 the three update rules are the same algorithm (one pair per block),
# so only `sequential` is run there. The two m=2 random sequential baselines
# (legacy and blocked kernel) are always included, whatever the overrides.
# ---------------------------------------------------------------------------

GRID_J18 = {
    "m": [2, 4, 8, 16],
    "sampling": ["random", "partition"],
    "update": ["sequential", "sum", "mean"],
    "k": [25, 50, 100, 200],
}

GRID_K8 = {
    "m": [2, 4, 8],
    "sampling": ["random", "partition"],
    "update": ["sequential"],
    "k": [100, 200],
}

SUITES = {
    # suite -> (datasets, n_seeds, results dirname, grid)
    "j18": (J_DATASETS, 5, "experiment_n_j18", GRID_J18),
    "k8": (KO.DATASETS, 3, "experiment_n_k8", GRID_K8),
}

BASELINES = [
    dict(m=2, sampling="random", update="sequential", kernel="legacy"),
    dict(m=2, sampling="random", update="sequential", kernel="blocked"),
]


def arms(grid: Dict[str, list]) -> List[Dict[str, Any]]:
    out = [dict(b) for b in BASELINES]
    for m in grid["m"]:
        for sampling in grid["sampling"]:
            for update in grid["update"]:
                if m == 2 and update != "sequential":
                    continue
                arm = dict(m=m, sampling=sampling, update=update,
                           kernel="blocked")
                if arm not in out:
                    out.append(arm)
    return out


def arm_name(arm: Dict[str, Any]) -> str:
    return f"m{arm['m']}-{arm['sampling']}-{arm['update']}-{arm['kernel']}"


def valid_k(N: int, ks: List[int]) -> List[int]:
    """J's filter: a budget of k*N pairs must stay below N(N-1)/2."""
    cap = (N - 1) // 2
    return [k for k in ks if k < cap]


def k8_rungs(N_full: int) -> List[int]:
    ladder = KO.make_ladder(N_full)
    top = max(n for n in ladder if n <= K8_TOP_CAP)
    return sorted({n for n in ladder if n == 10_000} | {top})


# ---------------------------------------------------------------------------
# Reference stresses for the reproduction check
# ---------------------------------------------------------------------------

def _reference_budget_stress(suite: str, dataset: str
                             ) -> Dict[Tuple[int, int, int], float]:
    """(N, k, seed) -> stored budget-run stress from J / K-optimized."""
    if suite == "j18":
        path = RESULTS_ROOT / "experiment_j" / dataset / "results.csv"
    else:
        path = RESULTS_ROOT / "experiment_k_optimized" / dataset / "results.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df = df[(df["phase"] == "budget") & (df["max_iter"] == 30)]
    return {(int(r.n_samples), int(r.k), int(r.seed)): float(r.stress)
            for r in df.itertuples()}


def _squad_reference(dataset: str, N: int) -> Optional[Tuple[float, float]]:
    """Best SQuaD-MDS (median stress, its median time) at this N, if on disk."""
    path = RESULTS_ROOT / "experiment_m_k8" / dataset / "results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["n_samples"] == N]
    if df.empty:
        return None
    keys = ["n_iter", "lr", "exaggerate_d", "init", "max_step_frac"]
    g = df.groupby(keys).agg(stress=("stress", "median"),
                             time=("time_solver", "median"))
    best = g.loc[g["stress"].idxmin()]
    return float(best["stress"]), float(best["time"])


# ---------------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------------

def run_dataset(dataset: str, suite: str, grid: Dict[str, list],
                n_seeds: int) -> None:
    _, _, dirname, _ = SUITES[suite]
    out_dir = RESULTS_ROOT / dirname / dataset
    out_csv = out_dir / "results.csv"
    out_meta = out_dir / "meta.json"
    if out_csv.exists() and out_meta.exists():
        print(f"Skipping {dataset} (results already exist).")
        return

    print(f"\n=== Experiment N [{suite}]: {dataset} ===")
    X_full, _ = load_local_dataset(dataset)
    X_full = np.ascontiguousarray(X_full, dtype=np.float64)
    N_full, D = X_full.shape

    if suite == "j18":
        rungs = [N_full]
        assert j_make_ladder(N_full)[-1] == N_full, (
            f"{dataset}: Experiment J's top rung is not full N; the legacy "
            f"arm would no longer reproduce J."
        )
    else:
        rungs = k8_rungs(N_full)

    idm = intrinsic_dim_measures(X_full)
    arm_list = arms(grid)
    seeds = [SEED_BASE + r for r in range(n_seeds)]
    ref_budget = _reference_budget_stress(suite, dataset)
    print(f"  full N={N_full}, D={D}, rungs={rungs}")
    print(f"  strain_top2={idm['strain_top2_frac']:.3f}  "
          f"pca_PR={idm['pca_participation_ratio']:.1f}")
    print(f"  {len(arm_list)} arms x k in {grid['k']} x {n_seeds} seeds")

    rows: List[Dict[str, Any]] = []
    done = set()
    if out_csv.exists():
        rows = pd.read_csv(out_csv).to_dict("records")
        for r in rows:
            done.add((int(r["n_samples"]), int(r["k"]), int(r["m"]),
                      r["sampling"], r["update"], r["kernel"], int(r["seed"])))
        print(f"  Resuming: {len(done)} completed fit(s) on disk.")

    def checkpoint():
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    for N in rungs:
        ks = valid_k(N, grid["k"])

        # One X per seed on j18 (J's per-seed row permutation); one fixed
        # subsample shared by every seed on k8 (K-optimized's convention).
        if suite == "j18":
            groups = [(seed, [seed]) for seed in seeds]
        else:
            groups = [(KO.SUBSAMPLE_SEED, seeds)]

        todo = [(sub, seed, k, arm) for sub, group_seeds in groups
                for seed in group_seeds for k in ks for arm in arm_list
                if (N, k, arm["m"], arm["sampling"], arm["update"],
                    arm["kernel"], seed) not in done]
        print(f"\n  --- N={N:,}  k in {ks}: {len(todo)} fit(s) to run ---")
        counter = 0

        for sub_seed, group_seeds in groups:
            if suite == "j18":
                idx = np.random.RandomState(sub_seed).choice(
                    N_full, size=N, replace=False)
                X = np.ascontiguousarray(X_full[idx], dtype=np.float64)
            elif N < N_full:
                idx = np.random.RandomState(sub_seed).choice(
                    N_full, size=N, replace=False)
                X = np.ascontiguousarray(X_full[idx], dtype=np.float64)
            else:
                X = X_full

            pending: List[Dict[str, Any]] = []
            embeddings: List[np.ndarray] = []

            def score_pending():
                if not pending:
                    return
                finite = [i for i, e in enumerate(embeddings)
                          if np.all(np.isfinite(e))]
                t0 = time()
                scored = stress_and_optimal_scale_chunked_multi(
                    X, [embeddings[i] for i in finite],
                    block_size=SCORING_BLOCK) if finite else []
                per_fit = (time() - t0) / max(len(finite), 1)
                results = dict(zip(finite, scored))
                for i, row in enumerate(pending):
                    stress, stress_raw, alpha = results.get(
                        i, (np.inf, np.inf, np.nan))
                    row.update(time_scoring=per_fit if i in results else 0.0,
                               stress=float(stress),
                               stress_raw=float(stress_raw),
                               scale_alpha=float(alpha),
                               diverged=i not in results)
                    ref = row.pop("_ref")
                    row["repro_rel_err"] = (
                        abs(stress_raw - ref) / ref
                        if ref is not None and np.isfinite(stress_raw)
                        else np.nan)
                    rows.append(row)
                pending.clear()
                embeddings.clear()
                checkpoint()

            for seed in group_seeds:
                for k in ks:
                    legacy_embedding = None
                    for arm in arm_list:
                        key = (N, k, arm["m"], arm["sampling"],
                               arm["update"], arm["kernel"], seed)
                        if key in done:
                            continue
                        counter += 1
                        with contextlib.redirect_stdout(io.StringIO()):
                            r = run_random_budget_benchmark(
                                X, n_updates_per_epoch=k * N,
                                random_state=seed, compute_stress=False,
                                block_size=arm["m"],
                                block_sampling=arm["sampling"],
                                block_update=arm["update"],
                                force_blocked_kernel=(
                                    arm["kernel"] == "blocked"),
                                **SGD_COMMON)
                        emb = r["embedding"]

                        is_baseline = (arm["m"], arm["sampling"],
                                       arm["update"]) == (2, "random",
                                                          "sequential")
                        matches_legacy = np.nan
                        if is_baseline and arm["kernel"] == "legacy":
                            legacy_embedding = emb
                        elif is_baseline and legacy_embedding is not None:
                            matches_legacy = bool(
                                np.array_equal(emb, legacy_embedding))

                        P = arm["m"] * (arm["m"] - 1) // 2
                        pending.append({
                            "dataset": dataset, "suite": suite,
                            "n_samples": N, "n_features": D,
                            "arm": arm_name(arm),
                            "m": arm["m"], "sampling": arm["sampling"],
                            "update": arm["update"], "kernel": arm["kernel"],
                            "k": k, "pairs_per_epoch": k * N,
                            "blocks_per_epoch": -(-k * N // P),
                            "max_iter": SGD_COMMON["max_iter"],
                            "seed": seed,
                            "subsample_seed": (sub_seed if suite == "k8"
                                               else seed),
                            # full fit_transform wall time, scoring excluded
                            "time_solver": float(r["time"]),
                            "matches_legacy": matches_legacy,
                            "_ref": (ref_budget.get((N, k, seed))
                                     if arm["kernel"] == "legacy" else None),
                        })
                        embeddings.append(emb)
                        print(f"    [{counter}/{len(todo)}] k={k:<3} "
                              f"{arm_name(arm):<36} seed={seed}  "
                              f"{r['time']:7.2f}s")
                        if len(pending) >= SCORE_CHUNK:
                            score_pending()
            score_pending()

    df = pd.DataFrame(rows)
    repro = df[df["kernel"].eq("legacy") & df["repro_rel_err"].notna()]
    ident = df[df["matches_legacy"].notna()]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": dataset, "suite": suite,
        "n_samples": N_full, "n_features": D,
        "rungs": rungs, "grid": grid, "n_seeds": n_seeds,
        "seed_base": SEED_BASE, "sgd": SGD_COMMON,
        "arms": [arm_name(a) for a in arm_list],
        "reproduction": {
            "reference": ("experiment_j" if suite == "j18"
                          else "experiment_k_optimized"),
            "n_checked": int(len(repro)),
            "max_rel_err": (float(repro["repro_rel_err"].max())
                            if len(repro) else None),
            "blocked_m2_bitwise_equal_legacy": (
                f"{int(ident['matches_legacy'].astype(bool).sum())}/"
                f"{len(ident)}"),
        },
        "notes": "budget counted in pair updates for every m; stress is the "
                 "stress-optimal rescale, stress_raw unscaled; scoring "
                 "outside the timed region.",
        **idm,
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    summarise(df, dataset, suite)
    print(f"\nSaved: {out_csv}")


def summarise(df: pd.DataFrame, dataset: str, suite: str) -> None:
    """Per (N, k): each arm's stress and time against both baselines."""
    rep = df[df["kernel"].eq("legacy") & df["repro_rel_err"].notna()]
    if len(rep):
        print(f"\n  reproduction vs reference: {len(rep)} legacy fit(s), "
              f"max rel err {rep['repro_rel_err'].max():.1e}")
    ident = df[df["matches_legacy"].notna()]
    if len(ident):
        print(f"  blocked kernel at m=2 bit-identical to legacy: "
              f"{int(ident['matches_legacy'].astype(bool).sum())}/{len(ident)}")

    med = df.groupby(["n_samples", "k", "arm"]).agg(
        stress=("stress", "median"), time=("time_solver", "median"),
        diverged=("diverged", "sum"))
    for N in sorted(df["n_samples"].unique()):
        cyc = _reference_cycle_stress(suite, dataset, int(N), SEED_BASE)
        squad = _squad_reference(dataset, int(N)) if suite == "k8" else None
        print(f"\n  === {dataset} N={N:,}: median over seeds ===")
        if squad and cyc:
            print(f"  best SQuaD-MDS (Exp. M): {squad[0] / cyc:.3f}x cycle "
                  f"in {squad[1]:.1f}s")
        print(f"  {'k':>4} {'arm':<36} {'stress/legacy':>13} "
              f"{'x cycle':>8} {'time':>8} {'fix':>6} {'block':>6}")
        for k in sorted(df.loc[df["n_samples"] == N, "k"].unique()):
            g = med.loc[(N, k)]
            if not {"m2-random-sequential-legacy",
                    "m2-random-sequential-blocked"} <= set(g.index):
                print(f"  {k:>4} (baselines missing -- skipped)")
                continue
            leg = g.loc["m2-random-sequential-legacy"]
            blk = g.loc["m2-random-sequential-blocked"]
            for arm, r in g.iterrows():
                s_leg = r["stress"] / leg["stress"]
                s_cyc = f"{r['stress'] / cyc:.3f}" if cyc else "   -"
                flag = f"  ({int(r['diverged'])} diverged)" if r["diverged"] else ""
                print(f"  {k:>4} {arm:<36} {s_leg:>13.3f} {s_cyc:>8} "
                      f"{r['time']:>7.2f}s {leg['time'] / r['time']:>5.2f}x "
                      f"{blk['time'] / r['time']:>5.2f}x{flag}")
    print("\n  fix   = legacy time / arm time (everything the arm gains)")
    print("  block = blocked-m2 time / arm time (what blocking alone gains)")


def _parse_list(raw: str, cast):
    return [cast(v) for v in raw.split(",") if v != ""]


def main():
    args = sys.argv[1:]
    suite = "j18"
    overrides: Dict[str, list] = {}
    n_seeds = None

    def take(flag):
        if flag in args:
            i = args.index(flag)
            val = args[i + 1]
            del args[i:i + 2]
            return val
        return None

    if (v := take("--suite")) is not None:
        suite = v
    if (v := take("--m")) is not None:
        overrides["m"] = _parse_list(v, int)
    if (v := take("--sampling")) is not None:
        overrides["sampling"] = _parse_list(v, str)
    if (v := take("--update")) is not None:
        overrides["update"] = _parse_list(v, str)
    if (v := take("--k")) is not None:
        overrides["k"] = _parse_list(v, int)
    if (v := take("--seeds")) is not None:
        n_seeds = int(v)

    if args and args[0] == "--list":
        for name, (ds, seeds, dirname, grid) in SUITES.items():
            arm_list = arms(grid)
            print(f"\nsuite '{name}' -> results/{dirname}/  "
                  f"({len(ds)} datasets, {seeds} seeds, {len(arm_list)} "
                  f"arms x {len(grid['k'])} k = "
                  f"{len(arm_list) * len(grid['k']) * seeds} fits/rung)")
            print("  " + ", ".join(ds))
            for key, vals in grid.items():
                print(f"    {key:<9} {vals}")
            print("    arms:")
            for a in arm_list:
                print(f"      {arm_name(a)}")
        return
    if suite not in SUITES:
        raise SystemExit(f"unknown suite {suite!r}; choose from {list(SUITES)}")
    bad = set(overrides.get("sampling", [])) - {"random", "partition"}
    bad |= set(overrides.get("update", [])) - {"sequential", "sum", "mean"}
    if bad or any(m < 2 for m in overrides.get("m", [])):
        raise SystemExit(f"invalid override value(s): {sorted(map(str, bad))}")

    grid = dict(SUITES[suite][3], **overrides)
    n_seeds = n_seeds or SUITES[suite][1]
    if overrides or n_seeds != SUITES[suite][1]:
        print(f"Grid overrides applied: {overrides}  seeds={n_seeds}")
    targets = args or SUITES[suite][0]
    for ds in targets:
        run_dataset(ds, suite, grid, n_seeds)


if __name__ == "__main__":
    main()
