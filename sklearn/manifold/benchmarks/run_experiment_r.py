"""
Experiment R: Enhanced Force-Scheme (Ros, Arleo & Paulovich, Computers &
Graphics 135, 2026) as a baseline, on the inputs, seeds and scoring of O and P.

The code is the authors', from the submodule ``external/enhanced-force-scheme``
at a pinned commit, run with their defaults. Arms:

  efs          EFS(): up to 200 iterations of every pair once, lr 1 decaying
               by 0.9, early stop on the mean absolute residual; the distance
               matrix comes from scipy's pdist, inside the timed region
  efs-sample   EFS(sample=True), the authors' "EFS_sample": sqrt(N) rounds of
               N/2 pairs per iteration, distances computed on the fly

Suites:

  small   O's 17 datasets at full N, seeds 42-51: both arms
  large   P's 8 sources on its ladder, with its fixed subsamples, seeds 42-46:
          efs-sample at every N; efs at N=10,000 only, on seeds 42-44 like P's
          full sweep (one fit at 50k would take hours)
  epochs  O's 17 datasets, seeds 42-51: full SGDMDS (one thread) at 30, 60, 120
          and 200 epochs (``full-e{n}``). A control: EFS runs 80-130 iterations
          to our 30 epochs, so this asks whether its lower stress comes from
          the extra iterations. More epochs stretch the whole learning-rate
          schedule (the hybrid switch sits at half of them)

The main arms run on one thread, like every other method. EFS is built for
threads (its kernel is numba-parallel over the disjoint pairs of a round, so
its result doesn't depend on the count), so every arm also runs on the job's 8
cores, as ``efs-t8`` and ``efs-sample-t8``; ``--threads`` changes that list
(``--threads ""`` for none). Histories are snapshotted
after every iteration and scored after the fit, exactly in small and from
P's 10^7-pair sample in large. Results go to ``experiment_r_o/`` and
``experiment_r_p/``, to be read next to O and P.

Usage:
    python run_experiment_r.py small                  # all 17 datasets
    python run_experiment_r.py epochs coil20          # one dataset
    python run_experiment_r.py large cifar10_raw      # one source
    python run_experiment_r.py small seismic --seeds 1 --quick
    python run_experiment_r.py large california_housing --ladder 2000 --seeds 1 \\
        --quick --score-pairs 1000000
    python run_experiment_r.py --list
"""

from __future__ import annotations

import sys

import paper_config as C
import paper_harness as H
import run_experiment_o as O
import run_experiment_p as P

EXPERIMENTS = {"small": "experiment_r_o", "large": "experiment_r_p",
               "epochs": "experiment_r_epochs"}
EPOCHS = [30, 60, 120, 200]
QUICK_EPOCHS = [3, 6]
SMALL_SEEDS = O.SEEDS
LARGE_SEEDS = P.SEEDS
FULL_EFS_N = {10_000: [42, 43, 44]}
THREADS = [8]
QUICK_MAX_ITER = 20


def efs_arms(kinds, seeds, threads=(), quick=False):
    """(config id, "efs", parameters, seeds) per kind, at one and more threads."""
    arms = []
    for kind in kinds:
        cid, params = H.efs_config(kind)
        if quick:
            params = dict(params, max_iter=QUICK_MAX_ITER)
        arms.append((cid, "efs", params, seeds))
        arms += [(f"{cid}-t{t}", "efs", dict(params, threads=t), seeds)
                 for t in threads]
    return arms


def epoch_arms(seeds, quick=False):
    """(config id, "sgdmds", parameters, seeds): full SGDMDS per epoch count."""
    return [(f"full-e{e}", "sgdmds", dict(sampling_strategy="cycle", max_iter=e),
             seeds) for e in (QUICK_EPOCHS if quick else EPOCHS)]


def large_arms(n_samples, seeds, threads, quick):
    """efs-sample at every N; efs only where FULL_EFS_N lists it (on the first
    seed at every N in a quick run)."""
    full_seeds = (seeds[:1] if quick else
                  [s for s in FULL_EFS_N.get(n_samples, []) if s in seeds])
    arms = efs_arms(["full"], full_seeds, threads, quick) if full_seeds else []
    return arms + efs_arms(["sample"], seeds, threads, quick)


def run_small(name, seeds, arms, suite, out_root):
    """Every arm on one of O's datasets at full N, scored exactly."""
    out_dir = out_root / EXPERIMENTS[suite] / name
    if H.is_complete(out_dir):
        print(f"Skipping {name} (complete).")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    H.check_provenance(out_dir)
    X, _ = O.load_input(name)
    N, D = X.shape
    store = H.ResultStore(out_dir)
    H.write_meta(out_dir, dict(
        experiment=f"R-{suite}", dataset=name, n_samples=N, n_features=D,
        seeds=seeds, arms={cid: p for cid, _, p, _ in arms},
        dataset_sha256=H.dataset_checksum(name), complete=False))
    print(f"\n=== Experiment R ({suite}): {name}  N={N}  D={D}  "
          f"({len(store)} fits on disk) ===")
    scorer = H.ExactScorer(X)
    canary = H.Canary(out_dir)

    base = dict(dataset=name, n_samples=N, n_features=D, subsample_seed=None)
    for cid, algo, params, seed in H.seed_major(arms):
        if store.done(N, cid, seed):
            continue
        timing = canary.maybe_run()
        fit, fields = H.run_fit(algo, X, params, seed)
        history = H.score_history(fit, scorer)
        final = scorer(fit["embedding"])
        row = H.make_row(fit, history, final, algo=algo, config=cid,
                         seed=seed, time_scoring=None, **base, **fields,
                         **timing)
        store.add(row, history,
                  fit["embedding"] if seed == O.SAVE_SEED else None)
        print(f"  {cid:<20} seed={seed}  {fit['time_fit']:8.2f}s  "
              f"{fit['n_iter_run']:4d} iterations  stress={final[0]:.6g}")

    H.write_meta(out_dir, dict(complete=True))
    print(f"Saved: {store.csv}")


def run_large(name, seeds, threads, quick, ladder_override, n_score_pairs,
              out_root):
    out_dir = out_root / EXPERIMENTS["large"] / name
    if H.is_complete(out_dir):
        print(f"Skipping {name} (complete).")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    H.check_provenance(out_dir)
    X_full, _ = P.load_input(name, float("inf"))
    N_full, D = X_full.shape
    del X_full
    ladder = ladder_override or [n for n in C.make_ladder(N_full)
                                 if n <= P.MAX_N]
    store = H.ResultStore(out_dir)
    H.write_meta(out_dir, dict(
        experiment="R-large", dataset=name, n_samples=N_full, n_features=D,
        ladder=ladder, seeds=seeds, full_efs_n=FULL_EFS_N,
        arms={cid: p for cid, _, p, _ in efs_arms(["full", "sample"], seeds,
                                                  threads, quick)},
        n_score_pairs=n_score_pairs, subsample_seed=C.SUBSAMPLE_SEED,
        dataset_sha256=H.dataset_checksum(name), complete=False))
    print(f"\n=== Experiment R (large): {name}  N_full={N_full:,}  D={D}  "
          f"ladder={ladder}  ({len(store)} fits on disk) ===")
    canary = H.Canary(out_dir)

    for N in ladder:
        X, sub_seed = P.load_input(name, N)
        P.run_rung(name, X, N, D, sub_seed,
                   large_arms(N, seeds, threads, quick), store, canary,
                   top_rung=(N == ladder[-1]), n_score_pairs=n_score_pairs)
    H.write_meta(out_dir, dict(complete=not ladder_override))
    print(f"Saved: {store.csv}")


def check_threads(threads):
    """Stop before any fit if numba can't run the requested thread counts."""
    import numba

    limit = numba.config.NUMBA_NUM_THREADS
    if threads and max(threads) > limit:
        raise SystemExit(
            f"--threads asks for {max(threads)} threads, but numba can use {limit} "
            "here (the CPUs this process may run on). Under srun, pass "
            "--cpus-per-task; or lower --threads.")


def main():
    args, opts = H.parse_args(sys.argv[1:], {
        "--seeds": int, "--quick": bool, "--list": bool, "--threads": H.int_list,
        "--ladder": H.int_list, "--score-pairs": int})
    if opts.get("--list"):
        print("small:", ", ".join(O.DATASETS), f"  seeds={SMALL_SEEDS}")
        print("large:", ", ".join(P.DATASETS), f"  seeds={LARGE_SEEDS}  "
              f"efs at {FULL_EFS_N}")
        for kind in ("full", "sample"):
            print(" ", *H.efs_config(kind))
        print("epochs:", [cid for cid, *_ in epoch_arms(SMALL_SEEDS)],
              f"  seeds={SMALL_SEEDS}")
        return
    if not args or args[0] not in EXPERIMENTS:
        raise SystemExit("usage: run_experiment_r.py {small,large,epochs} [datasets] "
                         "(see the docstring)")
    suite, names = args[0], args[1:]
    quick = opts.get("--quick", False)
    threads = opts.get("--threads", THREADS)
    print(f"Results root: {H.RESULTS_ROOT}")
    if suite == "epochs":
        seeds = SMALL_SEEDS[: opts.get("--seeds", len(SMALL_SEEDS))]
        for name in names or O.DATASETS:
            run_small(name, seeds, epoch_arms(seeds, quick), suite, H.RESULTS_ROOT)
        return
    check_threads(threads)
    H.efs_warmup()
    if suite == "small":
        seeds = SMALL_SEEDS[: opts.get("--seeds", len(SMALL_SEEDS))]
        arms = efs_arms(["full", "sample"], seeds, threads, quick)
        for name in names or O.DATASETS:
            run_small(name, seeds, arms, suite, H.RESULTS_ROOT)
    else:
        seeds = LARGE_SEEDS[: opts.get("--seeds", len(LARGE_SEEDS))]
        for name in names or P.DATASETS:
            run_large(name, seeds, threads, quick, opts.get("--ladder"),
                      opts.get("--score-pairs", P.N_SCORE_PAIRS),
                      H.RESULTS_ROOT)


if __name__ == "__main__":
    main()
