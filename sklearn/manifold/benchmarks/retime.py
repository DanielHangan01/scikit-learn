"""
Re-time fits of Experiments O, P or Q without rerunning anything else.

Contention from other jobs can only make a fit slower; it cannot change what
the fit computes. So when the canary shows that fits ran on a busy node, those
fits are rerun here (ideally on an exclusive node), and only their time is
replaced. Each re-timed fit is checked to compute exactly what the original
did: its last stress snapshot (or SMACOF's last recorded stress) must match the
stored one. A fit that doesn't match is reported and gets no new time.

Results go to retimed.csv next to results.csv; nothing in results.csv changes.
The analysis uses the re-timed value wherever one exists.

Which fits:
    --slow F     every fit whose canary (D=784) was more than F times the
                 median canary of the whole experiment
    --config RE  every fit whose config id matches the regular expression
Both may be given; a fit is re-timed if either selects it.

Usage:
    python retime.py experiment_p cifar10_raw --slow 1.15
    python retime.py experiment_o coil20 --config '^blocked' --dry-run
"""

from __future__ import annotations

import json
import re
import socket
import sys

import numpy as np
import pandas as pd

import paper_harness as H
import run_experiment_o
import run_experiment_p
import run_experiment_q

RUNNERS = {m.EXPERIMENT: m for m in (run_experiment_o, run_experiment_p,
                                     run_experiment_q)}
RTOL = 1e-9


def _canary_baseline(exp_dir):
    files = list(exp_dir.glob("*/canary.csv"))
    if not files:
        return None
    return float(pd.concat(pd.read_csv(f) for f in files)["canary_d784"].median())


def _history(out_dir):
    out = {}
    for line in open(out_dir / "histories.jsonl"):
        h = json.loads(line)
        out[(h["n_samples"], h["config"], h["seed"])] = h
    return out


def main():
    args, opts = H.parse_args(sys.argv[1:], {"--slow": float, "--config": str,
                                             "--dry-run": bool})
    if len(args) != 2 or args[0] not in RUNNERS:
        raise SystemExit(__doc__)
    exp, name = args
    runner = RUNNERS[exp]
    out_dir = H.RESULTS_ROOT / exp / name
    df = pd.read_csv(out_dir / "results.csv")

    pick = np.zeros(len(df), dtype=bool)
    if "--slow" in opts:
        base = _canary_baseline(H.RESULTS_ROOT / exp)
        pick |= (df["canary_d784"] > opts["--slow"] * base).to_numpy()
        print(f"canary baseline (median D=784): {base:.1f} ns per pair")
    if "--config" in opts:
        pick |= df["config"].str.contains(opts["--config"], regex=True).to_numpy()
    todo = df[pick]
    print(f"{len(todo)} of {len(df)} fits selected")
    if opts.get("--dry-run") or todo.empty:
        print(todo[["n_samples", "config", "seed", "time_fit", "canary_d784"]]
              .to_string(index=False))
        return

    H.check_provenance(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text())
    histories = _history(out_dir)
    out_csv = out_dir / "retimed.csv"
    canary = H.Canary(out_dir, every=0)  # measure before every re-timed fit

    for N, rows in todo.groupby("n_samples"):
        X, _ = runner.load_input(name, int(N))
        if exp == "experiment_p":
            scorer = H.PairSampleScorer(X, n_pairs=meta["n_score_pairs"], seed=0)
        else:
            scorer = H.ExactScorer(X)
        kwargs = {"seeds": [int(s) for s in rows["seed"]]}
        if runner is run_experiment_q:
            kwargs["random_auto"] = True  # so its fits can be re-timed if run
        arms = {cid: (algo, params) for cid, algo, params, _ in runner.arms_for(
            int(N), **kwargs)}
        for r in rows.itertuples():
            algo, params = arms[r.config]
            timing = canary.maybe_run()
            fit, _ = H.run_fit(algo, X, params, int(r.seed))
            new = H.score_history(fit, scorer)["stress_raw"][-1]
            old = histories[(int(N), r.config, int(r.seed))]["stress_raw"][-1]
            match = bool(np.isclose(new, old, rtol=RTOL, atol=0))
            pd.DataFrame([dict(n_samples=int(N), config=r.config, seed=int(r.seed),
                               time_fit_old=r.time_fit, time_fit=fit["time_fit"],
                               matches=match, host=socket.gethostname(),
                               **timing)]).to_csv(
                out_csv, mode="a", header=not out_csv.exists(), index=False)
            print(f"  N={N:<6} {r.config:<36} seed={r.seed}  {r.time_fit:8.2f}s -> "
                  f"{fit['time_fit']:8.2f}s  {'ok' if match else 'MISMATCH'}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
