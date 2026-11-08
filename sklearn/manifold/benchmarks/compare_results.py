"""
Compare stress in the new results tree with the experiments it reproduces.

Each check matches fits by (dataset, N, configuration, seed). Differences come
from a different machine and BLAS build, and from a different scorer, so expect
relative errors around 1e-10 for budgeted SGD, and possibly larger where BLAS
or LAPACK enter the fit (the full sweep's distance matrix, SQuaD-MDS's PCA init).

  P budget-k{k}  vs  K-optimized budget k     (raw stress)
  P full         vs  K-optimized cycle         (raw stress, seed 42)
  O squad-...    vs  Experiment M j18          (scaled stress)
  P squad-...    vs  Experiment M k8           (scaled stress)

Usage:
    python compare_results.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import paper_harness as H


def _load(root, exp):
    frames = [pd.read_csv(p) for p in sorted((root / exp).glob("*/results.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _report(label, new, ref, keys, new_col, ref_col):
    if new.empty or ref.empty:
        print(f"{label:<34} no data")
        return
    m = new.merge(ref, on=keys, suffixes=("", "_ref"))
    if m.empty:
        print(f"{label:<34} no matching fits")
        return
    err = (m[new_col] - m[ref_col]).abs() / m[ref_col].abs()
    print(f"{label:<34} {len(m):5d} fits  median {err.median():.1e}  "
          f"max {err.max():.1e}")


def _squad_keys(df):
    out = df.copy()
    out["max_step_frac"] = out["max_step_frac"].fillna(np.inf).astype(float)
    out["exaggerate_d"] = out["exaggerate_d"].astype(bool)
    out["n_iter"] = out["n_iter"].astype(int)
    return out


def main():
    print(f"new: {H.RESULTS_ROOT}\nreference: {H.REFERENCE_ROOT}\n")
    p = _load(H.RESULTS_ROOT, "experiment_p")
    o = _load(H.RESULTS_ROOT, "experiment_o")
    ko = _load(H.REFERENCE_ROOT, "experiment_k_optimized")

    if not p.empty and not ko.empty:
        bud = p[p["config"].str.match(r"budget-k\d+$")]
        _report("P budget vs K-optimized", bud,
                ko[ko["phase"] == "budget"].rename(columns={"stress": "ref"}),
                ["dataset", "n_samples", "k", "seed"], "stress_raw", "ref")
        _report("P full vs K-optimized cycle", p[p["config"] == "full"],
                ko[ko["phase"] == "cycle"].rename(columns={"stress": "ref"}),
                ["dataset", "n_samples", "seed"], "stress_raw", "ref")

    keys = ["dataset", "n_samples", "n_iter", "lr", "exaggerate_d", "init",
            "max_step_frac", "seed"]
    for label, new, exp in (("O squad vs M j18", o, "experiment_m_j18"),
                            ("P squad vs M k8", p, "experiment_m_k8")):
        ref = _load(H.REFERENCE_ROOT, exp)
        if new.empty or ref.empty:
            print(f"{label:<34} no data")
            continue
        _report(label, _squad_keys(new[new["algo"] == "squad"]),
                _squad_keys(ref).rename(columns={"stress": "ref"}),
                keys, "stress_scaled", "ref")


if __name__ == "__main__":
    main()
