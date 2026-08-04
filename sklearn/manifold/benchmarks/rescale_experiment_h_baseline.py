"""
One-off repair for experiment H: recompute the PivotMDS spectral-baseline
stress with the optimal uniform rescale.

Why: classical-MDS-family embeddings fit inner products (strain), not
distances, so they come out at a stress-suboptimal global scale. The
original H run compared their *unscaled* raw stress against SGD (which
optimizes scale implicitly), overstating the spectral baseline's gap by
up to ~6x on some datasets. `run_pivot_mds_benchmark` now applies the
closed-form optimal rescale alpha = <D_emb, D> / <D_emb, D_emb> before
computing stress; this script back-fills the existing H results.

The PivotMDS solve is deterministic given the seed, so re-running
reproduces the original embeddings bit-for-bit. As a guard, the script
asserts that the recomputed *raw* (unscaled) stress matches the stored
value before replacing anything. For each PivotMDS row it then:

  - moves the old raw stress into a new `stress_raw` column,
  - replaces `stress` with the scale-optimized value,
  - patches the matching entry in histories.json.

SGD rows are untouched (`stress_raw` = NaN). Wall times are kept from
the original run. Re-running this script is a no-op-safe error: the
determinism assert fires because `stress` no longer holds raw values.

Usage:
    python rescale_experiment_h_baseline.py            # all datasets
    python rescale_experiment_h_baseline.py coil20     # subset
"""

from __future__ import annotations

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

from bench_utils import load_local_dataset, run_pivot_mds_benchmark

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_h"

DATASETS = ["fashion_mnist", "coil20", "fmd", "hiva", "cifar10", "cnae9"]
K_VALUES = [10, 25, 50, 100, 200]
STRATEGIES = [("maxmin", None, "maxmin"), ("maxmin_pca", 10, "pca10")]
N_REPEATS = 5
SEED_BASE = 42


def patch_dataset(dataset: str) -> None:
    csv_path = RESULTS_DIR / dataset / "results.csv"
    hist_path = RESULTS_DIR / dataset / "histories.json"
    if not csv_path.exists():
        print(f"Skipping {dataset} (no results.csv).")
        return

    df = pd.read_csv(csv_path)
    histories = json.loads(hist_path.read_text()) if hist_path.exists() else {}

    if "stress_raw" in df.columns and df["stress_raw"].notna().any():
        print(f"Skipping {dataset} (stress_raw already present — patched).")
        return
    df["stress_raw"] = np.nan

    print(f"\n=== Patching {dataset} ===")
    X, _ = load_local_dataset(dataset)
    N = X.shape[0]
    valid_k = [k for k in K_VALUES if 2 < k < N]

    n_patched = 0
    for k in valid_k:
        for strategy, pca_dim, tag in STRATEGIES:
            for rep in range(N_REPEATS):
                seed = SEED_BASE + rep
                r = run_pivot_mds_benchmark(
                    X, n_pivots=k, random_state=seed,
                    pivot_strategy=strategy,
                    pivot_pca_dim=(pca_dim if pca_dim is not None else 30),
                )
                mask = (
                    (df["algo"] == f"PivotMDS-{tag}")
                    & (df["k"] == k)
                    & (df["rep"] == rep)
                )
                assert mask.sum() == 1, (dataset, tag, k, rep, int(mask.sum()))

                stored_raw = float(df.loc[mask, "stress"].iloc[0])
                rel_err = abs(stored_raw - r["stress_raw"]) / max(stored_raw, 1e-12)
                assert rel_err < 1e-6, (
                    f"Determinism check failed for {dataset} {tag} k={k} "
                    f"rep={rep}: stored raw={stored_raw:.6e}, recomputed "
                    f"raw={r['stress_raw']:.6e} (rel err {rel_err:.2e}). "
                    "Refusing to patch."
                )

                df.loc[mask, "stress_raw"] = r["stress_raw"]
                df.loc[mask, "stress"] = r["stress"]
                rid = df.loc[mask, "run_id"].iloc[0]
                if rid in histories and histories[rid].get("history"):
                    t = histories[rid]["history"][0][0]
                    histories[rid]["history"] = [[t, float(r["stress"])]]
                n_patched += 1
        print(f"  k={k}: patched {2 * N_REPEATS} rows "
              f"(alpha at pca10 rep0 ~{r['scale_alpha']:.2f})")

    df.to_csv(csv_path, index=False)
    hist_path.write_text(json.dumps(histories, indent=2))

    cyc = df[df["algo"] == "SGD-cycle"]["stress"].median()
    print(f"  patched {n_patched} rows total. New medians (ratio to cycle):")
    for tag in ("maxmin", "pca10"):
        for k in valid_k:
            sub = df[(df["algo"] == f"PivotMDS-{tag}") & (df["k"] == k)]
            print(f"    {tag:6s} k={k:4d}: {sub['stress'].median() / cyc:6.2f}x "
                  f"(raw was {sub['stress_raw'].median() / cyc:6.2f}x)")


if __name__ == "__main__":
    targets = sys.argv[1:] or DATASETS
    for ds in targets:
        patch_dataset(ds)
