#!/usr/bin/env python3
"""
Auto-select epsilon and a robust "golden" SGD-MDS configuration from Experiment C
when results are stored in per-dataset folders:

  <RESULTS_DIR>/fashion_mnist/results.csv
  <RESULTS_DIR>/imdb/results.csv
  <RESULTS_DIR>/s_curve/results.csv

Default RESULTS_DIR matches your repo layout:
  sklearn/manifold/benchmarks/results/experiment_c

Outputs (created inside RESULTS_DIR/_selection):
  - combined_results.csv
  - agg_per_config.csv
  - epsilon_regret_table.csv
  - epsilon_summary.csv
  - epsilon_selection.json
  - golden_config_table.csv
  - golden_summary.csv
  - golden_selection.json

Selection logic:
  1) Aggregate repeats per config using MEDIAN (default; robust).
  2) For each (dataset, max_iter), compute S* = best achievable median stress over ALL eps/configs.
  3) For each epsilon, compute regret of the best config under that epsilon vs S* for each (dataset, max_iter).
  4) Pick epsilon* by MINIMAX regret across (dataset, max_iter) (default).
  5) Within epsilon*, pick a single configuration theta by MINIMAX regret across (dataset, max_iter).

Notes:
- Works with your columns: dataset,max_iter,scheduler_label,lr,epsilon,switch_ratio,stress,time,...
- If you have config_id, it will use it; otherwise it will construct a config key from hyperparams.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------
DATASETS = ["fashion_mnist", "imdb", "s_curve"]

# Adjust this if your repo layout differs
DEFAULT_RESULTS_DIR = Path("results/experiment_c")

# Use robust aggregation by default
AGG_STAT = "median"      # "median" or "mean"
EPSILON_MODE = "minimax" # "minimax" or "medianrobust"
INCLUDE_SWITCH_RATIO_IN_GOLDEN = True
TARGET_T = 30


# -----------------------------
# Helpers
# -----------------------------
def _as_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _ensure_dataset_col(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if "dataset" not in df.columns:
        df = df.copy()
        df["dataset"] = dataset_name
    return df


def load_all_results(results_dir: Path, datasets: list[str]) -> pd.DataFrame:
    frames = []
    for ds in datasets:
        p = results_dir / ds / "results.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing results.csv for dataset '{ds}': {p}")
        df = pd.read_csv(p)
        df = _ensure_dataset_col(df, ds)

        # Basic type cleanup
        if "lr" in df.columns:
            df["lr"] = df["lr"].apply(_as_float)
        if "epsilon" in df.columns:
            df["epsilon"] = df["epsilon"].apply(_as_float)
        if "switch_ratio" in df.columns:
            df["switch_ratio"] = df["switch_ratio"].apply(_as_float)
        if "max_iter" in df.columns:
            df["max_iter"] = df["max_iter"].astype(int)

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out


def add_config_id_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a stable config_id column exists.
    If your CSV already has config_id, we keep it.
    Otherwise, derive it from hyperparams that define a configuration.
    """
    if "config_id" in df.columns:
        return df

    required = ["dataset", "max_iter", "epsilon", "scheduler_label", "lr", "switch_ratio"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot derive config_id; missing columns: {missing}")

    df = df.copy()
    # Switch ratio only meaningful for hybrid; for others it may be constant 0.4 in your code.
    # Still include it for uniqueness.
    key = (
        df["dataset"].astype(str)
        + "|T=" + df["max_iter"].astype(str)
        + "|eps=" + df["epsilon"].astype(str)
        + "|sched=" + df["scheduler_label"].astype(str)
        + "|lr=" + df["lr"].astype(str)
        + "|r=" + df["switch_ratio"].astype(str)
    )
    df["config_id"] = pd.util.hash_pandas_object(key, index=False).astype(str)
    return df


def aggregate_per_config(df: pd.DataFrame, stat: str = "median") -> pd.DataFrame:
    """
    Aggregate repeated runs (seeds/inits) into one row per configuration.
    """
    df = add_config_id_if_missing(df)

    group_cols = [
        "config_id",
        "dataset",
        "max_iter",
        "epsilon",
        "scheduler_label",
        "scheduler",
        "lr",
        "switch_ratio",
    ]
    # Some columns might not exist in older CSVs (e.g., 'scheduler')
    group_cols = [c for c in group_cols if c in df.columns]

    if stat == "median":
        stress_agg = ("stress", "median")
    elif stat == "mean":
        stress_agg = ("stress", "mean")
    else:
        raise ValueError("stat must be 'median' or 'mean'")

    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            stress_agg=stress_agg,
            stress_std=("stress", "std"),
            time_mean=("time", "mean") if "time" in df.columns else ("stress", "size"),
            n_runs=("stress", "size"),
        )
        .reset_index()
    )
    return agg


def compute_S_star(agg: pd.DataFrame) -> pd.DataFrame:
    """
    S_star(dataset, max_iter) = best achievable stress_agg over ALL eps/configs.
    """
    return (
        agg.groupby(["dataset", "max_iter"], dropna=False)["stress_agg"]
        .min()
        .reset_index()
        .rename(columns={"stress_agg": "S_star"})
    )


def epsilon_regret_table(agg: pd.DataFrame, s_star: pd.DataFrame) -> pd.DataFrame:
    """
    For each epsilon, for each (dataset, max_iter), compute:
      S_best_eps = best stress_agg under that epsilon
      regret_pct = (S_best_eps - S_star) / S_star * 100
    """
    best_eps = (
        agg.groupby(["dataset", "max_iter", "epsilon"], dropna=False)["stress_agg"]
        .min()
        .reset_index()
        .rename(columns={"stress_agg": "S_best_eps"})
    )
    out = best_eps.merge(s_star, on=["dataset", "max_iter"], how="left")
    out["regret_pct"] = 100.0 * (out["S_best_eps"] - out["S_star"]) / out["S_star"]
    return out


def select_epsilon(reg_tbl: pd.DataFrame, mode: str = "minimax") -> tuple[float, pd.DataFrame]:
    """
    mode:
      - minimax: choose epsilon minimizing max regret across (dataset, max_iter)
      - medianrobust: choose epsilon minimizing median regret across (dataset, max_iter)
    Returns: (epsilon_star, summary_df)
    """
    summary = (
        reg_tbl.groupby("epsilon", dropna=False)["regret_pct"]
        .agg(regret_max="max", regret_median="median", regret_mean="mean")
        .reset_index()
    )

    if mode == "minimax":
        summary = summary.sort_values(["regret_max", "regret_median", "regret_mean"])
    elif mode == "medianrobust":
        summary = summary.sort_values(["regret_median", "regret_max", "regret_mean"])
    else:
        raise ValueError("mode must be 'minimax' or 'medianrobust'")

    eps_star = float(summary.iloc[0]["epsilon"])
    return eps_star, summary


def select_golden_config(
    agg: pd.DataFrame,
    s_star: pd.DataFrame,
    epsilon_star: float,
    include_switch_ratio: bool = True,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Within chosen epsilon*, select a single config robustly across (dataset, max_iter)
    using minimax regret vs S_star.

    Returns:
      - golden_minimax dict
      - theta_summary df
      - theta_detail df (regret per dataset/budget)
    """
    sub = agg[np.isclose(agg["epsilon"], epsilon_star)].copy()
    if sub.empty:
        raise ValueError(f"No configs found for epsilon_star={epsilon_star}")

    key_cols = ["scheduler_label", "lr"]
    if include_switch_ratio and "switch_ratio" in sub.columns:
        key_cols.append("switch_ratio")

    theta_detail = sub[["dataset", "max_iter", "stress_agg"] + key_cols].merge(
        s_star, on=["dataset", "max_iter"], how="left"
    )
    theta_detail["regret_pct"] = 100.0 * (theta_detail["stress_agg"] - theta_detail["S_star"]) / theta_detail["S_star"]

    theta_summary = (
        theta_detail.groupby(key_cols, dropna=False)["regret_pct"]
        .agg(regret_max="max", regret_median="median", regret_mean="mean")
        .reset_index()
        .sort_values(["regret_max", "regret_median", "regret_mean"])
    )

    golden = theta_summary.iloc[0].to_dict()
    # Make JSON-friendly
    for k, v in list(golden.items()):
        if isinstance(v, (np.floating, np.integer)):
            golden[k] = float(v)

    return golden, theta_summary, theta_detail


# -----------------------------
# Main
# -----------------------------
def main():
    results_dir = DEFAULT_RESULTS_DIR.resolve()
    selection_dir = results_dir / "_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] Reading results from: {results_dir}")
    df = load_all_results(results_dir, DATASETS)

    # ---- NEW: restrict everything to one budget ----
    print(f"[filter] Keeping only T={TARGET_T}…")
    df = df[df["max_iter"] == TARGET_T].copy()
    if df.empty:
        raise ValueError(f"No rows found for max_iter={TARGET_T} in the combined results.")

    df = add_config_id_if_missing(df)
    df.to_csv(selection_dir / f"combined_results_T{TARGET_T}.csv", index=False)

    print(f"[agg] Aggregating per config using {AGG_STAT} stress over seeds/inits…")
    agg = aggregate_per_config(df, stat=AGG_STAT)
    agg.to_csv(selection_dir / f"agg_per_config_T{TARGET_T}.csv", index=False)

    print("[S*] Computing global best S* per (dataset, T)…")
    s_star = compute_S_star(agg)

    print("[eps] Computing epsilon regret table…")
    reg_tbl = epsilon_regret_table(agg, s_star)
    reg_tbl.to_csv(selection_dir / f"epsilon_regret_table_T{TARGET_T}.csv", index=False)

    print(f"[eps] Selecting epsilon* using mode='{EPSILON_MODE}'…")
    eps_star, eps_summary = select_epsilon(reg_tbl, mode=EPSILON_MODE)
    eps_summary.to_csv(selection_dir / f"epsilon_summary_T{TARGET_T}.csv", index=False)

    eps_payload = {
        "agg_stat": AGG_STAT,
        "epsilon_mode": EPSILON_MODE,
        "target_T": TARGET_T,
        "chosen_epsilon": eps_star,
    }
    with open(selection_dir / f"epsilon_selection_T{TARGET_T}.json", "w") as f:
        json.dump(eps_payload, f, indent=2)

    print(f"[golden] Selecting golden config within epsilon*={eps_star} (include_switch_ratio={INCLUDE_SWITCH_RATIO_IN_GOLDEN})…")
    golden, theta_summary, theta_detail = select_golden_config(
        agg, s_star, eps_star, include_switch_ratio=INCLUDE_SWITCH_RATIO_IN_GOLDEN
    )
    theta_summary.to_csv(selection_dir / f"golden_summary_T{TARGET_T}.csv", index=False)
    theta_detail.to_csv(selection_dir / f"golden_config_table_T{TARGET_T}.csv", index=False)

    golden_payload = {
        "agg_stat": AGG_STAT,
        "epsilon_star": eps_star,
        "target_T": TARGET_T,
        "include_switch_ratio": INCLUDE_SWITCH_RATIO_IN_GOLDEN,
        "golden_minimax": golden,
    }
    with open(selection_dir / f"golden_selection_T{TARGET_T}.json", "w") as f:
        json.dump(golden_payload, f, indent=2)

    print("\n=== DONE ===")
    print(f"Outputs: {selection_dir}")
    print(f"Target T: {TARGET_T}")
    print(f"Chosen epsilon*: {eps_star}")
    print("Golden config (minimax regret):")
    print(golden)


if __name__ == "__main__":
    main()
