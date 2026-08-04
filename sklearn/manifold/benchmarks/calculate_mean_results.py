from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).parent / "results" / "experiment_b" / "scaling_mixed_euclid_vs_lazy"
DATASETS = ["s_curve", "swiss_roll", "torus"]

ALGO_ORDER = ["SMACOF-euclidean", "SGD-euclidean", "SGD-lazy"]


def load_results_csv(dataset: str) -> pd.DataFrame:
    path = RESULTS_DIR / dataset / "results.csv"
    df = pd.read_csv(path)
    df["dataset"] = df.get("dataset", dataset)
    df["n_samples"] = df["n_samples"].astype(int)
    df["time_solver"] = df["time_solver"].astype(float)
    df["algo"] = df["algo"].astype(str)
    df["run_id"] = df["run_id"].astype(str)
    return df


def load_histories_json(dataset: str) -> dict:
    path = RESULTS_DIR / dataset / "histories.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing histories.json for {dataset}: {path}")
    with open(path, "r") as f:
        return json.load(f)


def last_cum_time_from_history(history) -> float:
    """History is expected as list of [cum_time, stress] pairs."""
    if not history:
        return np.nan
    last = history[-1]
    if isinstance(last, (list, tuple)) and len(last) >= 1:
        try:
            return float(last[0])
        except Exception:
            return np.nan
    return np.nan


def build_cum_time_df(histories: dict) -> pd.DataFrame:
    rows = []
    for run_id, rec in histories.items():
        hist = rec.get("history", None)
        cum_t = last_cum_time_from_history(hist)
        rows.append(
            {
                "run_id": str(run_id),
                "cum_time": float(cum_t) if np.isfinite(cum_t) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_mean_results_for_dataset(dataset: str) -> pd.DataFrame:
    df = load_results_csv(dataset)
    histories = load_histories_json(dataset)
    df_cum = build_cum_time_df(histories)

    # Merge cum_time onto per-run results
    df = df.merge(df_cum, on="run_id", how="left")

    # Now aggregate for mean_results.csv
    g = df.groupby(["dataset", "algo", "n_samples"], dropna=False)
    out = g.agg(
        mean_total_time=("time_solver", "mean"),
        std_total_time=("time_solver", "std"),
        mean_cum_time=("cum_time", "mean"),
        std_cum_time=("cum_time", "std"),
        n_runs=("run_id", "size"),
    ).reset_index()

    # Ratios/overhead (only where cum_time exists)
    out["ratio_total_over_cum"] = out["mean_total_time"] / out["mean_cum_time"]
    out["frac_overhead"] = (out["mean_total_time"] - out["mean_cum_time"]) / out["mean_total_time"]

    # Stable ordering
    out["algo"] = pd.Categorical(out["algo"], categories=ALGO_ORDER, ordered=True)
    out = out.sort_values(["dataset", "algo", "n_samples"])

    return out


def main():
    all_out = []
    for ds in DATASETS:
        all_out.append(compute_mean_results_for_dataset(ds))

    out_df = pd.concat(all_out, ignore_index=True)
    out_path = RESULTS_DIR / "mean_results.csv"
    out_df.to_csv(out_path, index=False)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
