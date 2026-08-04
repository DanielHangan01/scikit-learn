from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================
RESULTS_DIR = Path(__file__).parent / "results" / "experiment_a" / "sgd_30_epochs"
OUT_DIR = RESULTS_DIR  # write tables next to results


# =============================================================================
# STATS HELPERS
# =============================================================================
def _t_crit_975(df: int) -> float:
    """t_{0.975, df}. Falls back to ~1.96 if scipy not available."""
    try:
        from scipy.stats import t  # type: ignore
        return float(t.ppf(0.975, df))
    except Exception:
        return 1.96


def _paired_t_pvalue(t_stat: float, df: int) -> float:
    """Two-sided p-value for paired t-stat. Returns NaN if scipy not available."""
    try:
        from scipy.stats import t  # type: ignore
        return float(2.0 * (1.0 - t.cdf(abs(t_stat), df)))
    except Exception:
        return float("nan")


def mean_ci95(x: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Returns (mean, std, ci_low, ci_high) for 95% CI of mean using t-interval.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    mu = float(np.mean(x))
    if n == 1:
        return (mu, np.nan, mu, mu)

    s = float(np.std(x, ddof=1))
    se = s / math.sqrt(n) if np.isfinite(s) else np.nan
    tcrit = _t_crit_975(n - 1)
    half = tcrit * se if np.isfinite(se) else np.nan
    return (mu, s, mu - half, mu + half)


def paired_stress_stats(stress_sgd: np.ndarray, stress_smacof: np.ndarray) -> dict:
    """
    Paired stats for delta_stress = SGD - SMACOF:
      - mean, std, ci95, t_stat, p_value (if scipy available)
    """
    stress_sgd = np.asarray(stress_sgd, dtype=float)
    stress_smacof = np.asarray(stress_smacof, dtype=float)

    mask = np.isfinite(stress_sgd) & np.isfinite(stress_smacof)
    d = (stress_sgd[mask] - stress_smacof[mask]).astype(float)

    mu, s, lo, hi = mean_ci95(d)
    n = int(d.size)

    if n >= 2 and np.isfinite(s) and s > 0:
        se = s / math.sqrt(n)
        t_stat = mu / se
        p_val = _paired_t_pvalue(t_stat, n - 1)
    else:
        t_stat = np.nan
        p_val = np.nan

    return {
        "delta_mean": mu,
        "delta_std": s,
        "delta_ci95_lo": lo,
        "delta_ci95_hi": hi,
        "t_stat": t_stat,
        "p_value": p_val,
        "n": n,
    }


# =============================================================================
# IO
# =============================================================================
def load_all_runs(results_dir: Path) -> pd.DataFrame:
    """
    Loads all run-level results from results.json across dataset folders.
    Returns long DF with columns: dataset, algo, seed, stress, time.
    """
    rows = []
    for ds_dir in sorted(results_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        res_path = ds_dir / "results.json"
        if not res_path.exists():
            continue

        with open(res_path, "r") as f:
            data = json.load(f)

        dataset = data.get("dataset", ds_dir.name)
        for r in data.get("runs", []):
            algo = r.get("algo")
            if algo not in ("SMACOF", "SGD-MDS"):
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "algo": algo,
                    "seed": int(r.get("seed")),
                    "stress": float(r.get("stress")),
                    "time": float(r.get("time")),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No runs found under {results_dir}")
    return df


# =============================================================================
# SUMMARIES
# =============================================================================
def summarize_means_per_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-dataset mean stress/time for each algo + derived metrics.
    Output columns:
      dataset, stress_smacof, stress_sgd, time_smacof, time_sgd, speedup, stress_ratio
    """
    g = df.groupby(["dataset", "algo"], dropna=False).agg(
        stress_mean=("stress", "mean"),
        time_mean=("time", "mean"),
    ).reset_index()

    piv = g.pivot(index="dataset", columns="algo")
    piv.columns = [f"{metric}_{algo.replace('-', '')}" for metric, algo in piv.columns]
    piv = piv.reset_index()

    piv["speedup"] = piv["time_mean_SMACOF"] / piv["time_mean_SGDMDS"]
    piv["stress_ratio"] = piv["stress_mean_SGDMDS"] / piv["stress_mean_SMACOF"]

    out = piv.rename(
        columns={
            "stress_mean_SMACOF": "stress_smacof",
            "stress_mean_SGDMDS": "stress_sgd",
            "time_mean_SMACOF": "time_smacof",
            "time_mean_SGDMDS": "time_sgd",
        }
    )[
        [
            "dataset",
            "stress_smacof",
            "stress_sgd",
            "time_smacof",
            "time_sgd",
            "speedup",
            "stress_ratio",
        ]
    ].sort_values("dataset")

    return out


def summarize_paired_stress_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each dataset, compute paired statistics over matched seeds for:
      delta_stress = stress_SGD - stress_SMACOF

    We assume n_init=20 across the board, so we do NOT include an 'n' column in the table.
    """
    wide = df.pivot_table(index=["dataset", "seed"], columns="algo", values="stress", aggfunc="mean").reset_index()

    if "SMACOF" not in wide.columns or "SGD-MDS" not in wide.columns:
        raise RuntimeError("Missing SMACOF or SGD-MDS in stress pivot; check input data.")

    # keep only seeds where both exist
    wide = wide.dropna(subset=["SMACOF", "SGD-MDS"])

    rows = []
    for dataset, sub in wide.groupby("dataset", dropna=False):
        st = paired_stress_stats(sub["SGD-MDS"].to_numpy(), sub["SMACOF"].to_numpy())

        # “Mean not contained in 95% CI” note:
        # For delta, 0 not in CI => likely statistically significant (two-sided).
        sig_ci = bool(np.isfinite(st["delta_ci95_lo"]) and np.isfinite(st["delta_ci95_hi"]) and
                      not (st["delta_ci95_lo"] <= 0.0 <= st["delta_ci95_hi"]))

        rows.append(
            {
                "dataset": dataset,
                "delta_stress_mean": st["delta_mean"],      # SGD - SMACOF (negative => SGD better)
                "delta_stress_std": st["delta_std"],
                "delta_stress_ci95_lo": st["delta_ci95_lo"],
                "delta_stress_ci95_hi": st["delta_ci95_hi"],
                "delta_stress_t": st["t_stat"],
                "delta_stress_p": st["p_value"],            # blank if scipy missing
                "sig_by_ci": sig_ci,                         # 0 not in CI
            }
        )

    return pd.DataFrame(rows).sort_values("dataset")


# =============================================================================
# LATEX WRITERS
# =============================================================================
def to_latex_table_means(means: pd.DataFrame) -> str:
    fmt = {
        "stress_smacof": "{:.3e}",
        "stress_sgd": "{:.3e}",
        "time_smacof": "{:.3f}",
        "time_sgd": "{:.3f}",
        "speedup": "{:.2f}",
        "stress_ratio": "{:.3f}",
    }

    def f(col, v):
        if pd.isna(v):
            return ""
        return fmt[col].format(v) if col in fmt else str(v)

    lines = []
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Stress (SMACOF) & Stress (SGD) & Time (SMACOF) & Time (SGD) & Speedup & Stress ratio \\")
    lines.append(r"\midrule")

    for _, r in means.iterrows():
        lines.append(
            f"{r['dataset']} & "
            + " & ".join(
                [
                    f("stress_smacof", r["stress_smacof"]),
                    f("stress_sgd", r["stress_sgd"]),
                    f("time_smacof", r["time_smacof"]),
                    f("time_sgd", r["time_sgd"]),
                    f("speedup", r["speedup"]),
                    f("stress_ratio", r["stress_ratio"]),
                ]
            )
            + r" \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def to_latex_table_paired_stress(stats: pd.DataFrame) -> str:
    """
    Paired delta stress statistics table:
      Δσ = σ_SGD - σ_SMACOF

    Key interpretability:
      - If 0 is NOT in the 95% CI of Δσ, the difference is likely statistically significant.
      - 'sig' column marks that condition (✓ / blank).
      - p-values included if SciPy is available, otherwise blank.
    """
    def f_e(x):
        return "" if pd.isna(x) else f"{x:.3e}"

    def f_p(x):
        if pd.isna(x):
            return ""
        if x < 1e-4:
            return r"$<10^{-4}$"
        return f"{x:.4f}"

    def f_sig(b):
        return r"$\checkmark$" if bool(b) else ""

    lines = []
    # dataset + 4 numeric + p + sig = 7 columns
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & $\Delta\sigma$ mean & $\Delta\sigma$ CI lo & $\Delta\sigma$ CI hi & $\Delta\sigma$ std & $p$ & sig \\"
    )
    lines.append(r"\midrule")

    for _, r in stats.iterrows():
        lines.append(
            f"{r['dataset']} & "
            f"{f_e(r['delta_stress_mean'])} & "
            f"{f_e(r['delta_stress_ci95_lo'])} & "
            f"{f_e(r['delta_stress_ci95_hi'])} & "
            f"{f_e(r['delta_stress_std'])} & "
            f"{f_p(r['delta_stress_p'])} & "
            f"{f_sig(r['sig_by_ci'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    df = load_all_runs(RESULTS_DIR)

    means = summarize_means_per_dataset(df)
    paired_stress = summarize_paired_stress_only(df)

    latex_means = to_latex_table_means(means)
    latex_paired = to_latex_table_paired_stress(paired_stress)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "table_experiment_a_per_dataset.tex").write_text(latex_means)
    (OUT_DIR / "table_experiment_a_paired_stress_stats.tex").write_text(latex_paired)

    # CSVs for convenience/debugging
    means.to_csv(OUT_DIR / "experiment_a_means_per_dataset.csv", index=False)
    paired_stress.to_csv(OUT_DIR / "experiment_a_paired_stress_stats.csv", index=False)

    print("Wrote:")
    print(" -", OUT_DIR / "table_experiment_a_per_dataset.tex")
    print(" -", OUT_DIR / "table_experiment_a_paired_stress_stats.tex")
    print(" -", OUT_DIR / "experiment_a_means_per_dataset.csv")
    print(" -", OUT_DIR / "experiment_a_paired_stress_stats.csv")

    if paired_stress["delta_stress_p"].isna().all():
        print(
            "\n[Info] p-values are blank because scipy is not available. "
            "The 95% CI + 'sig' column are still computed (t-crit falls back to ~1.96 if needed)."
        )


if __name__ == "__main__":
    main()
