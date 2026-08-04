"""
Experiment B: Scaling Study on Synthetic Datasets (Plotting).

Reads:
  results/experiment_b/<EXPERIMENT_RUN_NAME>/<dataset>/results.csv

Writes:
  - runtime_vs_n_<dataset>.pdf/.png
  - speedup_vs_n_<dataset>.pdf/.png
  - stress_vs_n_<dataset>.pdf/.png (optional)
  - combined_runtime_vs_n.pdf/.png (all datasets)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
RESULTS_DIR = Path(__file__).parent / "results" / "experiment_b" / "scaling_synthetic_euclid_vs_lazy"
DATASETS = ["s_curve", "swiss_roll", "torus"]

# Order and display names for algorithms
ALGO_ORDER = ["SMACOF-euclidean", "SGD-euclidean", "SGD-lazy"]
ALGO_LABEL = {
    "SMACOF-euclidean": "SMACOF (euclidean)",
    "SGD-euclidean": "SGD-MDS (euclidean)",
    "SGD-lazy": "SGD-MDS (lazy)",
}
N_MAX_PLOT = 5000

# Plot styling (kept simple; matplotlib default colors)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
})

# -----------------------------
# HELPERS
# -----------------------------
def load_dataset_results(dataset: str) -> pd.DataFrame:
    path = RESULTS_DIR / dataset / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)

    # Sanity conversions
    df["n_samples"] = df["n_samples"].astype(int)
    df["algo"] = df["algo"].astype(str)
    df["time_solver"] = df["time_solver"].astype(float)
    df["stress"] = df["stress"].astype(float)
    return df

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize per (dataset, algo, n_samples) over repeats.
    Computes median + IQR (25/75), mean, std.
    """
    gcols = ["dataset", "algo", "n_samples"]
    g = df.groupby(gcols, dropna=False)

    out = g.agg(
        time_median=("time_solver", "median"),
        time_mean=("time_solver", "mean"),
        time_std=("time_solver", "std"),
        time_q25=("time_solver", lambda x: np.quantile(x, 0.25)),
        time_q75=("time_solver", lambda x: np.quantile(x, 0.75)),
        stress_median=("stress", "median"),
        stress_mean=("stress", "mean"),
        stress_std=("stress", "std"),
        stress_q25=("stress", lambda x: np.quantile(x, 0.25)),
        stress_q75=("stress", lambda x: np.quantile(x, 0.75)),
        n_runs=("time_solver", "size"),
    ).reset_index()

    # enforce algo order in plotting
    out["algo"] = pd.Categorical(out["algo"], categories=ALGO_ORDER, ordered=True)
    out = out.sort_values(["dataset", "algo", "n_samples"])
    return out

def _plot_lines_with_iqr(ax, sub, xcol, y_med, y_q25, y_q75, title, ylabel):
    for algo in ALGO_ORDER:
        a = sub[sub["algo"] == algo]
        if a.empty:
            continue
        x = a[xcol].to_numpy()
        y = a[y_med].to_numpy()
        lo = a[y_q25].to_numpy()
        hi = a[y_q75].to_numpy()

        ax.plot(x, y, marker="o", linewidth=2, label=ALGO_LABEL.get(algo, algo))
        ax.fill_between(x, lo, hi, alpha=0.15)

    ax.set_title(title)
    ax.set_xlabel("N (samples)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)

def save_fig(fig, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"))
    plt.close(fig)

# -----------------------------
# PLOTS
# -----------------------------
def plot_runtime_vs_n(summary_df: pd.DataFrame, dataset: str, out_dir: Path):
    sub = summary_df[summary_df["dataset"] == dataset].copy()
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    _plot_lines_with_iqr(
        ax,
        sub=sub,
        xcol="n_samples",
        y_med="time_median",
        y_q25="time_q25",
        y_q75="time_q75",
        title=f"{dataset}: Runtime scaling (median over repeats, IQR band)",
        ylabel="Runtime per run (seconds)",
    )

    ax.legend(loc="upper left")
    save_fig(fig, out_dir / f"runtime_vs_n_{dataset}")

def plot_speedup_vs_n(summary_df: pd.DataFrame, dataset: str, out_dir: Path):
    """
    Speedup = (SMACOF median runtime) / (algo median runtime).
    Uses medians only (keeps it readable).
    """
    sub = summary_df[summary_df["dataset"] == dataset].copy()

    # pivot for speedup
    piv = sub.pivot_table(index="n_samples", columns="algo", values="time_median")
    if "SMACOF-euclidean" not in piv.columns:
        print(f"[Warn] No SMACOF-euclidean for {dataset}; skipping speedup.")
        return

    sm = piv["SMACOF-euclidean"]
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    for algo in ["SGD-euclidean", "SGD-lazy"]:
        if algo not in piv.columns:
            continue
        speedup = sm / piv[algo]
        ax.plot(speedup.index.to_numpy(), speedup.to_numpy(), marker="o", linewidth=2, label=ALGO_LABEL[algo])

    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_title(f"{dataset}: Speedup vs SMACOF (median runtime ratio)")
    ax.set_xlabel("N (samples)")
    ax.set_ylabel("Speedup (SMACOF time / algo time)")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.25)

    save_fig(fig, out_dir / f"speedup_vs_n_{dataset}")

def plot_stress_vs_n(summary_df: pd.DataFrame, dataset: str, out_dir: Path):
    sub = summary_df[summary_df["dataset"] == dataset].copy()
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    _plot_lines_with_iqr(
        ax,
        sub=sub,
        xcol="n_samples",
        y_med="stress_median",
        y_q25="stress_q25",
        y_q75="stress_q75",
        title=f"{dataset}: Final stress vs N (median over repeats, IQR band)",
        ylabel="Final stress",
    )

    ax.legend(loc="upper left")
    save_fig(fig, out_dir / f"stress_vs_n_{dataset}")

def plot_combined_runtime(summary_df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 6.4), sharey=False)

    DATASET_DIMS = {
        "s_curve": 100,
        "swiss_roll": 300,
        "torus": 500,
    }

    if len(DATASETS) == 1:
        axes = [axes]

    for i, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        sub = summary_df[summary_df["dataset"] == dataset].copy()
        D = DATASET_DIMS.get(dataset, None)
        _plot_lines_with_iqr(
            ax,
            sub=sub,
            xcol="n_samples",
            y_med="time_median",
            y_q25="time_q25",
            y_q75="time_q75",
            title = f"{dataset} (D = {D})" if D is not None else dataset,          # e.g. "s_curve (D = 100)"
            ylabel="Runtime (s)" if i == 0 else "",  # optional: only left y-label
        )

        # Remove the middle x-label to avoid colliding with legend
        if len(axes) == 3 and i == 1:
            ax.set_xlabel("")

        ax.set_ylim(0, 100)

    # Collect legend handles (dedup)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        borderaxespad=0.0,
    )

    #fig.suptitle("Experiment B: Runtime scaling on synthetic datasets", y=0.98)

    # More bottom room for legend, more overall height
    fig.subplots_adjust(top=0.88, bottom=0.22, wspace=0.28)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_dir / "combined_runtime_vs_n")


# -----------------------------
# MAIN
# -----------------------------
def main():
    all_rows = []
    for ds in DATASETS:
        df = load_dataset_results(ds)
        # ensure dataset column exists (your script writes it)
        if "dataset" not in df.columns:
            df["dataset"] = ds
        all_rows.append(df)

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all = df_all[df_all["n_samples"] <= N_MAX_PLOT].copy()

    # summarize across repeats
    df_sum = summarize(df_all)

    combined_dir = RESULTS_DIR / "_plots_combined"

    for ds in DATASETS:
        out_dir = RESULTS_DIR / ds
        plot_runtime_vs_n(df_sum, ds, out_dir)
        plot_speedup_vs_n(df_sum, ds, out_dir)
        plot_stress_vs_n(df_sum, ds, out_dir)

    plot_combined_runtime(df_sum, combined_dir)

    print(f"Saved plots under: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
