"""
Experiment C: Visualization (Professional Upgrade).
Generates publication-quality plots:
1. Raw & Relative Stress Heatmaps.
2. Convergence Dashboards (Log-Stress + Distance-to-Optimum).
3. Budget Scalability Plots.
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib import gridspec
from pathlib import Path

# =============================================================================
# CONFIGURATION & STYLING
# =============================================================================

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_c"
DATASETS = ["fashion_mnist", "imdb", "s_curve"] 

# Professional Plotting Theme
sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

# Consistent Color Palette for Schedulers
# Hybrids = Blues/Purples, Exponential = Oranges, Constant = Grays
SCHEDULER_COLORS = {
    "hybrid (r=0.1)": "#c6dbef",
    "hybrid (r=0.2)": "#9ecae1",
    "hybrid (r=0.3)": "#6baed6",
    "hybrid (r=0.4)": "#3182bd",
    "hybrid (r=0.5)": "#08519c",
    "exponential": "#e6550d",
    "constant": "#636363",
    "harmonic": "#31a354"
}

# Fallback for unknown schedulers
DEFAULT_PALETTE = sns.color_palette("husl", 10)

def get_color(label):
    return SCHEDULER_COLORS.get(label, "#333333")

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


def aggregate_runs(df, stress_agg="median"):
    """
    Aggregates multiple init runs per (dataset, config_id) into one row.
    Expects columns: dataset, config_id, max_iter, scheduler_label, lr, epsilon, stress, time
    """
    group_cols = ["dataset", "config_id", "max_iter", "scheduler_label", "lr", "epsilon"]
    # keep switch_ratio if present (useful for hybrid labeling/debug)
    if "switch_ratio" in df.columns:
        group_cols.append("switch_ratio")

    g = df.groupby(group_cols, dropna=False)

    out = g.agg(
        stress_median=("stress", "median"),
        stress_mean=("stress", "mean"),
        stress_std=("stress", "std"),
        stress_q25=("stress", lambda x: np.quantile(x, 0.25)),
        stress_q75=("stress", lambda x: np.quantile(x, 0.75)),
        time_median=("time", "median"),
        n_runs=("stress", "size"),
    ).reset_index()

    out["stress_agg"] = out["stress_median"] if stress_agg == "median" else out["stress_mean"]
    return out



def plot_relative_regret_heatmaps_grid(
    df_agg,
    dataset_name,
    out_dir,
    max_iters=(15, 30, 50),
    epsilons=(0.001, 0.01, 0.1),
    vmax=50.0,
):
    max_iters = list(max_iters)
    epsilons = list(epsilons)

    sched_order = sorted(df_agg["scheduler_label"].unique(), key=lambda s: ("hybrid" not in s, s))
    lr_order = sorted(df_agg["lr"].unique(), key=lambda v: float(v))

    n_rows = len(max_iters)
    n_cols = len(epsilons)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.8 * n_cols, 3.6 * n_rows),
        sharex=True, sharey=True
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for r, it in enumerate(max_iters):
        for c, eps in enumerate(epsilons):
            ax = axes[r, c]
            subset = df_agg[(df_agg["max_iter"] == it) & (df_agg["epsilon"] == eps)]

            if subset.empty:
                ax.set_axis_off()
                continue

            pivot_raw = subset.pivot(index="scheduler_label", columns="lr", values="stress_agg")
            pivot_raw = pivot_raw.reindex(index=sched_order, columns=lr_order)

            # --- best stress (within this budget+epsilon) ---
            min_stress = np.nanmin(pivot_raw.values)

            # Relative regret (% worse than best)
            pivot_rel = (pivot_raw - min_stress) / min_stress * 100.0

            sns.heatmap(
                pivot_rel,
                ax=ax,
                annot=True,
                fmt=".3f",
                cmap="Reds",
                vmin=0,
                vmax=vmax,
                linewidths=0.5,
                cbar=(r == 0 and c == n_cols - 1),
                cbar_kws={'label': '% Worse than Best'} if (r == 0 and c == n_cols - 1) else None,
                # annot_kws={"size": 9},  # optional readability tweak
            )

            # Highlight best cell (same as before)
            best_row, best_col = pivot_raw.stack().idxmin()
            rr = list(pivot_raw.index).index(best_row)
            cc = list(pivot_raw.columns).index(best_col)
            ax.add_patch(Rectangle((cc, rr), 1, 1, fill=False, edgecolor='blue', lw=2.5, clip_on=False))

            # --- titles/labels ---
            # Column headers: epsilon only (top row)
            if r == 0:
                ax.set_title(f"$\\epsilon$ = {eps}", pad=18)

            # Per-heatmap annotation: best value above *every* heatmap
            ax.text(
                0.5, 1.03,
                f"best = {min_stress:.4e}",
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=11
            )

            # Row labels as you already do
            if c == 0:
                ax.set_ylabel(f"T = {it}\n\nscheduler")
            else:
                ax.set_ylabel("")
            if r == n_rows - 1:
                ax.set_xlabel("lr")
            else:
                ax.set_xlabel("")

    fig.suptitle(f"{dataset_name}: Relative Regret across Budgets and $\\epsilon$ (median over seeds)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = out_dir / f"heatmap_regret_grid_{dataset_name}.png"
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_budget_efficiency(df, dataset_name, out_dir):
    """
    Plots the scalability of each scheduler as budget increases.
    Uses dashed lines for styling distinction.
    """
    best_per_budget = df.groupby(["scheduler_label", "max_iter"])["stress"].min().reset_index()
    
    plt.figure(figsize=(12, 8))
    
    # Sort labels to ensure consistent legend order
    labels = sorted(best_per_budget["scheduler_label"].unique())

    print(best_per_budget[best_per_budget["scheduler_label"].str.contains("exponential", na=False)])
    print(best_per_budget[best_per_budget["scheduler_label"].str.contains("harmonic", na=False)])
    
    for label in labels:
        data = best_per_budget[best_per_budget["scheduler_label"] == label]
        
        # Style logic
        ls = "-" if "hybrid" in label else "--"
        marker = "o" if "hybrid" in label else "s"
        lw = 2.5 if "hybrid" in label else 2
        alpha = 0.9 if "hybrid" in label else 0.7
        
        plt.plot(data["max_iter"], data["stress"], label=label, 
                 color=get_color(label), linestyle=ls, marker=marker, linewidth=lw, alpha=alpha)
    
    plt.xlabel("Iteration Budget")
    plt.ylabel("Best Final Stress")
    plt.title(f"{dataset_name}: Convergence Efficiency vs Budget")
    plt.grid(True, alpha=0.3)
    plt.xticks(sorted(df["max_iter"].unique()))
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Scheduler")
    plt.tight_layout()
    plt.savefig(out_dir / f"efficiency_curve_{dataset_name}.pdf", dpi=300)
    plt.close()

def plot_convergence_dashboard(df, histories, dataset_name, out_dir, max_iter):
    """
    Creates a dashboard with two panels:
    1. Log-Scale Convergence (The standard view).
    2. Distance to Best (Optimization View) - Shows rate of convergence clearly.
    """
    subset = df[df["max_iter"] == max_iter]
    
    # Get top configurations (Top 1 per scheduler type to avoid clutter)
    best_runs = subset.loc[subset.groupby("scheduler_label")["stress"].idxmin()].sort_values("stress")
    
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    # Find global minimum for "Distance to Best" calculation
    # We scan ALL histories for the absolute lowest stress ever recorded
    global_min = float('inf')
    for run_id, data in histories.items():
        if not data["history"]: continue
        run_min = min(h[1] for h in data["history"])
        if run_min < global_min: global_min = run_min
            
    # Add a tiny epsilon to avoid log(0)
    global_min -= 1e-5 
    
    for _, row in best_runs.iterrows():
        run_id = row["run_id"]
        label = row['scheduler_label']
        color = get_color(label)
        
        if run_id in histories:
            history_data = histories[run_id]["history"]
            if not history_data: continue
            
            times, stresses = zip(*history_data)
            epochs = np.arange(len(stresses))
            stresses = np.array(stresses)
            
            # --- Panel 1: Standard Log Scale ---
            # Exclude Epoch 0 for scaling
            if len(stresses) > 1:
                ax1.plot(epochs, stresses, label=label, color=color, linewidth=2, alpha=0.8)
            
            # --- Panel 2: Distance to Best (Log Scale) ---
            # diff = stress - global_min
            diff = stresses - global_min
            diff[diff <= 0] = 1e-8 # Safety clamp
            
            if len(diff) > 1:
                ax2.plot(epochs, diff, label=label, color=color, linewidth=2, alpha=0.8)

    # Styling Panel 1
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Raw Stress (Log)")
    ax1.set_yscale("log")
    ax1.set_title("Absolute Convergence Trajectory")
    ax1.grid(True, which="both", alpha=0.2)
    
    # Auto-scale Y (Skip Epoch 0)
    # We do this manually to avoid the "L-shape" issue
    y_vals = []
    for line in ax1.get_lines():
        y_vals.extend(line.get_ydata()[1:]) # Skip index 0
    if y_vals:
        ax1.set_ylim(min(y_vals)*0.99, max(y_vals))

    # Styling Panel 2
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Distance to Optimum (Log)")
    ax2.set_yscale("log")
    ax2.set_title(f"Optimization Rate (Stress - {global_min:.2e})")
    ax2.grid(True, which="both", alpha=0.2)
    ax2.legend(loc='upper right', fontsize='small', frameon=True)

    plt.suptitle(f"{dataset_name}: Convergence Dynamics (Budget={max_iter})", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_dir / f"convergence_dashboard_iter_{max_iter}_{dataset_name}.pdf", dpi=300)
    plt.close()

def plot_relative_regret_heatmaps_triptych_fixed_epsilon(
    df_agg,
    dataset_name,
    out_dir,
    epsilon=0.001,
    max_iters=(15, 30, 50),
    vmax=10.0,
):
    """
    One row of 3 relative-regret heatmaps (T=15/30/50), for a fixed epsilon.
    Each heatmap shows % worse than the best configuration within that (T, epsilon).
    Uses df_agg with column 'stress_agg'.

    Saves:
      heatmap_regret_triptych_eps_<epsilon>_<dataset_name>.png
    """
    max_iters = list(max_iters)

    # Stable ordering
    sched_order = sorted(df_agg["scheduler_label"].unique(), key=lambda s: ("hybrid" not in s, s))
    lr_order = sorted(df_agg["lr"].unique(), key=lambda v: float(v))

    fig, axes = plt.subplots(1, len(max_iters), figsize=(6.8 * len(max_iters), 7.5), sharey=True)

    if len(max_iters) == 1:
        axes = [axes]

    for i, it in enumerate(max_iters):
        ax = axes[i]
        subset = df_agg[(df_agg["max_iter"] == it) & (df_agg["epsilon"] == epsilon)]

        if subset.empty:
            ax.set_axis_off()
            ax.set_title(f"T = {it}\n(no data)")
            continue

        pivot_raw = subset.pivot(index="scheduler_label", columns="lr", values="stress_agg")
        pivot_raw = pivot_raw.reindex(index=sched_order, columns=lr_order)

        # best stress within this (T, eps)
        min_stress = np.nanmin(pivot_raw.values)
        pivot_rel = (pivot_raw - min_stress) / min_stress * 100.0

        sns.heatmap(
            pivot_rel,
            ax=ax,
            annot=True,
            fmt=".3f",
            cmap="Reds",
            vmin=0,
            vmax=vmax,
            linewidths=0.5,
            cbar=(i == len(max_iters) - 1),  # colorbar only on last heatmap
            cbar_kws={'label': '% Worse than Best'} if (i == len(max_iters) - 1) else None,
        )

        # Highlight best
        best_row, best_col = pivot_raw.stack().idxmin()
        rr = list(pivot_raw.index).index(best_row)
        cc = list(pivot_raw.columns).index(best_col)
        ax.add_patch(Rectangle((cc, rr), 1, 1, fill=False, edgecolor='blue', lw=2.5, clip_on=False))

        ax.set_title(f"T = {it}\n(best = {min_stress:.4e})")

        ax.set_xlabel("lr", labelpad=18)
        if i == 0:
            ax.set_ylabel("scheduler")
        else:
            ax.set_ylabel("")

    fig.suptitle(
        f"{dataset_name}: Relative Regret Heatmaps (lower is better)",
        y=0.975
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    # filename safe epsilon (avoid dots)
    eps_str = str(epsilon).replace(".", "p")
    out_path = out_dir / f"heatmap_regret_triptych_{dataset_name}.pdf"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# =============================================================================
# MAIN VISUALIZATION LOOP
# =============================================================================

def visualize_dataset(dataset_name):
    target_dir = RESULTS_DIR / dataset_name
    if not target_dir.exists():
        print(f"No results found for {dataset_name}.")
        return

    print(f"Generating professional plots for {dataset_name}...")

    try:
        df = pd.read_csv(target_dir / "results.csv")
        with open(target_dir / "histories.json", "r") as f:
            histories = json.load(f)
    except FileNotFoundError:
        print("Missing data files.")
        return

    # --- Aggregate over seeds/inits (median is the default recommendation) ---
    df_agg = aggregate_runs(df, stress_agg="median")

    # Budgets: use the ones you actually ran
    max_iters = tuple(sorted(df_agg["max_iter"].unique()))
    epsilons = tuple(sorted(df_agg["epsilon"].unique()))

    # 1) Regret heatmaps: budgets x epsilons in one figure
    plot_relative_regret_heatmaps_grid(
        df_agg=df_agg,
        dataset_name=dataset_name,
        out_dir=target_dir,
        max_iters=max_iters,
        epsilons=epsilons,
        vmax=10.0,
    )

    # 2) Efficiency curve: should use aggregated stress (median), not raw rows
    plot_budget_efficiency(df_agg.rename(columns={"stress_agg": "stress"}), dataset_name, target_dir)

    # 3) Convergence dashboards
    # Recommendation: pick ONE epsilon to show (otherwise too many plots).
    # Choose epsilon that yields best median stress at largest budget.
    #max_budget = max_iters[-1]
    #best_eps = (
    #    df_agg[df_agg["max_iter"] == max_budget]
    #    .groupby("epsilon")["stress_agg"].min()
    #    .idxmin()
    #)
    best_eps = 0.001
    max_budget = 30
    print(f"  Using epsilon={best_eps} for convergence dashboards (best at T={max_budget}).")

    df_for_curves = df[df["epsilon"] == best_eps]
    for it in max_iters:
        plot_convergence_dashboard(df_for_curves, histories, dataset_name, target_dir, it)

    plot_relative_regret_heatmaps_triptych_fixed_epsilon(
        df_agg=df_agg,
        dataset_name=dataset_name,
        out_dir=target_dir,
        epsilon=0.001,
        max_iters=max_iters,  # or explicitly (15, 30, 50)
        vmax=8.0,
    )

    print(f"  -> Saved to {target_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_datasets = sys.argv[1:]
    else:
        target_datasets = DATASETS
        
    for ds in target_datasets:
        visualize_dataset(ds)