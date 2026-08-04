"""
Experiment A: Visualization.
Generates:
1. Convergence Speed (Time) - Per Dataset
2. Algorithmic Efficiency (Epochs) - Per Dataset
3. Side-by-Side Embeddings (Quality) - Per Dataset
4. Stability Analysis (Box Plots) - Per Dataset & Global Summary
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_a" / "sgd_30_epochs"

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_convergence(dataset_name, best_smacof, best_sgd, out_dir):
    """
    Plots two charts: Stress vs Time and Stress vs Epochs.
    Adds a horizontal baseline for the global minimum stress.
    """
    # Prepare Data
    smacof_hist = np.array(best_smacof["history"])
    sgd_hist = np.array(best_sgd["history"])
    
    # --- Determine Common Y-Axis Limits (Zoomed) ---
    all_stress = []
    if len(smacof_hist) > 1: all_stress.extend(smacof_hist[1:, 1])
    if len(sgd_hist) > 1: all_stress.extend(sgd_hist[1:, 1])
    
    if all_stress:
        y_min = min(all_stress)
        y_bottom = y_min * 0.95
        y_top = y_min * 2.0 
        global_min_stress = y_min
    else:
        y_bottom, y_top = None, None
        global_min_stress = None

    # --- PLOT 1: TIME (Log Scale) ---
    plt.figure(figsize=(10, 7))
    
    if global_min_stress:
        plt.axhline(y=global_min_stress, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
                    label=f"Global Min ({global_min_stress:.2e})")

    plt.plot(smacof_hist[:, 0], smacof_hist[:, 1], label=f"SMACOF (Best)", 
             color="black", linestyle="--", linewidth=2, alpha=0.8)
    
    plt.plot(sgd_hist[:, 0], sgd_hist[:, 1], label=f"SGD-MDS (Best)", 
             color="#d62728", linestyle="-", linewidth=2.5) 
    
    plt.yscale("log")
    if y_bottom:
        plt.ylim(y_bottom, y_top)
        
    plt.xlabel("Time (seconds)")
    plt.ylabel("Stress (Log Scale)")
    plt.title(f"{dataset_name}: Convergence Speed (Time)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_time.pdf", dpi=300)
    plt.close()
    
    # --- PLOT 2: ITERATIONS (Efficiency) ---
    plt.figure(figsize=(10, 7))
    
    if global_min_stress:
        plt.axhline(y=global_min_stress, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
                    label=f"Global Min ({global_min_stress:.2e})")
    
    plt.plot(range(len(smacof_hist)), smacof_hist[:, 1], label="SMACOF Iterations",
             color="black", linestyle="--", linewidth=2, alpha=0.8)
             
    plt.plot(range(len(sgd_hist)), sgd_hist[:, 1], label="SGD Epochs",
             color="#d62728", linestyle="-", linewidth=2.5)
             
    plt.yscale("log")
    if y_bottom:
        plt.ylim(y_bottom, y_top)
        
    plt.xlabel("Passes through Data (Iterations / Epochs)")
    plt.ylabel("Stress (Log Scale)")
    plt.title(f"{dataset_name}: Algorithmic Efficiency")
    plt.legend()
    plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_epochs.pdf", dpi=300)
    plt.close()


def plot_embeddings(dataset_name, best_smacof, best_sgd, out_dir):
    """
    Side-by-side scatter plot of the best embeddings found.
    """
    try:
        emb_smacof = np.load(out_dir / "best_embedding_smacof.npy")
        emb_sgd = np.load(out_dir / "best_embedding_sgd.npy")
    except FileNotFoundError:
        print(f"  [Warn] Embeddings not found for {dataset_name}. Skipping plot.")
        return

    if (out_dir / "y_labels.npy").exists():
        y = np.load(out_dir / "y_labels.npy")
        cmap = "Spectral"
    else:
        y = None
        cmap = None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    def _scatter(ax, X, title, stress, time):
        if y is not None:
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=10, alpha=0.6)
        else:
            ax.scatter(X[:, 0], X[:, 1], c='blue', s=10, alpha=0.6)
        
        ax.set_title(f"{title}\nStress: {stress:.2e} | Time: {time:.2f}s")
        ax.axis('off')

    _scatter(ax1, emb_smacof, "SMACOF", best_smacof['stress'], best_smacof['time'])
    _scatter(ax2, emb_sgd, "SGD-MDS", best_sgd['stress'], best_sgd['time'])
    
    plt.suptitle(f"Visual Comparison: {dataset_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_embeddings.pdf", dpi=300)
    plt.close()

def plot_stability_single(dataset_name, stability_data, out_dir):
    """
    Creates a single box plot for the current dataset.
    """
    if not stability_data:
        return
        
    df = pd.DataFrame(stability_data)
    
    plt.figure(figsize=(6, 5))
    
    # Box + Strip Plot
    sns.boxplot(x="algo", y="stress", hue="algo", data=df, width=0.5, 
                palette={"SMACOF": "#dddddd", "SGD-MDS": "#ff9999"}, 
                showfliers=False, legend=False)
    
    sns.stripplot(x="algo", y="stress", data=df, 
                  color="black", alpha=0.6, jitter=True)
    
    # Title with Stats
    smacof_std = df[df["algo"] == "SMACOF"]["stress"].std()
    sgd_std = df[df["algo"] == "SGD-MDS"]["stress"].std()
    
    plt.title(f"{dataset_name}: Stability Analysis\n(Std: SMACOF={smacof_std:.2e} | SGD={sgd_std:.2e})")
    plt.xlabel("")
    plt.ylabel("Final Stress")
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_stability.pdf", dpi=300)
    plt.close()

def plot_time_summary(best_runs_by_dataset, out_dir: Path, cols: int = 3):
    datasets = sorted([k for k in best_runs_by_dataset.keys() if k != "s_curve"])
    n = len(datasets)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 4.8 * rows), sharey=False)
    axes = np.array(axes).reshape(rows, cols)

    legend_handles = None
    legend_labels = None

    for idx, dataset in enumerate(datasets):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        best = best_runs_by_dataset[dataset]
        sm = best.get("SMACOF")
        sg = best.get("SGD-MDS")
        if sm is None or sg is None:
            ax.set_axis_off()
            continue

        sm_hist = np.array(sm["history"])
        sg_hist = np.array(sg["history"])

        # --- global min across the two displayed runs (for this dataset) ---
        y_min = np.inf
        if sm_hist.size:
            y_min = min(y_min, np.nanmin(sm_hist[:, 1]))
        if sg_hist.size:
            y_min = min(y_min, np.nanmin(sg_hist[:, 1]))

        # --- plot with thinner lines ---
        ax.plot(sm_hist[:, 0], sm_hist[:, 1],
                label="SMACOF (best)",
                color="black", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.plot(sg_hist[:, 0], sg_hist[:, 1],
                label="SGD-MDS (best)",
                color="#d62728", linestyle="-", linewidth=1.2, alpha=0.9)

        # horizontal line at best stress
        if np.isfinite(y_min):
            ax.axhline(y=y_min, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)

        ax.set_title(dataset)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Stress")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.2)

        # Major ticks at 1, 2, 5 × 10^k (these will be labeled)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))

        # Minor ticks for the rest (unlabeled)
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    # Turn off unused axes
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].set_axis_off()

    # --- legend: move it UP so it doesn't create a huge bottom margin ---
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.02),   # inside the figure, a bit above bottom
        )

    # no need for a big suptitle if it wastes space; keep it tight if you want it
    # fig.suptitle("Convergence (Stress vs Time) per dataset", y=0.98)

    # Reserve a small band at bottom for legend (instead of a giant empty area)
    fig.tight_layout(rect=[0, 0.06, 1, 1.0])

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "time_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_embedding_summary(best_runs_by_dataset, results_dir: Path, out_dir: Path):
    """
    Global figure:
      - 2 datasets per row
      - each dataset: SMACOF vs SGD-MDS (2 subplots)
      - total grid: rows=ceil(n_datasets/2), cols=4
    """
    datasets = sorted([k for k in best_runs_by_dataset.keys() if k != "s_curve"])
    n = len(datasets)
    if n == 0:
        print("No datasets to plot.")
        return

    rows = math.ceil(n / 2)
    cols = 4  # (SMACOF, SGD) x 2 datasets per row

    # Keep overall width similar to before; halve row height
    fig_w = 12.5
    row_h = 1.9  # ~half of your previous 3.8
    fig_h = row_h * rows

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(fig_w, fig_h),
        sharex=False, sharey=False
    )

    # Normalize axes shape
    if rows == 1:
        axes = np.array([axes])

    def _style_cell(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.9)
        ax.set_facecolor("#fbfbfb")

    def _scatter(ax, X, y=None):
        # Slightly smaller points to fit dense plots in smaller panels
        if y is not None:
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap="Spectral", s=3.5, alpha=0.65, rasterized=True)
        else:
            ax.scatter(X[:, 0], X[:, 1], s=3.5, alpha=0.65, rasterized=True)

    # Column headers (once): two identical header pairs
    axes[0, 0].set_title("SMACOF", fontsize=11, fontweight="bold", pad=6)
    axes[0, 1].set_title("SGD-MDS", fontsize=11, fontweight="bold", pad=6)
    axes[0, 2].set_title("SMACOF", fontsize=11, fontweight="bold", pad=6)
    axes[0, 3].set_title("SGD-MDS", fontsize=11, fontweight="bold", pad=6)

    # We'll add dataset labels *centered above each pair* via fig.text after layout is known
    pending_pair_labels = []  # list of (ax_left, ax_right, dataset_name)

    for idx, dataset in enumerate(datasets):
        r = idx // 2
        block = idx % 2         # 0 = left dataset in row, 1 = right dataset in row
        c0 = block * 2          # start col for that dataset pair

        ax_smacof = axes[r, c0]
        ax_sgd = axes[r, c0 + 1]

        _style_cell(ax_smacof)
        _style_cell(ax_sgd)

        ds_dir = results_dir / dataset

        # Load embeddings
        try:
            emb_smacof = np.load(ds_dir / "best_embedding_smacof.npy")
            emb_sgd = np.load(ds_dir / "best_embedding_sgd.npy")
        except FileNotFoundError:
            ax_smacof.text(0.5, 0.5, "missing embedding", ha="center", va="center",
                           transform=ax_smacof.transAxes, fontsize=8)
            ax_sgd.text(0.5, 0.5, "missing embedding", ha="center", va="center",
                        transform=ax_sgd.transAxes, fontsize=8)
            pending_pair_labels.append((ax_smacof, ax_sgd, dataset))
            continue

        y = None
        if (ds_dir / "y_labels.npy").exists():
            y = np.load(ds_dir / "y_labels.npy")

        best = best_runs_by_dataset[dataset]
        sm = best["SMACOF"]
        sg = best["SGD-MDS"]

        _scatter(ax_smacof, emb_smacof, y=y)
        _scatter(ax_sgd, emb_sgd, y=y)

        # Captions UNDER each subplot (smaller font)
        ax_smacof.text(
            0.5, -0.12,
            f"stress={sm['stress']:.2e} | time={sm['time']:.2f}s",
            transform=ax_smacof.transAxes,
            ha="center", va="top", fontsize=7.5
        )
        ax_sgd.text(
            0.5, -0.12,
            f"stress={sg['stress']:.2e} | time={sg['time']:.2f}s",
            transform=ax_sgd.transAxes,
            ha="center", va="top", fontsize=7.5
        )

        pending_pair_labels.append((ax_smacof, ax_sgd, dataset))

    # Turn off any unused axes (happens if odd number of datasets)
    if n % 2 == 1:
        last_row = (n - 1) // 2
        # Right-side pair is empty
        for cc in (2, 3):
            axes[last_row, cc].set_axis_off()

    # Layout: leave room for under-plot captions; no suptitle to avoid wasted space
    fig.subplots_adjust(
        left=0.045, right=0.995,
        top=0.98, bottom=0.03,
        wspace=0.06, hspace=0.70
    )

    # Now place dataset names centered above each (SMACOF, SGD) pair
    fig.canvas.draw()  # ensures positions are known
    for ax_l, ax_r, name in pending_pair_labels:
        if not ax_l.axison or not ax_r.axison:
            continue
        bb_l = ax_l.get_position()
        bb_r = ax_r.get_position()
        x_mid = 0.5 * (bb_l.x0 + bb_r.x1)
        y_top = max(bb_l.y1, bb_r.y1) + 0.006
        fig.text(x_mid, y_top, name, ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "embedding_comparison.pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def plot_stability_summary(all_data, out_dir):
    """
    Creates a faceted box plot: One subplot per dataset showing stress variance.
    """
    if not all_data:
        return

    datasets = sorted([k for k in all_data.keys() if k != "s_curve"])
    n_datasets = len(datasets)
    
    cols = 3
    rows = (n_datasets + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=False)
    
    if n_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        df = pd.DataFrame(all_data[dataset])
        
        sns.boxplot(x="algo", y="stress", hue="algo", data=df, ax=ax, width=0.5, 
                    palette={"SMACOF": "#dddddd", "SGD-MDS": "#ff9999"}, 
                    showfliers=False, legend=False)
        
        sns.stripplot(x="algo", y="stress", data=df, ax=ax, 
                      color="black", alpha=0.6, jitter=True)
        
        smacof_std = df[df["algo"] == "SMACOF"]["stress"].std()
        sgd_std = df[df["algo"] == "SGD-MDS"]["stress"].std()
        
        ax.set_title(f"{dataset}\n(Std: SMACOF={smacof_std:.2e} | SGD={sgd_std:.2e})")
        ax.set_xlabel("")
        ax.set_ylabel("Final Stress")
        ax.grid(True, axis='y', alpha=0.3)
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(out_dir / "stability_comparison.pdf", dpi=300)
    print(f"Saved global stability plot to {out_dir / 'stability_comparison.pdf'}")
    plt.close()


def process_dataset(dataset_dir):
    dataset_name = dataset_dir.name
    res_file = dataset_dir / "results.json"
    if not res_file.exists():
        return None

    with open(res_file, "r") as f:
        data = json.load(f)

    runs = data["runs"]

    best_smacof, best_sgd = None, None
    min_smacof, min_sgd = float("inf"), float("inf")

    stress_data = []
    time_data = []

    for r in runs:
        # stress stability
        stress_data.append({"algo": r["algo"], "stress": r["stress"]})

        # time stability (clock time)
        time_data.append({"algo": r["algo"], "time": r["time"]})

        # best runs
        if r["algo"] == "SMACOF" and r["stress"] < min_smacof:
            min_smacof = r["stress"]
            best_smacof = r
        elif r["algo"] == "SGD-MDS" and r["stress"] < min_sgd:
            min_sgd = r["stress"]
            best_sgd = r

    if best_smacof and best_sgd:
        plot_convergence(dataset_name, best_smacof, best_sgd, dataset_dir)
        plot_embeddings(dataset_name, best_smacof, best_sgd, dataset_dir)
        plot_stability_single(dataset_name, stress_data, dataset_dir)

    return dataset_name, stress_data, time_data, {"SMACOF": best_smacof, "SGD-MDS": best_sgd}

def plot_time_stability_summary(all_time_data, out_dir: Path, cols: int = 3):
    """
    Faceted violin plots: one subplot per dataset.
    X-axis: algo (SMACOF vs SGD-MDS)
    Y-axis: wall-clock time per run (seconds)
    Layout: 3 columns, enough rows (e.g., 6 rows for ~18 datasets)
    """
    if not all_time_data:
        return

    datasets = sorted([k for k in all_time_data.keys() if k != "s_curve"])
    n = len(datasets)
    if n == 0:
        return

    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.0 * rows), sharey=False)
    axes = np.array(axes).reshape(rows, cols)

    for i, dataset in enumerate(datasets):
        r = i // cols
        c = i % cols
        ax = axes[r, c]

        df = pd.DataFrame(all_time_data[dataset])
        if df.empty:
            ax.set_axis_off()
            continue

        # Keep consistent order
        order = ["SMACOF", "SGD-MDS"]
        df["algo"] = pd.Categorical(df["algo"], categories=order, ordered=True)

        # Violin with inner quartiles; slightly trimmed tails
        sns.violinplot(
            x="algo", y="time", data=df, ax=ax,
            order=order,
            inner="quartile",
            cut=0,
            linewidth=1.0,
            palette={"SMACOF": "#dddddd", "SGD-MDS": "#ff9999"},
        )

        # Overlay points (small jitter) to show individual runs
        sns.stripplot(
            x="algo", y="time", data=df, ax=ax,
            order=order,
            color="black",
            alpha=0.55,
            jitter=0.18,
            size=3,
        )

        # Helpful title stats
        sm_med = df[df["algo"] == "SMACOF"]["time"].median()
        sg_med = df[df["algo"] == "SGD-MDS"]["time"].median()
        ax.set_title(f"{dataset}\n(median: SM={sm_med:.2f}s | SGD={sg_med:.2f}s)", fontsize=10)

        ax.set_xlabel("")
        ax.set_ylabel("Time (s)")
        ax.grid(True, axis="y", alpha=0.25)

    # turn off unused axes
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].set_axis_off()

    fig.tight_layout()
    fig.savefig(out_dir / "time_stability_comparison.pdf", dpi=300)
    plt.close(fig)

    print(f"Saved global time stability plot to {out_dir / 'time_stability_comparison.pdf'}")

def main():
    if not RESULTS_DIR.exists():
        print("No results found.")
        return

    all_stress = {}
    all_time = {}
    best_runs_by_dataset = {}

    for d in RESULTS_DIR.iterdir():
        if not d.is_dir():
            continue
        out = process_dataset(d)
        if out is None:
            continue
        dataset_name, stress_data, time_data, best_runs = out
        all_stress[dataset_name] = stress_data
        all_time[dataset_name] = time_data
        best_runs_by_dataset[dataset_name] = best_runs

    if all_stress:
        plot_stability_summary(all_stress, RESULTS_DIR)

    if all_time:
        plot_time_stability_summary(all_time, RESULTS_DIR)   # NEW

    if best_runs_by_dataset:
        plot_time_summary(best_runs_by_dataset, RESULTS_DIR, cols=3)
        plot_embedding_summary(best_runs_by_dataset, RESULTS_DIR, RESULTS_DIR)


if __name__ == "__main__":
    main()