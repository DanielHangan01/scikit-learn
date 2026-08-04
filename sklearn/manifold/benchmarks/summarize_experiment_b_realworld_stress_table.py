# summarize_experiment_b_realworld_stress_table.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
RESULTS_DIR = Path(__file__).parent / "results" / "experiment_b" / "scaling_mixed_euclid_vs_lazy"
OUT_DIR = RESULTS_DIR / "_plots_combined"  # or RESULTS_DIR if you prefer

REALWORLD_DATASETS = ["fashion_mnist", "spambase", "seismic"]

ALGO_ORDER = ["SMACOF-euclidean", "SGD-euclidean", "SGD-lazy"]
ALGO_COL_LABELS = {
    "SMACOF-euclidean": "Stress (SMACOF)",
    "SGD-euclidean": "Stress (SGD-Euclidean)",
    "SGD-lazy": "Stress (SGD-lazy)",
}


def _latex_escape_dataset(name: str) -> str:
    # Minimal escaping for LaTeX tables
    return name.replace("_", r"\_")


def load_dataset_results(dataset: str) -> pd.DataFrame:
    path = RESULTS_DIR / dataset / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)

    # normalize / sanity
    if "dataset" not in df.columns:
        df["dataset"] = dataset

    df["dataset"] = df["dataset"].astype(str)
    df["algo"] = df["algo"].astype(str)
    df["stress"] = pd.to_numeric(df["stress"], errors="coerce")

    # Keep only the three algos we care about
    df = df[df["algo"].isin(ALGO_ORDER)].copy()

    # For real-world datasets, there is only one N; but if any N column exists,
    # we just ignore it and average across all runs for that dataset+algo.
    return df


def summarize_realworld_stress(datasets: list[str]) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        df = load_dataset_results(ds)
        if df.empty:
            continue

        g = df.groupby(["dataset", "algo"], dropna=False).agg(
            stress_mean=("stress", "mean"),
            stress_std=("stress", "std"),
            n_runs=("stress", "size"),
        ).reset_index()

        rows.append(g)

    if not rows:
        raise RuntimeError("No real-world Experiment B results found.")

    out = pd.concat(rows, ignore_index=True)

    # Pivot wide: columns = algos
    piv = out.pivot(index="dataset", columns="algo", values="stress_mean").reset_index()

    # Ensure column order
    for algo in ALGO_ORDER:
        if algo not in piv.columns:
            piv[algo] = np.nan
    piv = piv[["dataset"] + ALGO_ORDER]

    return piv


def to_latex_table(piv: pd.DataFrame) -> str:
    # Header
    col_labels = ["Dataset"] + [ALGO_COL_LABELS[a] for a in ALGO_ORDER]

    def f_e(x: float) -> str:
        if pd.isna(x):
            return ""
        return f"{x:.3e}"

    latex = []
    latex.append(r"\begin{table}[!htbp]")
    latex.append(r"\centering")
    latex.append(
        r"\caption{Average final stress over $n_{\text{init}}=10$ runs for SMACOF and SGD-MDS in the "
        r"\texttt{euclidean} and \texttt{lazy} regimes on three real-world datasets.}"
    )
    latex.append(r"\label{tab:expB_realworld_stress}")
    latex.append(r"\vspace{6pt}")
    latex.append(r"\begin{tabular}{lrrr}")
    latex.append(r"\toprule")
    latex.append(" & ".join(col_labels) + r" \\")
    latex.append(r"\midrule")

    for _, row in piv.iterrows():
        ds = _latex_escape_dataset(str(row["dataset"]))
        vals = [f_e(row[a]) for a in ALGO_ORDER]
        latex.append(ds + " & " + " & ".join(vals) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    return "\n".join(latex)


def main():
    piv = summarize_realworld_stress(REALWORLD_DATASETS)
    tex = to_latex_table(piv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "table_experiment_b_realworld_stress.tex"
    out_path.write_text(tex)

    print("Wrote:")
    print(" -", out_path)


if __name__ == "__main__":
    main()
