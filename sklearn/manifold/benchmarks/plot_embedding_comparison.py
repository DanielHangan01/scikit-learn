"""Figure 1 of the paper: SMACOF vs SGD-MDS embeddings, one row per dataset.

Rebuilds the qualitative comparison from the saved Experiment A artefacts
(`best_embedding_{smacof,sgd}.npy` + `y_labels.npy`), rather than stacking the
per-dataset `comparison_embeddings.pdf` files -- those carry a large title and a
2:1 aspect ratio each, which stacks into a very tall figure with most of its
area whitespace.

Stress and time are read from `results.json`, taking each algorithm's own
best (lowest-stress) run, which is the run whose embedding was saved.

Usage:
    python plot_embedding_comparison.py
    python plot_embedding_comparison.py coil20 fashion_mnist spambase
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(__file__).parent / "results" / "experiment_a" / "sgd_30_epochs"
OUT = Path(__file__).parents[3] / "SGD for MDS" / "figures"

DEFAULT = ["coil20", "fashion_mnist", "spambase"]

# Panel height / width. Below 1 gives the short, wide panels of the original
# single-dataset figure; the data limits in panel() widen to match, so the
# embedding itself stays undistorted.
BOX_ASPECT = 0.62
PRETTY = {"coil20": "coil20", "fashion_mnist": "fashion-mnist", "spambase": "spambase"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 9, "axes.titlesize": 9,
})


def best_run(runs, algo):
    """The lowest-stress run for `algo` -- the one whose embedding was saved."""
    cand = [r for r in runs if r["algo"].upper().startswith(algo)]
    return min(cand, key=lambda r: r["stress"])


def panel(ax, xy, labels, subtitle):
    if labels is not None and len(np.unique(labels)) > 1:
        c = plt.cm.Spectral(
            (labels - labels.min()) / max(labels.max() - labels.min(), 1))
    else:
        c = "#4d4d4d"
    ax.scatter(xy[:, 0], xy[:, 1], s=1.4, c=c, linewidths=0)
    ax.set_facecolor("#f7f7f7")
    ax.set_xticks([]), ax.set_yticks([])

    # Wide, short panels, with the data limits widened to match so that one
    # data unit is the same length on both axes. An MDS embedding must not be
    # stretched -- a squashed panel would misrepresent the very distances the
    # method optimizes -- so the panel is shortened by adding horizontal margin,
    # not by compressing the point cloud. Nothing is clipped.
    cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
    px, py = np.ptp(xy[:, 0]), np.ptp(xy[:, 1])
    half_y = 0.52 * max(py, px * BOX_ASPECT)
    half_x = half_y / BOX_ASPECT
    ax.set_xlim(cx - half_x, cx + half_x)
    ax.set_ylim(cy - half_y, cy + half_y)
    ax.set_box_aspect(BOX_ASPECT)

    for s in ax.spines.values():
        s.set_linewidth(0.6)
    ax.set_xlabel(subtitle, fontsize=7.5, labelpad=2)


def main(datasets):
    n = len(datasets)
    fig, axes = plt.subplots(n, 2, figsize=(4.6, 1.62 * n))
    axes = np.atleast_2d(axes)

    for row, ds in enumerate(datasets):
        d = SRC / ds
        meta = json.loads((d / "results.json").read_text())
        y = np.load(d / "y_labels.npy") if (d / "y_labels.npy").exists() else None
        if y is not None:
            y = y.astype(float).ravel()

        for col, (algo, tag, emb) in enumerate([
            ("SMACOF", "SMACOF", "best_embedding_smacof.npy"),
            ("SGD", "SGD-MDS", "best_embedding_sgd.npy"),
        ]):
            r = best_run(meta["runs"], algo)
            panel(axes[row, col], np.load(d / emb), y,
                  f"stress={r['stress']:.2e} | time={r['time']:.2f}s")
            if row == 0:
                axes[row, col].set_title(tag, fontweight="bold", pad=6)

        axes[row, 0].set_ylabel(PRETTY.get(ds, ds), fontweight="bold",
                                fontsize=9.5, labelpad=6)

    fig.tight_layout(h_pad=1.6, w_pad=1.0)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"embedding_comparison_grid.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'embedding_comparison_grid.pdf'}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)
