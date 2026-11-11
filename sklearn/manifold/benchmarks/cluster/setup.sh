#!/usr/bin/env bash
# One-time setup on COMA, and again after pulling .pyx changes. Building
# scikit-learn is too heavy for the login node, so run it in an allocation on
# the nodes the experiments use:
#
#   salloc -p compute --exclude=gpu-nvidia-h100,gpu-nvidia-h200-1-studvm-[1-3] -c 16 --mem=32G -t 02:00:00
#   bash sklearn/manifold/benchmarks/cluster/setup.sh [git ref]
#
# Without a ref it builds whatever is checked out. The reference results for
# compare_results.py come from the laptop (from its repo root, over eduVPN):
#
#   rsync -avP --exclude=.git sklearn/manifold/benchmarks/results/ \
#       login.coma-cluster.de:nobackup/manifold-results-reference/
set -euo pipefail
REPO="${SGDMDS_REPO:-$HOME/scikit-learn}"
DATASETS="${SGDMDS_DATASETS:-$HOME/nobackup/manifold-datasets}"
CONDA="${SGDMDS_CONDA:-$HOME/miniforge3}"
LINK="$REPO/sklearn/manifold/benchmarks/datasets"

if [ -n "${1:-}" ]; then
    git -C "$REPO" fetch --tags origin
    git -C "$REPO" checkout --detach "$1"
fi
echo "Building $(git -C "$REPO" log -1 --format='%h %s')"

# Keep an existing dataset link; only create one if there is none.
if [ ! -e "$LINK" ]; then
    ln -s "$DATASETS" "$LINK"
fi
[ -f "$LINK/seismic/X.npy" ] || { echo "No datasets at $LINK"; exit 1; }

# Miniforge, not Anaconda's installer (the cluster's policy).
source "$CONDA/etc/profile.d/conda.sh"
ENV_FILE="$REPO/sklearn/manifold/benchmarks/environment.yml"
if conda env list | grep -q '^sklearn-dev '; then
    conda env update -n sklearn-dev -f "$ENV_FILE"
else
    conda env create -f "$ENV_FILE"
fi
conda activate sklearn-dev

(cd "$REPO" && pip install -e . --no-build-isolation)

python -c "from sklearn.manifold._sgd_mds_cython import run_sgd_epoch_blocked; \
from sklearn.manifold._squad_mds_cython import run_squad_iteration; \
import sklearn; print('build ok', sklearn.__version__)"
python -m pytest "$REPO/sklearn/manifold/tests/test_sgd_mds.py" -q

SUMS="$HOME/nobackup/manifold-datasets.sha256"
(cd "$LINK" && find -L . -name X.npy -print0 | sort -z | xargs -0 sha256sum) > "$SUMS"
echo "Dataset checksums: $SUMS (compare with the laptop's)"
