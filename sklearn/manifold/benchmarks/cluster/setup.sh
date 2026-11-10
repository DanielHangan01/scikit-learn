#!/usr/bin/env bash
# One-time setup on COMA. Building scikit-learn is too heavy for the login node,
# so run this inside an allocation on the nodes the experiments will use:
#
#   salloc -p compute --exclude=gpu-nvidia-h100 -c 16 --mem=32G -t 02:00:00
#   bash setup.sh <git ref>        # e.g. a tag made for the campaign
#
# Before it, copy the data from the laptop (from the repo root, over eduVPN):
#
#   rsync -avP --exclude=.git sklearn/manifold/benchmarks/datasets/ \
#       login.coma-cluster.de:nobackup/sgd-mds/datasets/
#   rsync -avP --exclude=.git sklearn/manifold/benchmarks/results/ \
#       login.coma-cluster.de:nobackup/sgd-mds/results_reference/
set -euo pipefail
REF="${1:?usage: setup.sh <git ref to check out>}"
REPO="${SGDMDS_REPO:-$HOME/sgd-mds/scikit-learn}"
DATA="${SGDMDS_DATA:-$HOME/nobackup/sgd-mds}"
CONDA="${SGDMDS_CONDA:-$HOME/miniforge3}"

# Miniforge, not Anaconda's installer (the cluster's policy).
if [ ! -d "$CONDA" ]; then
    curl -fsSLo /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/miniforge.sh -b -p "$CONDA"
fi
source "$CONDA/etc/profile.d/conda.sh"

if [ ! -d "$REPO" ]; then
    git clone https://github.com/DanielHangan01/scikit-learn.git "$REPO"
fi
git -C "$REPO" fetch --tags origin
git -C "$REPO" checkout --detach "$REF"

ENV_FILE="$REPO/sklearn/manifold/benchmarks/environment.yml"
if conda env list | grep -q '^sklearn-dev '; then
    conda env update -n sklearn-dev -f "$ENV_FILE"
else
    conda env create -f "$ENV_FILE"
fi
conda activate sklearn-dev

ln -sfn "$DATA/datasets" "$REPO/sklearn/manifold/benchmarks/datasets"
(cd "$REPO" && pip install -e . --no-build-isolation)

python -c "from sklearn.manifold._sgd_mds_cython import run_sgd_epoch_blocked; \
from sklearn.manifold._squad_mds_cython import run_squad_iteration; print('build ok')"
python -m pytest "$REPO/sklearn/manifold/tests/test_sgd_mds.py" -q
echo "Checksums, to compare with the laptop's:"
(cd "$DATA/datasets" && find -L . -name X.npy -print0 | sort -z | xargs -0 sha256sum) \
    > "$DATA/datasets.sha256"
echo "  $DATA/datasets.sha256"
