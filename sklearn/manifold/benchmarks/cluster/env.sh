# Sourced by every COMA job script. Edit the three paths if you set things up
# elsewhere; everything else follows from them.
REPO="${SGDMDS_REPO:-$HOME/sgd-mds/scikit-learn}"
DATA="${SGDMDS_DATA:-$HOME/nobackup/sgd-mds}"
CONDA="${SGDMDS_CONDA:-$HOME/miniforge3}"

source "$CONDA/etc/profile.d/conda.sh"
conda activate sklearn-dev

# Results in home: it is snapshotted daily and backed up off-site.
export SGDMDS_RESULTS_ROOT="${SGDMDS_RESULTS_ROOT:-$HOME/sgd-mds/results_paper}"
export SGDMDS_REFERENCE_ROOT="${SGDMDS_REFERENCE_ROOT:-$DATA/results_reference}"

# BLAS threads for scoring; every timed fit is limited to one thread anyway.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

cd "$REPO/sklearn/manifold/benchmarks"
