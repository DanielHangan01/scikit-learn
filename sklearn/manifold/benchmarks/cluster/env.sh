# Sourced by every COMA job script. The defaults match the layout on COMA; set
# the variables before submitting to override them.
REPO="${SGDMDS_REPO:-$HOME/scikit-learn}"
CONDA="${SGDMDS_CONDA:-$HOME/miniforge3}"

source "$CONDA/etc/profile.d/conda.sh"
conda activate sklearn-dev

# Results in home: it is snapshotted daily and backed up off-site.
export SGDMDS_RESULTS_ROOT="${SGDMDS_RESULTS_ROOT:-$HOME/sgd-mds-results}"
# The laptop's results/, for compare_results.py.
export SGDMDS_REFERENCE_ROOT="${SGDMDS_REFERENCE_ROOT:-$HOME/nobackup/manifold-results-reference}"

# BLAS threads for scoring; every timed fit is limited to one thread anyway.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

cd "$REPO/sklearn/manifold/benchmarks"
