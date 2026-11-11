#!/bin/bash
# Minutes-long end-to-end test of O, P, Q and the microbenchmark, into a scratch
# root. Also starts several Python processes at once, to check that concurrent
# imports of the editable build don't collide.
#SBATCH -J sgd-smoke
#SBATCH -p compute
#SBATCH --exclude=gpu-nvidia-h100,gpu-nvidia-h200-1-studvm-[1-3]
#SBATCH -c 4
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
export SGDMDS_RESULTS_ROOT="$HOME/sgd-mds-smoke"
rm -rf "$SGDMDS_RESULTS_ROOT"
for i in 1 2 3 4; do python -c "import sklearn.manifold._sgd_mds" & done; wait
python run_experiment_o.py seismic --seeds 1 --quick
python run_experiment_q.py torus --n 500 --seeds 1
python run_experiment_p.py california_housing --ladder 2000 --seeds 1 --quick \
    --score-pairs 1000000
python bench_blocked_kernel.py --quick
python compare_results.py
