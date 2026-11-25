#!/bin/bash
# Experiment R, epochs suite: full SGDMDS at 30, 60, 120 and 200 epochs on O's
# 17 datasets, one array task each. One thread per fit; epileptic is the longest
# task, about 2.5 hours (200 epochs are ~6.7x the 30-epoch fit).
#SBATCH -J sgd-r-epochs
#SBATCH -p compute
#SBATCH --exclude=gpu-nvidia-h100,gpu-nvidia-h200-1-studvm-[1-3]
#SBATCH -a 0-16%4
#SBATCH -c 8
#SBATCH --hint=nomultithread
# 8 cores per job, as in O: the fit uses one, scoring the rest.
#SBATCH --mem=8G
#SBATCH -t 10:00:00
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
NAME=$(python -c "import run_experiment_o as O; print(O.DATASETS[$SLURM_ARRAY_TASK_ID])")
# srun doesn't inherit -c from sbatch: without --cpus-per-task the run gets
# one core.
srun --cpu-bind=cores --cpus-per-task="$SLURM_CPUS_PER_TASK" python run_experiment_r.py epochs "$NAME"
