#!/bin/bash
# Experiment R, small suite: EFS and s-EFS on O's 17 datasets, one array task
# each, at 1 and 8 threads. Full EFS costs 130-250 ns per pair update at one
# thread (SGDMDS's full sweep: 26-110 ns) and runs 100-200 iterations, so
# epileptic is the longest task, about two hours.
#SBATCH -J sgd-r-o
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
srun --cpu-bind=cores python run_experiment_r.py small "$NAME"
