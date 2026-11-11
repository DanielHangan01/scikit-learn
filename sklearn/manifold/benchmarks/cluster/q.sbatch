#!/bin/bash
# Experiment Q: one array task per synthetic dataset (3).
#SBATCH -J sgd-q
#SBATCH -p compute
#SBATCH --exclude=gpu-nvidia-h100,gpu-nvidia-h200-1-studvm-[1-3]
#SBATCH -a 0-2%2
#SBATCH -c 8
#SBATCH --hint=nomultithread
# 8 cores per job, about one 8-core chiplet: the fit uses one, scoring the rest,
# and the others keep neighbours off the chiplet's L3 cache and memory link.
#SBATCH --mem=8G
#SBATCH -t 10:00:00
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
NAME=$(python -c "import run_experiment_q as Q; print(list(Q.DATASETS)[$SLURM_ARRAY_TASK_ID])")
srun --cpu-bind=cores python run_experiment_q.py "$NAME"
