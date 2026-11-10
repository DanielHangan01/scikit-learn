#!/bin/bash
# Experiment O: one array task per dataset (17). %8 caps how many share the
# nodes at once; set it from the calibration.
#SBATCH -J sgd-o
#SBATCH -p compute
#SBATCH --exclude=gpu-nvidia-h100
#SBATCH -a 0-16%8
#SBATCH -c 2
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
NAME=$(python -c "import run_experiment_o as O; print(O.DATASETS[$SLURM_ARRAY_TASK_ID])")
srun --cpu-bind=cores python run_experiment_o.py "$NAME"
