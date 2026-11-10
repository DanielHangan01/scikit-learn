#!/bin/bash
# Experiment P: one array task per source (8). The full sweep at N=50k needs
# ~84 GB, hence the memory request. cifar10_raw is the longest task.
#SBATCH -J sgd-p
#SBATCH -p compute
#SBATCH --exclude=gpu-nvidia-h100
#SBATCH -a 0-7%8
#SBATCH -c 4
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH -t 10:00:00
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
export SGDMDS_MAX_MEM_GB=120
NAME=$(python -c "import run_experiment_p as P; print(P.DATASETS[$SLURM_ARRAY_TASK_ID])")
srun --cpu-bind=cores python run_experiment_p.py "$NAME"
