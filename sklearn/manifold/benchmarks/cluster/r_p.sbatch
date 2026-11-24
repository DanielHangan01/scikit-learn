#!/bin/bash
# Experiment R, large suite: s-EFS on P's 8 sources over the ladder, and full
# EFS at N=10,000 (0.4 GB condensed distance matrix; ~250 ns per pair update,
# so 20-40 minutes per fit at one thread), each at 1 and 8 threads. cifar10_raw
# is the longest task: s-EFS computes each distance over D=3,072 on the fly.
# Runs resume per fit, so chain with ./submit.sh r_p.sbatch 2 if it overruns.
#SBATCH -J sgd-r-p
#SBATCH -p compute
#SBATCH --exclude=gpu-nvidia-h100,gpu-nvidia-h200-1-studvm-[1-3]
#SBATCH -a 0-7%4
#SBATCH -c 8
#SBATCH --hint=nomultithread
# 8 cores per job, as in P: the fit uses one, scoring the rest.
#SBATCH --mem=32G
#SBATCH -t 10:00:00
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
NAME=$(python -c "import run_experiment_p as P; print(P.DATASETS[$SLURM_ARRAY_TASK_ID])")
# srun doesn't inherit -c from sbatch: without --cpus-per-task the run gets
# one core, and the 8-thread EFS arms can't start.
srun --cpu-bind=cores --cpus-per-task="$SLURM_CPUS_PER_TASK" python run_experiment_r.py large "$NAME"
