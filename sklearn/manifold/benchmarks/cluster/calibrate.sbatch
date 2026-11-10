#!/bin/bash
# The kernel microbenchmark (spec section 6) on a whole node, then the same
# workload with 8, 16 and 32 copies running at once, to choose how many jobs
# may share a node. The cluster docs allow --exclusive for benchmarks; it
# blocks the node's GPUs too, so run it at a quiet time.
#SBATCH -J sgd-calibrate
#SBATCH -p compute
#SBATCH --exclude=gpu-nvidia-h100
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT="$HOME/sgd-mds-calibration-$SLURM_JOB_ID"

SGDMDS_RESULTS_ROOT="$ROOT/alone" python bench_blocked_kernel.py

for copies in 8 16 32; do
    for c in $(seq 1 "$copies"); do
        SGDMDS_RESULTS_ROOT="$ROOT/x$copies/$c" taskset -c "$((c - 1))" \
            python bench_blocked_kernel.py --dims 8,784 --reps 3 > /dev/null &
    done
    wait
done
python - "$ROOT" <<'PY'
import sys, pathlib, pandas as pd
root = pathlib.Path(sys.argv[1])
alone = pd.read_csv(root / "alone/bench_kernels/results.csv")
alone = alone[alone.D.isin([8, 784])].set_index(["arm", "D"]).ns_per_pair
for d in sorted(root.glob("x*")):
    runs = pd.concat(pd.read_csv(f) for f in d.glob("*/bench_kernels/results.csv"))
    slow = runs.groupby(["arm", "D"]).ns_per_pair.median() / alone
    print(f"{d.name:>4} copies: median slowdown {slow.median():.2f}x, worst {slow.max():.2f}x")
PY
