#!/usr/bin/env bash
# Queue a job script several times in a row, each starting when the previous one
# ends. Jobs are capped at 10 hours; every runner resumes per fit, so each link of
# the chain continues where the last one stopped, and finished datasets are
# skipped at once.
#
#   ./submit.sh p.sbatch 4      # four 10-hour runs of Experiment P, one after another
set -euo pipefail
SCRIPT="${1:?usage: submit.sh <job.sbatch> [number of runs]}"
RUNS="${2:-1}"
cd "$(dirname "$0")"
mkdir -p logs
prev=""
for i in $(seq 1 "$RUNS"); do
    dep=${prev:+--dependency=afterany:$prev}
    prev=$(sbatch --parsable $dep "$SCRIPT")
    echo "run $i: job $prev${dep:+ (after the previous one)}"
done
