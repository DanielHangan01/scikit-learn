#!/usr/bin/env bash
# The pilot: a few datasets at full size before the whole campaign. Its results
# are kept -- the full runs later skip what the pilot finished -- so nothing is
# lost by running it first. Inspect them (and the canary) before launching the
# rest with submit.sh.
#
#   O: coil20 (small) and epileptic (the largest N)
#   P: california_housing (the smallest source, two rungs)
#   Q: torus
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
index() {  # array index of a dataset in a runner's list
    (source env.sh >/dev/null && python -c "import $1 as m; d=list(m.DATASETS); print(d.index('$2'))")
}
O_IDS="$(index run_experiment_o coil20),$(index run_experiment_o epileptic)"
P_ID="$(index run_experiment_p california_housing)"
Q_ID="$(index run_experiment_q torus)"
sbatch --array="$O_IDS" o.sbatch
sbatch --array="$P_ID" p.sbatch
sbatch --array="$Q_ID" q.sbatch
