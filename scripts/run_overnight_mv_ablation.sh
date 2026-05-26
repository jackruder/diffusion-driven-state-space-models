#!/usr/bin/env bash
# Overnight ablation: 18 cells × 25 trials on the MV dataset (latent_dim=4).
# Concurrency = n_jobs=6 inside each cell's Optuna study; cells run
# sequentially because two concurrent studies would split the 24 GB VRAM
# and starve each other.
#
# Estimated wall-clock: ~11-14 h.
# Estimated VRAM peak: ~20 GB (6 × ~3.4 GB).
#
# Run:
#   bash scripts/run_overnight_mv_ablation.sh
# Or detach to keep it running after logout:
#   nohup bash scripts/run_overnight_mv_ablation.sh > runs/overnight.log 2>&1 &
#
# After it finishes, aggregate + plot:
#   .venv/bin/python -m experiments.init_centering.report all \
#       --sweeps-root runs/sweeps --optuna-dir runs/optuna \
#       --study-prefix overnight_$(date +%Y%m%d) \
#       --out runs/report/overnight_$(date +%Y%m%d)

set -euo pipefail

STAMP=${STAMP:-$(date +%Y%m%d)}
SBATCH_DIR=runs/sbatch/overnight_${STAMP}
N_TRIALS=${N_TRIALS:-25}
N_JOBS=${N_JOBS:-6}

mkdir -p runs/optuna runs/sweeps

# 1. Render the 18 MV-only sbatch scripts (with n_jobs=6 baked in).
.venv/bin/python -m experiments.init_centering.launch_ablation_tiny \
    --write-dir "$SBATCH_DIR" \
    --study-prefix overnight_${STAMP} \
    --n-trials $N_TRIALS \
    --n-jobs $N_JOBS \
    --datasets mv

# 2. Extract the actual ``python -m ddssm.app ...`` lines from each sbatch
#    script and run them sequentially on the local GPU. Skips the
#    SLURM #SBATCH directives entirely.
echo
echo "=== Starting MV ablation: 18 cells × $N_TRIALS trials, n_jobs=$N_JOBS ==="
echo "=== Logs land under each sbatch script's hydra.sweep.dir ==="
echo

cell_idx=0
for f in "$SBATCH_DIR"/*__mv.sbatch; do
    cell_idx=$((cell_idx + 1))
    cmd=$(grep '^exec python' "$f" | sed 's/^exec //')
    job_name=$(basename "$f" .sbatch)
    echo
    echo "--- [$cell_idx/18] $job_name ---"
    echo "+ $cmd"
    echo "--- started: $(date -Iseconds) ---"
    eval "$cmd"
    echo "--- finished: $(date -Iseconds) ---"
done

echo
echo "=== All 18 cells done. Aggregate + plot with:"
echo ".venv/bin/python -m experiments.init_centering.report all \\"
echo "    --sweeps-root runs/sweeps --optuna-dir runs/optuna \\"
echo "    --study-prefix overnight_${STAMP} \\"
echo "    --out runs/report/overnight_${STAMP}"
