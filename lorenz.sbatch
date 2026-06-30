#!/bin/bash
#SBATCH --job-name=ddssm-lorenz
#SBATCH --partition=gpupriority
#SBATCH --account=priority-michaelwojnowicz
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --output=runs/lorenz/slurm-%j.out

# Usage:
#   sbatch lorenz.sbatch <experiment_name> [hydra overrides...]
#
# Defaults come from .claude/cluster.local.env (git-ignored; see .template).
#
# To run two experiments concurrently (A100 + A40):
#   sbatch                      lorenz.sbatch lorenz_gaussian_full
#   sbatch --gres=gpu:a40:1     lorenz.sbatch lorenz_frozen_enc
#
# The --gres flag on the sbatch command overrides the #SBATCH --gres line above.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Source per-user cluster config if present (sets CLUSTER_NETID etc.).
if [ -f ".claude/cluster.local.env" ]; then source ".claude/cluster.local.env"; fi

EXP=${1:?'Usage: sbatch lorenz.sbatch <experiment_name>  (e.g. lorenz_gaussian_full)'}
shift  # remaining args forward as Hydra overrides

mkdir -p runs/lorenz

set +eu; source /etc/profile; module purge
module load Python/3.13.5-GCCcore-14.3.0 CUDA/13.0.0 tools/uv/0.9.22
set -eu
source .venv/bin/activate

exec python -m ddssm.app experiment=${EXP} \
    "hydra.run.dir=outputs/${EXP}/\${now:%Y-%m-%d_%H-%M-%S}" \
    "$@"