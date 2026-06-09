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
#   sbatch lorenz.sbatch <netID> <experiment_name> [hydra overrides...]
#
# To run two experiments concurrently (A100 + A40):
#   sbatch                      lorenz.sbatch q58r662 lorenz_gaussian_full
#   sbatch --gres=gpu:a40:1     lorenz.sbatch q58r662 lorenz_frozen_enc
#
# The --gres flag on the sbatch command overrides the #SBATCH --gres line above.

set -euo pipefail

NETID=${1:?'Usage: sbatch lorenz.sbatch <netID> <experiment_name>  (e.g. q58r662 lorenz_gaussian_full)'}
EXP=${2:?'Usage: sbatch lorenz.sbatch <netID> <experiment_name>  (e.g. q58r662 lorenz_gaussian_full)'}
shift 2  # remove netID and EXP so remaining args forward as Hydra overrides

cd "$SLURM_SUBMIT_DIR"

mkdir -p runs/lorenz

module load Python/3.13.5-GCCcore-14.3.0 CUDA/13.0.0 tools/uv/0.9.22
source /home/${NETID}/diffusion-driven-state-space-models/.venv/bin/activate

exec python -m ddssm.app experiment=${EXP} \
    "hydra.run.dir=outputs/${EXP}/\${now:%Y-%m-%d_%H-%M-%S}" \
    "$@"