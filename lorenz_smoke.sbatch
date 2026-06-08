#!/bin/bash
#SBATCH --job-name=ddssm-lorenz_smoke
#SBATCH --partition=gpupriority
#SBATCH --account=priority-michaelwojnowicz
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --output=runs/lorenz_smoke/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load Python/3.13.5-GCCcore-14.3.0 CUDA/13.0.0 tools/uv/0.9.22
source /home/z89p425/diffusion-driven-state-space-models/.venv/bin/activate

exec python -m ddssm.app experiment=lorenz_smoke "$@"