#!/bin/bash
#SBATCH --job-name=gdreamer-smoke
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

set -euo pipefail

REPO="/network/scratch/d/$USER/g_dreamer"
cd "$REPO"

module load miniconda/3
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gdreamer

export PATH="$HOME/.local/bin:$PATH"

echo "Python: $(python --version)"
echo "GPU:"
nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader || echo "no nvidia-smi"

echo "JAX devices:"
poetry run python -c "import jax; print(jax.devices())"

echo "Starting smoke test..."
poetry run python -m dgr.train \
    env=toy_consensus_debug_dense \
    steps=500 \
    --seed 0

echo "Smoke test done."
