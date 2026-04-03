#!/bin/bash
#SBATCH --job-name=gdreamer-train
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

# Usage:
#   sbatch train_dreamer.sh                          # train_ring_dense, seeds 0-4
#   sbatch --array=0 train_dreamer.sh                # single seed (smoke test of real run)
#   ENV=toy_consensus_train_sparse_hidden_smooth_aligned sbatch train_dreamer.sh
#   AGENT=graph_encoder sbatch --array=0 train_dreamer.sh

set -euo pipefail

REPO="/network/scratch/d/$USER/g_dreamer"
cd "$REPO"

# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gdreamer

export PATH="$HOME/.local/bin:$PATH"

SEED=$SLURM_ARRAY_TASK_ID
AGENT="${AGENT:-baseline}"
ENV="${ENV:-toy_consensus_train_dense}"
STEPS="${STEPS:-1000000}"

echo "Job $SLURM_JOB_ID  array=$SLURM_ARRAY_TASK_ID  agent=$AGENT  env=$ENV  steps=$STEPS  seed=$SEED"
echo "GPU: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null || echo none)"

poetry run python -m dgr.train \
    agent="$AGENT" \
    env="$ENV" \
    steps="$STEPS" \
    --seed "$SEED"

echo "Training done."
