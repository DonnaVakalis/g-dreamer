#!/bin/bash
#SBATCH --job-name=gdreamer-ctrl-eval
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# Usage:
#   sbatch eval_controllers.sh                          # eval_ring_dense, seed 0, 20 episodes
#   SCENARIO=train_ring_dense sbatch eval_controllers.sh
#   EPISODES=50 SEED=1 sbatch eval_controllers.sh

set -euo pipefail

REPO="/network/scratch/d/$USER/g_dreamer"
cd "$REPO"

# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gdreamer

export PATH="$HOME/.local/bin:$PATH"

SCENARIO="${SCENARIO:-eval_ring_dense}"
EPISODES="${EPISODES:-20}"
EVAL_SEED="${EVAL_SEED:-100}"

echo "Job $SLURM_JOB_ID  scenario=$SCENARIO  episodes=$EPISODES  eval_seed=$EVAL_SEED"

poetry run python scripts/compare_toy_controllers.py \
    "$SCENARIO" \
    --episodes "$EPISODES" \
    --seed "$EVAL_SEED" \
    --wandb

echo "Controller eval done."
