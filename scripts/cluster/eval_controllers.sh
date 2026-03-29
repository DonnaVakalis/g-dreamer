#!/bin/bash
#SBATCH --job-name=gdreamer-ctrl-eval
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4

# One task per eval seed. All controllers run within each task.
#
# Usage:
#   sbatch eval_controllers.sh                          # eval_ring_dense, 5 eval seeds
#   SCENARIO=train_ring_dense sbatch eval_controllers.sh
#   sbatch --array=0 eval_controllers.sh                # single seed (quick check)

set -euo pipefail

REPO="/network/scratch/d/$USER/g_dreamer"
cd "$REPO"

# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gdreamer

export PATH="$HOME/.local/bin:$PATH"

SCENARIO="${SCENARIO:-eval_ring_dense}"
EPISODES="${EPISODES:-20}"
EVAL_SEED_BASE="${EVAL_SEED_BASE:-100}"

EVAL_SEED=$(( EVAL_SEED_BASE + SLURM_ARRAY_TASK_ID ))

echo "Job $SLURM_JOB_ID  array=$SLURM_ARRAY_TASK_ID  scenario=$SCENARIO  episodes=$EPISODES  eval_seed=$EVAL_SEED"

poetry run python scripts/compare_toy_controllers.py \
    "$SCENARIO" \
    --episodes "$EPISODES" \
    --seed "$EVAL_SEED" \
    --wandb

echo "Controller eval done."
