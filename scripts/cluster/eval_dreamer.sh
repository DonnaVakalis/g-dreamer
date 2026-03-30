#!/bin/bash
#SBATCH --job-name=gdreamer-eval
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-24

# 5 training checkpoints x 5 eval seeds = 25 array tasks
# task_id = ckpt_idx * N_EVAL_SEEDS + seed_idx
#
# Usage:
#   sbatch eval_dreamer.sh                              # all 25 combinations
#   sbatch --array=0-4 eval_dreamer.sh                  # first checkpoint, all eval seeds
#   PATTERN=toy_consensus_train_dense__baseline__ sbatch eval_dreamer.sh
#   LOGDIR=experiments/runs/toy_consensus_train_dense__baseline__20260327_181955 sbatch --array=0-4 eval_dreamer.sh

set -euo pipefail

REPO="/network/scratch/d/$USER/g_dreamer"
cd "$REPO"

# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gdreamer

export PATH="$HOME/.local/bin:$PATH"

N_TRAIN_SEEDS="${N_TRAIN_SEEDS:-5}"
N_EVAL_SEEDS="${N_EVAL_SEEDS:-5}"
EVAL_SEED_BASE="${EVAL_SEED_BASE:-100}"
EPISODES="${EPISODES:-20}"
SCENARIO="${SCENARIO:-eval_ring_dense}"
PATTERN="${PATTERN:-toy_consensus_train_dense__baseline__}"

CKPT_IDX=$(( SLURM_ARRAY_TASK_ID / N_EVAL_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_EVAL_SEEDS ))
EVAL_SEED=$(( EVAL_SEED_BASE + SEED_IDX ))

if [ -n "${LOGDIR:-}" ]; then
    RUN_DIR="$LOGDIR"
else
    mapfile -t LOGDIRS < <(ls -d experiments/runs/${PATTERN}* 2>/dev/null | sort)
    if [ ${#LOGDIRS[@]} -eq 0 ]; then
        echo "No logdirs found matching experiments/runs/${PATTERN}*"
        exit 1
    fi
    if [ "$CKPT_IDX" -ge "${#LOGDIRS[@]}" ]; then
        echo "Checkpoint index $CKPT_IDX out of range (only ${#LOGDIRS[@]} logdirs found)"
        exit 1
    fi
    RUN_DIR="${LOGDIRS[$CKPT_IDX]}"
fi

echo "Job $SLURM_JOB_ID  array=$SLURM_ARRAY_TASK_ID  ckpt_idx=$CKPT_IDX  eval_seed=$EVAL_SEED"
echo "Evaluating: $RUN_DIR"
echo "GPU: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null || echo none)"

poetry run python -m dgr.eval \
    --logdir "$RUN_DIR" \
    --episodes "$EPISODES" \
    --scenario "$SCENARIO" \
    --seed "$EVAL_SEED" \
    --wandb

echo "Eval done."
