#!/bin/bash
#SBATCH --job-name=gdreamer-eval
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

# Usage:
#   sbatch eval_dreamer.sh                              # eval all 5 seeds of toy_consensus_train_dense
#   sbatch --array=3 eval_dreamer.sh                    # single seed
#   PATTERN=toy_consensus_train_dense__baseline__ sbatch eval_dreamer.sh
#   LOGDIR=experiments/runs/toy_consensus_train_dense__baseline__20260327_181955 sbatch --array=0 eval_dreamer.sh
#   EVAL_SEED=200 sbatch eval_dreamer.sh                # override eval seed (default 100, never use 0-4)

set -euo pipefail

REPO="/network/scratch/d/$USER/g_dreamer"
cd "$REPO"

# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gdreamer

export PATH="$HOME/.local/bin:$PATH"

EPISODES="${EPISODES:-20}"
SCENARIO="${SCENARIO:-eval_ring_dense}"
EVAL_SEED="${EVAL_SEED:-100}"
PATTERN="${PATTERN:-toy_consensus_train_dense__baseline__}"

if [ -n "${LOGDIR:-}" ]; then
    RUN_DIR="$LOGDIR"
else
    mapfile -t LOGDIRS < <(ls -d experiments/runs/${PATTERN}* 2>/dev/null | sort)
    if [ ${#LOGDIRS[@]} -eq 0 ]; then
        echo "No logdirs found matching experiments/runs/${PATTERN}*"
        exit 1
    fi
    if [ "$SLURM_ARRAY_TASK_ID" -ge "${#LOGDIRS[@]}" ]; then
        echo "Array task $SLURM_ARRAY_TASK_ID out of range (only ${#LOGDIRS[@]} logdirs found)"
        exit 1
    fi
    RUN_DIR="${LOGDIRS[$SLURM_ARRAY_TASK_ID]}"
fi

echo "Job $SLURM_JOB_ID  array=$SLURM_ARRAY_TASK_ID"
echo "Evaluating: $RUN_DIR"
echo "GPU: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null || echo none)"

poetry run python -m dgr.eval \
    --logdir "$RUN_DIR" \
    --episodes "$EPISODES" \
    --scenario "$SCENARIO" \
    --seed "$EVAL_SEED" \
    --wandb

echo "Eval done."
