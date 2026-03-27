#!/bin/bash
# Run once on the login node to create the conda env and install the project.
# Usage: bash scripts/cluster/setup_env.sh
set -euo pipefail

ENV_NAME="gdreamer"
REPO="/network/scratch/d/$USER/g_dreamer"

module load miniconda/3

if conda env list | grep -q "^$ENV_NAME "; then
    echo "Conda env '$ENV_NAME' already exists — skipping create."
else
    conda create -y -n "$ENV_NAME" python=3.11.9
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

pip install --quiet poetry

cd "$REPO"
poetry install --with dev,upstream

echo "Setup complete. Python: $(python --version)"
