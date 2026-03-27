#!/bin/bash
# Run once on the login node to create the conda env and install the project.
# Usage: bash scripts/cluster/setup_env.sh
set -euo pipefail

ENV_NAME="gdreamer"
REPO="/network/scratch/d/$USER/g_dreamer"
CONDA_DIR="$HOME/miniconda3"

# Install personal miniconda if not present (required for libmamba ownership)
if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing miniconda to $CONDA_DIR..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
fi

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Install libmamba solver into base once
if ! conda list -n base | grep -q conda-libmamba-solver; then
    echo "Installing libmamba solver..."
    conda install -y -n base conda-libmamba-solver
fi

if conda env list | grep -q "^$ENV_NAME "; then
    echo "Conda env '$ENV_NAME' already exists — skipping create."
else
    conda create -y -n "$ENV_NAME" python=3.11.9 -c conda-forge --solver=libmamba
fi

conda activate "$ENV_NAME"

pip install --quiet poetry

cd "$REPO"
poetry install --with dev,upstream

echo "Setup complete. Python: $(python --version)"
