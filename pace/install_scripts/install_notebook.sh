#!/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "$BASH_SOURCE[0]")/" && pwd)

cd "$SCRIPT_DIR"
source ../../mujoco_playground/.venv/bin/activate
uv pip install ipykernel
python -m ipykernel install --user --name Mujoco
