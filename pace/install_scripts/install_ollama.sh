#!/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "$BASH_SOURCE[0]")/" && pwd)

cd "$SCRIPT_DIR"
source ../../mujoco_playground/.venv/bin/activate
uv pip install ollama

module load ollama/0.9.0 || (curl -fsSL https://ollama.com/install.sh | sh)

ollama serve
ollama pull gemma3:27b

