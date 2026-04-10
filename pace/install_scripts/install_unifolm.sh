#! /usr/bin/env bash

# Installing unifolm world model

SCRIPT_DIR=$(cd -- "$(dirname -- "$BASH_SOURCE[0]")/" && pwd)

cd "$SCRIPT_DIR/../../unifolm-world-model-action"
git submodule init
git submodule update --recursive
git apply '../pace/install_scripts/unifolm.patch'

uv pip install -e . --no-deps
uv pip install gradio decord xformers libclang scikit-learn pytorch-lightning kornia torch==2.7.1 open_clip_torch==2.12.0

cd external/dlimp
uv pip install -e . --no-deps
