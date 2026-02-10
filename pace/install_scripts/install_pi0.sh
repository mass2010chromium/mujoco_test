SCRIPT_DIR=$(cd -- "$(dirname -- "$BASH_SOURCE[0]")/" && pwd)

cd "$SCRIPT_DIR/.."
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
git apply ../openpi_subtask.diff

source ../../mujoco_playground/.venv/bin/activate
UV_PIP_INSTALL () {
    GIT_LFS_SKIP_SMUDGE=1 uv pip install "$@"
}

GIT_LFS_SKIP_SMUDGE=1 uv sync
UV_PIP_INSTALL pytest
UV_PIP_INSTALL -e .

unset -f UV_PIP_INSTALL
