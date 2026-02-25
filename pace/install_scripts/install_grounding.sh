SCRIPT_DIR=$(cd -- "$(dirname -- "$BASH_SOURCE[0]")/" && pwd)

cd "$SCRIPT_DIR"
source ../../mujoco_playground/.venv/bin/activate
cd ../../thirdparty/sam3
uv pip install -e .
uv pip install pddlsim
