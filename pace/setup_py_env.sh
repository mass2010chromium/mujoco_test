SCRIPT_DIR=$(cd -- "$(dirname -- "$BASH_SOURCE[0]")/" && pwd)

cd "$SCRIPT_DIR"

bash install_uv.sh
source ~/.local/bin/env

cd ../

sbatch -W pace/install_py_packages.sh
