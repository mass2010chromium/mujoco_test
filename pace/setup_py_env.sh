#! /bin/env bash
SCRIPT_DIR=$(cd -- "$(dirname -- "$BASH_SOURCE[0]")/" && pwd)

cd "$SCRIPT_DIR"

bash install_uv.sh
source ~/.local/bin/env

cd ../

which sbatch > /dev/null
if [[ $? -eq 0 ]]; then
    echo "Found sbatch, running install with sbatch"
    sbatch -W pace/install_py_packages.sh
else
    echo "No sbatch found, running install without sbatch"
    bash pace/install_py_packages.sh
fi
