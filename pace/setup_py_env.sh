#! /bin/env bash

# Script does three things:
# 1. Install uv package manager
# 2. Create uv virtualenv (in mujoco_playground/.venv)
# 3. Build mujoco/madrona (submodules of this repo)
# 4. Install pi0

SCRIPT_DIR=$(cd -- "$(dirname -- "$BASH_SOURCE[0]")/" && pwd)

# 0. Option extraction: https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
POSITIONAL_ARGS=()
GPU_NAME=rtx_6000

while [[ $# -gt 0 ]]; do
  case $1 in
    -G|--gpu)
      GPU_NAME="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters
# END Option Extraction


# 1. Install uv package manager
cd "$SCRIPT_DIR"
git submodule update --recursive
bash install_scripts/install_uv.sh
source ~/.local/bin/env
# END install uv


# 2. Install all py packages
which sbatch > /dev/null
if [[ $? -eq 0 ]]; then
	# 2.1a. IF slurm exists, populate node lists.
	cd slurm_utils
	bash sinfo.sh
	python extract_nodes.py
	cd ../../
	
	# 2.2a. Run install scripts using slurm
    echo "Found sbatch, running install with sbatch"
    sbatch --gpus="$GPU_NAME" --nodefile="pace/nodes_$GPU_NAME.txt" -W pace/install_scripts/install_py_packages.sh

	echo "If running on PACE, you may run into issues with your .cache directory eating your entire disk quota."
	echo "It is recommended to move .cache to the scratch folder, and replace it with a symbolic link to it in your home directory."
	echo ""
	echo "cd ~/"
	echo "mv .cache scratch/.cache"
	echo "ln -s -T scratch/.cache .cache"
else
	# 2.1b. Run install scripts without slurm
	cd ../
    echo "No sbatch found, running install without sbatch"
    bash pace/install_scripts/install_py_packages.sh
fi
