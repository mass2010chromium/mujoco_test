#! /bin/env bash
#SBATCH -J InstallMujoco
#SBATCH --nodefile=pace/nodes_rtx_6000.txt
#SBATCH -N1 --ntasks-per-node=1 --cpus-per-task=12
#SBATCH --mem 32G
#SBATCH --gpus=rtx_6000

# Madrona-MJX install wants CUDA<12.5.1 for some reason
module load cuda/12.1.1

### BEGIN Mujoco install

cd mujoco_playground

uv venv --python 3.12
source .venv/bin/activate
# Jax versions >=0.6.0 will not work, since some API's were deprecated and removed. Until this library is updated, use any version before 0.6.0 -- Madrona
uv pip install -U "jax[cuda12]<0.6.0" --index-url https://pypi.org/simple --force-reinstall

python -c "import jax; print('JAX default backend (should be gpu):', jax.default_backend())"

uv --no-config sync --all-extras
uv --no-config run python -c "import mujoco_playground; print('mujoco_playground: import Success')"
uv --no-config run python -c "from mujoco_playground import locomotion; locomotion.load('G1JoystickFlatTerrain')"

cd ..

### END Mujoco install


### BEGIN Madrona-MJX install

cd madrona_mjx
git submodule update --init --recursive

mkdir build
cd build
cmake ..
make -j12
cd ..
uv pip install -e .

### END Madrona-MJX install
