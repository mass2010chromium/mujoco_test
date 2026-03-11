#! /bin/env bash
#SBATCH -J InstallMujoco
#SBATCH -N1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --mem 32G

# Madrona-MJX install wants CUDA<12.5.1 for some reason
module load cuda/12.1.1 || echo "module load failed, OK if running locally / without sbatch"


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
# cd madrona_mjx
# git submodule update --init --recursive
# 
# mkdir build
# cd build
# cmake ..
# make -j4
# cd ..
# uv pip install -e .
# cd ..
### END Madrona-MJX install

# Seems like at some point jax gets reinstalled... let's fix that
uv pip install -U "jax[cuda12]<0.6.0" --index-url https://pypi.org/simple --force-reinstall

### LIBERO
cd lerobot-libero
git apply ../pace/install_scripts/lerobot-libero.patch
uv pip install -r requirements.txt
uv pip install -e .
cd ..


### Pi0 install
cd pace/install_scripts
bash install_pi0.sh

### notebook install
bash install_notebook.sh

### SAM3, pddlsim install
bash install_grounding.sh

### Huggingface datasets issue?
# https://github.com/Physical-Intelligence/openpi/issues/561
uv pip install datasets==3.6.0
