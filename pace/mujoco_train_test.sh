#! /bin/env bash
#SBATCH -J IsaacSimBuildLibC
#SBATCH --nodefile=pace/nodes_rtx6000.txt
#SBATCH -N1 --ntasks-per-node=1 --cpus-per-task=12
#SBATCH --mem 32G
#SBATCH --gpus=rtx_6000

cd mujoco_playground
source .venv/bin/activate
train-jax-ppo --env_name CartpoleBalance
