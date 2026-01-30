"""
Copying from: https://github.com/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_1.ipynb
"""

# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
import os
import time

from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from IPython.display import clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
import numpy as np

from mujoco_playground import wrapper

np.set_printoptions(precision=3, suppress=True, linewidth=100)



from mujoco_playground import dm_control_suite

num_envs = 1024
ctrl_dt = 0.04
episode_length = int(3 / ctrl_dt)

config_overrides = {
    "vision": True,
    "vision_config.render_batch_size": num_envs,
    "action_repeat": 1,
    "ctrl_dt": ctrl_dt,
    "episode_length": episode_length,
}

env_name = "CartpoleBalance"
env = dm_control_suite.load(
    env_name, config_overrides=config_overrides
)

env = wrapper.wrap_for_brax_training(
    env,
    vision=True,
    num_vision_envs=num_envs,
    action_repeat=1,
    episode_length=episode_length,
)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)


key_reset, key_act = jax.random.split(jax.random.PRNGKey(0))
state = jit_reset(jax.random.split(key_reset, num_envs))

# Pre-compile
jit_step = jit_step.lower(
    state, jp.zeros((num_envs, env.action_size))
).compile()

t0 = time.time()
for i in range(N):
  act = jax.random.uniform(
      key_act, (num_envs, env.action_size), minval=-1.0, maxval=1.0
  )
  state = jit_step(state, act)

jax.tree_util.tree_map(
    lambda x: x.block_until_ready(), state
)  # Await device completion
dt = time.time() - t0

print("Madrona MJX: {:d} transitions per second".format(int(N * num_envs / dt)))
