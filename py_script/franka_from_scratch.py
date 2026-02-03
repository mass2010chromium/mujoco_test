"""
Script does not work. Potentially due to an issue with Madrona-MJX itself.
https://github.com/google-deepmind/mujoco_playground/issues/228
"""
import functools
import os
import time

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import jax
from jax import numpy as jp
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media

from mujoco_playground import wrapper

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

from panda_env import PandaPickCustom

import profiling as prof
prof.prof_start()

# Load pi model
vla_config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
# Create a trained policy.
policy = _policy_config.create_trained_policy(vla_config, checkpoint_dir)

# Coordinate between Jax and the Madrona rendering backend
def limit_jax_mem(limit):
	os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"

limit_jax_mem(0.6)
mem_limit = 4*1024*1024*1024
# Madrona memory limit: 2GB, i guess
os.environ["MADRONA_MWGPU_DEVICE_HEAP_SIZE"] = str(mem_limit)

N = 1000
num_envs = 1
ctrl_dt = 0.04
episode_length = int(9 / ctrl_dt)

config_overrides = {
    "vision": True,
    "vision_config.render_batch_size": num_envs,
    "action_repeat": 1,
    "ctrl_dt": ctrl_dt,
    "episode_length": episode_length,
}
env = PandaPickCustom(
    config_overrides=config_overrides
)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)



rng = jax.random.PRNGKey(42)
rollout = []
n_episodes = 1

env_render = prof.profiled(env.render, name='render')

@prof.profiled
def prompt_from_state(state):
    # qpos, qvel, gripper
    qpos = np.array(state.obs['state'][:7])
    gripper_pos = np.array(state.obs['state'][14])

    frame = env_render([state], width=224, height=224)[0]
    return {
        'observation/exterior_image_1_left': frame,
        'observation/wrist_image_left': np.zeros((224, 224, 3), dtype=np.uint8),
        'observation/joint_position': qpos,
        'observation/gripper_position': gripper_pos,
        'prompt': 'Grab and pick up the red cube'
    }

for _ in range(n_episodes):
  rng, state_rng = jax.random.split(rng)
  state = jit_reset(state_rng)
  prompt = prompt_from_state(state)
  actions = policy.infer(prompt)['actions']
  rollout.append(state)

  jit_step = jit_step.lower(
	state, actions[0]
  ).compile()
  jit_step = prof.profiled(jit_step, name='sim_step')
  policy_infer = prof.profiled(policy.infer, name='vla_inference')

  trajectory_idx = 0
  for i in range(episode_length):
    act = actions[trajectory_idx]
    state = jit_step(state, act)
    rollout.append(state)
    trajectory_idx += 1
    if trajectory_idx == len(actions):
        prompt = prompt_from_state(state)
        actions = policy_infer(prompt)['actions']
        trajectory_idx = 0

render_every = 1
frames = env.render(rollout[::render_every])
media.write_video('franka_reach.mp4', frames, fps=1.0 / env.dt / render_every)
