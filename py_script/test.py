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

# Coordinate between Jax and the Madrona rendering backend
def limit_jax_mem(limit):
	os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"

limit_jax_mem(0.6)
# Reduce madrona memory allocation to 1GB as cartpole doesn't need much
os.environ["MADRONA_MWGPU_DEVICE_HEAP_SIZE"] = "1073741824"


from mujoco_playground import dm_control_suite

N = 1000
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



from mujoco_playground.config import dm_control_suite_params

# Load vision-specific PPO configuration tuned for CartpoleBalance
ppo_params = dm_control_suite_params.brax_vision_ppo_config(env_name)
ppo_params.episode_length = episode_length
ppo_params.network_factory = ppo_networks_vision.make_ppo_networks_vision

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


def progress(num_steps, metrics):
	clear_output(wait=True)

	times.append(datetime.now())
	x_data.append(num_steps)
	y_data.append(metrics["eval/episode_reward"])
	y_dataerr.append(metrics["eval/episode_reward_std"])
	print(f"Step {x_data[-1]}, reward={y_data[-1]}, std={y_dataerr[-1]}, time={times[-1]}")

train_fn = functools.partial(
    ppo.train, **dict(ppo_params), progress_fn=progress
)

make_inference_fn, params, metrics = train_fn(environment=env)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

plt.figure(0)
plt.clf()
plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
plt.ylim([0, 100])
plt.xlabel("# environment steps")
plt.ylabel("reward per episode")
plt.title(f"y={y_data[-1]:.3f}")
plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
plt.savefig('train_reward.png')

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

rng = jax.random.PRNGKey(42)
rollout = []
n_episodes = 1


def unvmap(x):
  return jax.tree.map(lambda y: y[0], x)

for _ in range(n_episodes):
  key_rng = jax.random.split(rng, num_envs)
  state = jit_reset(key_rng)
  rollout.append(unvmap(state))
  for i in range(episode_length):
    act_rng, rng = jax.random.split(rng)
    act_rng = jax.random.split(act_rng, num_envs)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(unvmap(state))

render_every = 1
frames = env.render(rollout[::render_every], camera="fixed")
rewards = [s.reward for s in rollout]
media.write_video('vision_cartpole.mp4', frames, fps=1.0 / env.dt / render_every)

plt.figure(0)
plt.clf()
plt.plot(np.convolve(rewards, np.ones(10) / 10, mode="valid"))
plt.xlabel("time step")
plt.ylabel("reward")
