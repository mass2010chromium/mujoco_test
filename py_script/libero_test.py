"""
Adapted from openpi libero eval script
REQUIRES: robosuite 1.4.0
"""
import pathlib
import math
import os

def limit_jax_mem(limit):
	os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"
limit_jax_mem(0.6)

import numpy as np
import mediapy
import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def create_pi05(model_name):
    vla_config = _config.get_config(model_name)
    checkpoint_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{model_name}")
    policy = _policy_config.create_trained_policy(vla_config, checkpoint_dir)
    return policy

# EE control
def prompt_from_obs(obs, task):
    qpos = np.array(obs['robot0_joint_pos'])
    gripper_pos = np.array([obs['robot0_gripper_qpos'][0]])
    return {
        # Flip the ego camera?
        'observation/image': obs['agentview_image'][::-1, ::-1, :],
        'observation/wrist_image': obs['robot0_eye_in_hand_image'][::-1, ::-1, :],
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        'prompt': task
    }

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 224 # resolution used to render training data
TRIALS_PER_TASK = 10
SEED = 7
EPISODE_LENGTH = 400
FRAME_RATE = 20

if __name__ == "__main__":
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict['libero_spatial']()
    num_tasks_in_suite = task_suite.n_tasks

    print("Loading pi05...", end='', flush=True)
    policy = create_pi05('pi05_libero')
    print("Done.")

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, SEED)

        for episode_idx in tqdm.tqdm(range(TRIALS_PER_TASK)):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            print(obs)
            prompt = prompt_from_obs(obs, task_description)
            actions = policy.infer(prompt)['actions']

            rollout = []
            frames = []
            wrist_frames = []
            trajectory_idx = 0
            act_scale = 1
            for i in range(EPISODE_LENGTH):
                act = np.copy(actions[trajectory_idx])
                obs, reward, done, info = env.step(act * act_scale)
                rollout.append(obs)
                frames.append(obs['agentview_image'][::-1, ::-1, :])
                wrist_frames.append(obs['robot0_eye_in_hand_image'][::-1, ::-1, :])
                trajectory_idx += 1
                if trajectory_idx == (len(actions)//2):
                    prompt = prompt_from_obs(obs, task_description)
                    actions = policy.infer(prompt)['actions']
                    trajectory_idx = 0
            mediapy.write_video('franka_libero.mp4', frames, fps=FRAME_RATE)
            mediapy.write_video('franka_libero_wrist.mp4', wrist_frames, fps=FRAME_RATE)
            break
        break

