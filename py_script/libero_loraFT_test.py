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

def create_pi05(model_name, checkpoint_dir=None, assets_dir=None):
    vla_config = _config.get_config(model_name)
    if checkpoint_dir is None:
        checkpoint_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{model_name}")
    else:
        checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()
    
    # Load norm stats from assets dir if checkpoint doesn't have them
    norm_stats = None
    if assets_dir is not None:
        data_config = vla_config.data.create(vla_config.assets_dirs, vla_config.model)
        if data_config.asset_id is not None:
            from openpi.training import checkpoints as _checkpoints
            assets_path = pathlib.Path(assets_dir).resolve()
            norm_stats = _checkpoints.load_norm_stats(assets_path, data_config.asset_id)
    
    policy = _policy_config.create_trained_policy(
        vla_config, 
        checkpoint_dir,
        norm_stats=norm_stats
    )
    return policy

# EE control
# def prompt_from_obs(obs, task):
#     qpos = np.array(obs['robot0_joint_pos'])
#     gripper_pos = np.array([obs['robot0_gripper_qpos'][0]])
#     return {
#         # Flip the ego camera?
#         'observation/image': obs['agentview_image'][::-1, ::-1, :],
#         'observation/wrist_image': obs['robot0_eye_in_hand_image'][::-1, ::-1, :],
#         "observation/state": np.concatenate(
#             (
#                 obs["robot0_eef_pos"],
#                 _quat2axisangle(obs["robot0_eef_quat"]),
#                 obs["robot0_gripper_qpos"],
#             )
#         ),
#         'prompt': task
#     }

def prompt_from_obs(obs, task):
    state_vec = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )
    return {
        'observation/image': obs['agentview_image'][::-1, ::-1, :],
        'observation/wrist_image': obs['robot0_eye_in_hand_image'][::-1, ::-1, :],
        "observation/state": state_vec,
        "state": state_vec,  # required by LiberoReasonInputs
        'prompt': task, 

        # Pi0 Fuse requires "thought"; use task as initial context for action generation
        # NOTE Not sure if this is true, need to check!!!
        'thought': [task],
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

    # Point to openpi assets (where compute_norm_stats.py wrote them)
    openpi_root = pathlib.Path(__file__).resolve().parent.parent / "pace" / "openpi"
    assets_dir = openpi_root / "assets" / "pi05_libero_10_reason_lora"

    print("Loading pi05...", end='', flush=True)
    policy = create_pi05(   # Use LoRA weights:
        'pi05_libero_10_reason_lora',
        # checkpoint_dir='pace/openpi/checkpoints/pi05_libero_10_reason_lora/pi05_libero_10_reason_lora/9999',
        checkpoint_dir=(
            pathlib.Path(__file__).resolve().parent.parent 
            / 'pace' / 'openpi' / 'checkpoints' 
            / 'pi05_libero_10_reason_lora' / 'pi05_libero_10_reason_lora' / '9999'
        ),
        assets_dir=assets_dir
    )

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
            prompt = prompt_from_obs(obs, task_description)
            result = policy.infer(prompt)
            actions = result['actions']
            subtask = result.get('subtask', None)
            print(f"[Task {task_id}] Task: {task_description}")
            print(f"[Task {task_id}] Initial reasoning: {subtask}")

            rollout = []
            frames = []
            wrist_frames = []
            trajectory_idx = 0
            for i in range(EPISODE_LENGTH):
                act = np.copy(actions[trajectory_idx])
                obs, reward, done, info = env.step(act)
                rollout.append(obs)
                frames.append(obs['agentview_image'][::-1, ::-1, :])
                wrist_frames.append(obs['robot0_eye_in_hand_image'][::-1, ::-1, :])
                trajectory_idx += 1
                if trajectory_idx == (len(actions) // 2):
                    prompt = prompt_from_obs(obs, task_description)
                    result = policy.infer(prompt)
                    actions = result['actions']
                    subtask = result.get('subtask', None)
                    print(f"[Step {i}] Reasoning: {subtask}")
                    trajectory_idx = 0
            mediapy.write_image('franka_libero_f0.png', frames[0])
            mediapy.write_video('franka_libero.mp4', frames, fps=FRAME_RATE)
            mediapy.write_video('franka_libero_wrist.mp4', wrist_frames, fps=FRAME_RATE)
            break
        break

