"""
Reasoning-policy evaluation for Pi05 LoRA fine-tuning on LIBERO.

This script mirrors `libero_loraFT_test.py` but uses a stateful
ReasoningPolicy so thought updates happen inside the policy class.
"""

import math
import os
import pathlib


def limit_jax_mem(limit: float) -> None:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"


limit_jax_mem(0.6)

import mediapy
import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment and task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """Copied from robosuite transform utils."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def create_pi05_reasoning(model_name, checkpoint_dir=None, assets_dir=None):
    vla_config = _config.get_config(model_name)
    if checkpoint_dir is None:
        checkpoint_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{model_name}")
    else:
        checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()

    norm_stats = None
    if assets_dir is not None:
        data_config = vla_config.data.create(vla_config.assets_dirs, vla_config.model)
        if data_config.asset_id is not None:
            from openpi.training import checkpoints as _checkpoints

            assets_path = pathlib.Path(assets_dir).resolve()
            norm_stats = _checkpoints.load_norm_stats(assets_path, data_config.asset_id)

    initial_scene_plan = (
        "Plan: TBD.\n"
        " What I have done: TBD.\n"
        "Now I need to do: TBD.\n"
    )
    return _policy_config.create_trained_reasoning_policy(
        vla_config,
        checkpoint_dir,
        norm_stats=norm_stats,
        initial_scene_plan=initial_scene_plan,
        sample_kwargs={
            "temperature": 0.0,
            "max_reasoning_steps": 256,
            "force_initial_reasoning": True,
        },
    )


def prompt_from_obs(obs, task):
    state_vec = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )
    return {
        "observation/image": obs["agentview_image"][::-1, ::-1, :],
        "observation/wrist_image": obs["robot0_eye_in_hand_image"][::-1, ::-1, :],
        "observation/state": state_vec,
        "prompt": task,
    }


def infer_until_action(policy, prompt, max_think_rounds=8):
    """Query the policy until it returns actions (not a thinking response)."""
    result = policy.infer(prompt)
    think_round = 0
    while result.get("isthinking", False):
        think_round += 1
        print(f"[Thinking {think_round}] {result.get('thought', '')}")
        if think_round >= max_think_rounds:
            raise RuntimeError(
                f"Policy is still thinking after {max_think_rounds} rounds; "
                "aborting this rollout."
            )
        result = policy.infer(prompt)
    return result


LIBERO_ENV_RESOLUTION = 224
TRIALS_PER_TASK = 1
SEED = 7
EPISODE_LENGTH = 1000
FRAME_RATE = 20


if __name__ == "__main__":
    benchmark_dict = benchmark.get_benchmark_dict()
    # task_suite = benchmark_dict["libero_90"]()
    # task_suite = benchmark_dict['libero_spatial']()
    task_suite = benchmark_dict['libero_goal']()
    # task_suite = benchmark_dict["libero_10"]()
    num_tasks_in_suite = task_suite.n_tasks

    openpi_root = pathlib.Path(__file__).resolve().parent.parent / "pace" / "openpi"
    assets_dir = openpi_root / "assets" / "pi05_libero_reason_lora"

    print("Loading pi05 reasoning policy...", end="", flush=True)
    policy = create_pi05_reasoning(
        "pi05_libero_reason_lora",
        checkpoint_dir=(
            pathlib.Path(__file__).resolve().parent.parent
            / "pace"
            / "openpi"
            / "checkpoints"
            / "pi05_libero_reason_lora"
            / "pi05_libero_reason_lora"
            / "31799"
        ),
        assets_dir=assets_dir,
    )
    print("Done.")
    trials_success = []

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, SEED)

        # task_description = "pick up the wine bottle"
        # task_description = "grasp the stove knob"
        task_description = "open the top drawer of the cabinet"

        if task_id != 2:       # test task 10 --> drawer closed initially
            continue

        for episode_idx in tqdm.tqdm(range(TRIALS_PER_TASK)):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            if hasattr(policy, "start"):
                policy.start()

            result = infer_until_action(policy, prompt_from_obs(obs, task_description))
            actions = result["actions"]
            print(f"[Task {task_id}] Task: {task_description}")
            print(f"[Task {task_id}] Initial scene plan: {result.get('subtask', None)}")

            rollout = []
            frames = []
            wrist_frames = []
            trajectory_idx = 0
            done = False

            for i in range(EPISODE_LENGTH):
                act = np.copy(actions[trajectory_idx])
                obs, reward, done, info = env.step(act)
                rollout.append(obs)
                frames.append(obs["agentview_image"][::-1, ::-1, :])
                wrist_frames.append(obs["robot0_eye_in_hand_image"][::-1, ::-1, :])
                trajectory_idx += 1

                if done:
                    break

                if trajectory_idx == (len(actions) // 2):
                    result = infer_until_action(policy, prompt_from_obs(obs, task_description))
                    actions = result["actions"]
                    print(f"[Step {i}] Scene plan: {result.get('subtask', None)}")

                    mediapy.write_image(
                        "trial_imgs/frame_agentview" + str(i) + ".png",
                        np.copy(obs["agentview_image"][::-1, ::-1, :]),
                    )
                    mediapy.write_image(
                        "trial_imgs/frame_wrist" + str(i) + ".png",
                        np.copy(obs["robot0_eye_in_hand_image"][::-1, ::-1, :]),
                    )
                    trajectory_idx = 0

            if done:
                trials_success.append(1)
            else:
                trials_success.append(0)
            mediapy.write_image("franka_libero_f0.png", frames[0])
            mediapy.write_video("franka_libero_reason_" + str(task_id) + ".mp4", frames, fps=FRAME_RATE)
            # mediapy.write_video("franka_libero_wrist_reason_" + str(task_id) + ".mp4", wrist_frames, fps=FRAME_RATE)
        #     break
        # break

    print(f"Trials success: {trials_success}")
    print(f"Trials success rate: {sum(trials_success) / len(trials_success)}")