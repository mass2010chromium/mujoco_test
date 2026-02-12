"""
Test pi05_droid model weights on the LIBERO spatial benchmark.
Adapted from libero_test.py (which uses pi05_libero).

Key adaptations:
1. Loads pi05_droid model with DROID-specific norm stats
   (bypasses the hardcoded pi05_libero norm stats path in policy_config.py).
2. Uses JOINT_POSITION controller instead of OSC_POSE:
   - pi05_droid outputs 8D actions: delta joint positions (7) + gripper position (1)
   - pi05_libero outputs 7D actions: delta EEF pose (6) + gripper (1) for OSC_POSE
3. Maps LIBERO observations to DROID input format:
   - observation/exterior_image_1_left  <-- agentview_image
   - observation/wrist_image_left       <-- robot0_eye_in_hand_image
   - observation/joint_position         <-- robot0_joint_pos  (7D, same Franka Panda)
   - observation/gripper_position       <-- robot0_gripper_qpos[0] (normalized to ~[0,1])
4. Converts model's gripper position output to robosuite's [-1, +1] gripper action.
5. Widens the JOINT_POSITION controller's output range so the model's raw delta
   joint positions (in radians) pass through without unwanted clipping/scaling.
   CRITICAL: This must happen AFTER env.reset(), because robosuite re-creates
   the controller during reset, reverting any prior modifications.
6. pi05_droid uses action_horizon=15 (vs 10 for pi05_libero).

Notes:
- Both DROID and LIBERO use a 7-DOF Franka Panda robot, so the joint position
  spaces are compatible. The DROID norm stats should be reasonably applicable.
- The DROID training data was collected at ~15 Hz, while LIBERO runs at 20 Hz.
  This means the model's per-step deltas were calibrated for a slightly longer
  timestep. ACT_SCALE (default 1.0) can be tuned to compensate.
- Performance is NOT expected to match pi05_libero, since pi05_droid was fine-tuned
  on DROID data (different camera viewpoints, workspaces, tasks). This script is
  for experimental cross-evaluation.

REQUIRES: robosuite 1.4.0 (same as libero_test.py)
"""
import pathlib
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
from openpi.training import checkpoints as _checkpoints


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def _get_libero_env(task, resolution, seed):
    """Initialize LIBERO environment with JOINT_POSITION controller.

    pi05_droid was trained on DROID data using joint position actions
    (delta joint positions + gripper position). LIBERO's default OSC_POSE
    controller expects Cartesian end-effector deltas, which is incompatible.

    By switching to JOINT_POSITION, the env accepts 8D actions:
      [delta_joint_pos (7), gripper_action (1)]
    which aligns with pi05_droid's 8D output (after gripper conversion).
    """
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        # Switch from default OSC_POSE to JOINT_POSITION controller
        "controller": "JOINT_POSITION",
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)

    # NOTE: We do NOT call _widen_controller_range here because robosuite
    # re-creates the controller during env.reset(), undoing any modifications.
    # Instead, _widen_controller_range must be called AFTER every reset.

    return env, task_description


def _widen_controller_range(env, max_delta=5.0):
    """Widen the JOINT_POSITION controller's input/output range.

    After this, passing an action value of X (in radians) results in a joint
    position delta of X radians, without clipping for |X| < max_delta.

    IMPORTANT: Must be called AFTER env.reset() / env.set_init_state(),
    because robosuite re-creates the controller during reset, undoing any
    prior modifications.
    """
    controller = env.robots[0].controller
    n_joints = len(controller.output_max)  # Should be 7 for Panda
    controller.output_max = np.ones(n_joints) * max_delta
    controller.output_min = np.ones(n_joints) * (-max_delta)
    controller.input_max = np.ones(n_joints) * max_delta
    controller.input_min = np.ones(n_joints) * (-max_delta)
    # Reset the cached action_scale so it's recomputed with the new ranges.
    # (base_controller.scale_action lazily computes and caches this.)
    controller.action_scale = None
    controller.action_input_transform = None
    controller.action_output_transform = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def create_pi05_droid():
    """Load pi05_droid model with correct DROID norm stats.

    The modified policy_config.py in this codebase has a hardcoded path to
    pi05_libero for loading norm stats. For pi05_droid, we need the DROID
    norm stats, so we load them explicitly from the pi05_droid checkpoint
    and pass them to create_trained_policy to bypass the hardcoded path.
    """
    vla_config = _config.get_config("pi05_droid")
    checkpoint_dir = download.maybe_download(
        "gs://openpi-assets/checkpoints/pi05_droid"
    )

    # Load DROID norm stats from the pi05_droid checkpoint's assets directory.
    # The asset_id is "droid" (matches the pi05_droid config's AssetsConfig).
    norm_stats = _checkpoints.load_norm_stats(
        pathlib.Path(checkpoint_dir) / "assets", "droid"
    )

    # Pass norm_stats explicitly to skip the hardcoded pi05_libero path.
    policy = _policy_config.create_trained_policy(
        vla_config, checkpoint_dir, norm_stats=norm_stats
    )
    return policy


# ---------------------------------------------------------------------------
# Observation mapping: LIBERO --> DROID format
# ---------------------------------------------------------------------------

# Approximate max opening of Panda gripper fingers (in joint-space radians).
# Used to normalize LIBERO's raw gripper qpos to [0, 1] range expected by DROID.
PANDA_GRIPPER_MAX_QPOS = 0.04

def prompt_from_obs(obs, task):
    """Convert a LIBERO observation dict to the DROID input format.

    DROID model (DroidInputs) expects:
      observation/exterior_image_1_left : (H, W, 3) uint8  -- base camera
      observation/wrist_image_left      : (H, W, 3) uint8  -- wrist camera
      observation/joint_position        : (7,) float        -- joint angles (rad)
      observation/gripper_position      : (1,) float        -- gripper pos in [0,1]
      prompt                            : str               -- language instruction

    LIBERO provides:
      agentview_image              : (H, W, 3) uint8 (needs 180° rotation)
      robot0_eye_in_hand_image     : (H, W, 3) uint8 (needs 180° rotation)
      robot0_joint_pos             : (7,) float       (Franka Panda joint angles)
      robot0_gripper_qpos          : (2,) float       (left/right finger qpos)
    """
    joint_pos = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)       # (7,)

    # Normalize gripper finger position to [0, 1].
    # LIBERO robot0_gripper_qpos[0] is in ~[0, 0.04] (joint-space).
    # DROID gripper_position is in ~[0, 1]  (0 = closed, 1 = open).
    raw_gripper = obs["robot0_gripper_qpos"][0]
    gripper_pos = np.array(
        [np.clip(raw_gripper / PANDA_GRIPPER_MAX_QPOS, 0.0, 1.0)],
        dtype=np.float32,
    )

    print("task description:", task)
    return {
        # 180° rotation to match DROID camera convention (same as libero_test.py)
        "observation/exterior_image_1_left": obs["agentview_image"][::-1, ::-1, :],
        "observation/wrist_image_left": obs["robot0_eye_in_hand_image"][::-1, ::-1, :],
        "observation/joint_position": joint_pos,
        "observation/gripper_position": gripper_pos,
        "prompt": task,
    }


# ---------------------------------------------------------------------------
# Action mapping: DROID output --> robosuite JOINT_POSITION action
# ---------------------------------------------------------------------------

def convert_gripper_action(gripper_position):
    """Convert DROID gripper position to robosuite gripper action.

    DROID convention (approx):  1.0 = open,  0.0 = closed  (position)
    robosuite convention:      -1.0 = open, +1.0 = closed  (action command)

    Linear mapping:  action = 1.0 - 2.0 * position
      position=0 (closed) --> action=+1 (close)
      position=1 (open)   --> action=-1 (open)
    """
    return 1.0 - 2.0 * np.clip(gripper_position, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LIBERO_ENV_RESOLUTION = 224   # Rendering resolution (matches training data)
TRIALS_PER_TASK = 10
SEED = 7
EPISODE_LENGTH = 400
FRAME_RATE = 20

# Scale factor for joint position deltas.  Tune this if robot moves too
# fast/slow.  1.0 = pass model output directly (in radians).
# If DROID training was at ~15 Hz and LIBERO runs at 20 Hz, you may want
# ~0.75 to compensate for the shorter timestep (0.75 ≈ 15/20).
ACT_SCALE = 1.0


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    num_tasks_in_suite = task_suite.n_tasks

    print("Loading pi05_droid...", end="", flush=True)
    policy = create_pi05_droid()
    print("Done.")

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment with JOINT_POSITION controller
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, SEED)

        for episode_idx in tqdm.tqdm(range(TRIALS_PER_TASK)):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            # Widen controller AFTER reset (reset re-creates the controller
            # with default output_max=0.05, which scales down deltas by ~50x).
            _widen_controller_range(env, max_delta=5.0)

            prompt = prompt_from_obs(obs, task_description)
            result = policy.infer(prompt)
            # actions shape: (action_horizon, 8) -- 15 steps of 8D actions
            actions = result["actions"]

            rollout = []
            frames = []
            wrist_frames = []
            trajectory_idx = 0

            joint_start = np.array(obs["robot0_joint_pos"])
            for i in range(EPISODE_LENGTH):
                raw_act = actions[trajectory_idx]

                # First 7 dims: delta joint positions (radians)
                joint_delta = raw_act[:7] * ACT_SCALE
                # 8th dim: gripper position --> robosuite gripper action
                gripper_act = convert_gripper_action(raw_act[7])
                # Combined 8D action for JOINT_POSITION controller
                act = np.concatenate([joint_delta, [gripper_act]])

                obs, reward, done, info = env.step(act)
                rollout.append(obs)
                frames.append(obs["agentview_image"][::-1, ::-1, :])
                wrist_frames.append(obs["robot0_eye_in_hand_image"][::-1, ::-1, :])

                # Log action stats every 50 steps
                if i % 50 == 0:
                    j_now = np.array(obs["robot0_joint_pos"])
                    print(f"  step {i:3d}: |delta|={np.abs(joint_delta).mean():.5f}  "
                          f"grip_out={raw_act[7]:.3f}  "
                          f"total_joint_disp={np.abs(j_now - joint_start).sum():.4f}")

                trajectory_idx += 1
                # Re-plan at halfway through the action chunk
                # (action_horizon=15, so re-plan after ~7 steps)
                if trajectory_idx == (len(actions) // 2):
                    prompt = prompt_from_obs(obs, task_description)
                    result = policy.infer(prompt)
                    actions = result["actions"]
                    trajectory_idx = 0

            mediapy.write_video("franka_libero_droid.mp4", frames, fps=FRAME_RATE)
            mediapy.write_video(
                "franka_libero_droid_wrist.mp4", wrist_frames, fps=FRAME_RATE
            )
            break
        break
