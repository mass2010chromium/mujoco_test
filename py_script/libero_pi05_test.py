"""
Run the LIBERO-finetuned vanilla pi05 model (`pi05_libero_100`) on LIBERO rollouts.

Mirrors the benchmark structure of ``libero_atomicVLA_test.py`` but uses the
plain :class:`openpi.policies.policy.Policy` produced by
``create_trained_policy`` — no AtomicVLA reasoning, no router, no overlays.

Data flow (matches the training contract of `pi05_libero_100`):
- prompt = task instruction (LIBERO env's task language)
- observation/image: agentview, after LeRobot 180-deg rotation (``[::-1, ::-1, :]``)
- observation/wrist_image: wrist eye-in-hand, same flip
- observation/state: 8-D = (eef_xyz | eef_axisangle | gripper_qpos[2])

The policy returns an action chunk of shape ``(action_horizon, 7)``
(``LiberoOutputs`` trims to the first 7 channels).

Default checkpoint:
``/work/hdd/bgtb/zhong2/checkpoints/pi05_libero_100/pi05_libero_100/99999``
"""
from __future__ import annotations

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
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


MODEL_NAME = "pi05_libero_100"
TASK_SUITE_NAME = "libero_goal"    # "libero_10"  "libero_goal"  "libero_spatial"  "libero_object"
TARGET_TASK_ID = None
# Set to a specific step directory if auto-discovery doesn't find your run.
CHECKPOINT_DIR = "/work/hdd/bgtb/zhong2/checkpoints/pi05_libero_100/pi05_libero_100/99999"
TRIALS_PER_TASK = 10
SEED = 7
EPISODE_LENGTH = 800
FRAME_RATE = 20
LIBERO_ENV_RESOLUTION = 224
REPLAN_FRACTION = 2  # re-query the policy every action_horizon // REPLAN_FRACTION env steps
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

print("Task Suite ----------------> ", TASK_SUITE_NAME)

print(
    "TASK_SUITE_NAME:", TASK_SUITE_NAME,
    "; TRIALS_PER_TASK:", TRIALS_PER_TASK,
    "; TARGET_TASK_ID:", TARGET_TASK_ID,
)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """robosuite quaternion (x, y, z, w) -> axis-angle 3-vector."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def prompt_from_obs(obs, task):
    """Build the raw obs dict expected by the pi05 LIBERO policy.

    Matches `LeRobotLiberoDataConfig` -> `LiberoReasonInputs` -> `LiberoInputs`,
    which keys off ``observation/image``, ``observation/wrist_image``,
    ``observation/state``, and ``prompt``.
    """
    return {
        "observation/image": obs["agentview_image"][::-1, ::-1, :],
        "observation/wrist_image": obs["robot0_eye_in_hand_image"][::-1, ::-1, :],
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": task,
    }


def _select_checkpoint_step(path: pathlib.Path) -> pathlib.Path | None:
    path = path.expanduser().resolve()
    if not path.exists():
        return None
    if (path / "params").is_dir():
        return path

    step_dirs = [
        child
        for child in path.iterdir()
        if child.is_dir() and child.name.isdigit() and (child / "params").is_dir()
    ]
    if not step_dirs:
        return None
    return max(step_dirs, key=lambda p: int(p.name))


def _candidate_checkpoint_dirs(model_name: str, exp_name: str) -> list[pathlib.Path]:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "checkpoints" / model_name / exp_name,
        repo_root / "pace" / "openpi" / "checkpoints" / model_name / exp_name,
    ]
    user = os.environ.get("USER")
    if user:
        candidates.append(pathlib.Path("/work/hdd/bgtb") / user / "checkpoints" / model_name / exp_name)
    return candidates


def resolve_checkpoint_dir(
    model_name: str,
    *,
    checkpoint_dir: str | os.PathLike[str] | None = None,
    exp_name: str | None = None,
) -> pathlib.Path:
    exp_name = exp_name or model_name

    if checkpoint_dir is not None:
        checkpoint_str = os.fspath(checkpoint_dir)
        if checkpoint_str.startswith("gs://"):
            resolved = pathlib.Path(download.maybe_download(checkpoint_str)).resolve()
        else:
            resolved = pathlib.Path(checkpoint_str).expanduser().resolve()
        step_dir = _select_checkpoint_step(resolved)
        if step_dir is None:
            raise FileNotFoundError(f"Checkpoint path does not contain a valid checkpoint: {resolved}")
        return step_dir

    searched = []
    for candidate in _candidate_checkpoint_dirs(model_name, exp_name):
        searched.append(str(candidate))
        step_dir = _select_checkpoint_step(candidate)
        if step_dir is not None:
            return step_dir

    raise FileNotFoundError(
        "Could not locate a trained pi05_libero_100 checkpoint. Checked:\n- " + "\n- ".join(searched)
    )


def _load_pi05_norm_stats(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
    *,
    assets_dir: str | os.PathLike[str] | None = None,
):
    """Prefer norm_stats saved alongside the checkpoint; fall back to a user-provided assets dir."""
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = data_config.norm_stats
    asset_id = data_config.asset_id

    if asset_id is None:
        return data_config, norm_stats

    candidate_roots = [checkpoint_dir / "assets"]
    if assets_dir is not None:
        candidate_roots.append(pathlib.Path(assets_dir).expanduser().resolve())

    loaded_from: pathlib.Path | None = None
    for root in candidate_roots:
        if (root / asset_id).exists():
            norm_stats = _checkpoints.load_norm_stats(root, asset_id)
            loaded_from = root / asset_id
            break

    if norm_stats is None:
        print(
            f"[norm_stats] WARNING: no norm_stats found for asset_id={asset_id!r}; "
            f"searched: {[str(r) for r in candidate_roots]}"
        )
    else:
        print(f"[norm_stats] loaded from {loaded_from}")
        for key, stats in norm_stats.items():
            mean = np.asarray(stats.mean)
            std = np.asarray(stats.std)
            print(
                f"  {key}: shape={mean.shape}, "
                f"mean[:3]={np.round(mean.flatten()[:3], 4).tolist()}, "
                f"std[:3]={np.round(std.flatten()[:3], 4).tolist()}"
            )

    return data_config, norm_stats


def create_pi05FT_policy(model_name, checkpoint_dir=None, assets_dir=None):
    train_config = _config.get_config(model_name)
    checkpoint_dir = resolve_checkpoint_dir(model_name, checkpoint_dir=checkpoint_dir, exp_name=model_name)
    _, norm_stats = _load_pi05_norm_stats(train_config, checkpoint_dir, assets_dir=assets_dir)

    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        norm_stats=norm_stats,
    )
    return policy, checkpoint_dir


if __name__ == "__main__":
    benchmark_dict = benchmark.get_benchmark_dict()
    if TASK_SUITE_NAME not in benchmark_dict:
        raise SystemExit(
            f"Unknown task suite: {TASK_SUITE_NAME!r}. Available: {list(benchmark_dict)!r}"
        )
    task_suite = benchmark_dict[TASK_SUITE_NAME]()
    num_tasks_in_suite = task_suite.n_tasks

    openpi_root = pathlib.Path(__file__).resolve().parent.parent / "pace" / "openpi"
    assets_dir = openpi_root / "assets" / MODEL_NAME

    print("Loading pi05_libero_100 policy...", end="", flush=True)
    policy, checkpoint_dir = create_pi05FT_policy(
        MODEL_NAME,
        checkpoint_dir=CHECKPOINT_DIR,
        assets_dir=assets_dir,
    )
    print("Done.")
    print(f"Using checkpoint: {checkpoint_dir}")

    output_dir = pathlib.Path(f"{TASK_SUITE_NAME}_benchmark_pi05FT")
    output_dir.mkdir(parents=True, exist_ok=True)

    trials_success: list[int] = []
    failure_records: list[dict] = []

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        if TARGET_TASK_ID is not None and task_id != TARGET_TASK_ID:
            continue

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, SEED)
        print(f"[Task {task_id}] '{task_description}'", flush=True)

        for episode_idx in tqdm.tqdm(range(TRIALS_PER_TASK), leave=False):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            # Settle the env (matches the AtomicVLA / TraceVLA reference scripts).
            for _ in range(20):
                obs, _reward, _done, _info = env.step(LIBERO_DUMMY_ACTION)

            # Initial action chunk.
            actions = np.asarray(policy.infer(prompt_from_obs(obs, task_description))["actions"])
            if actions.ndim != 2:
                raise ValueError(f"Expected an action chunk of shape [horizon, dim], got {actions.shape!r}")

            replan_interval = max(1, len(actions) // REPLAN_FRACTION)
            trajectory_idx = 0
            done = False
            frames = []
            wrist_frames = []

            for step_idx in range(EPISODE_LENGTH):
                act = np.copy(actions[trajectory_idx])
                obs, _reward, done, _info = env.step(act)
                frames.append(np.copy(obs["agentview_image"][::-1, ::-1, :]))
                wrist_frames.append(np.copy(obs["robot0_eye_in_hand_image"][::-1, ::-1, :]))
                trajectory_idx += 1

                if done:
                    break

                if trajectory_idx >= replan_interval:
                    actions = np.asarray(policy.infer(prompt_from_obs(obs, task_description))["actions"])
                    trajectory_idx = 0
                    replan_interval = max(1, len(actions) // REPLAN_FRACTION)

            trials_success.append(1 if done else 0)
            if not done:
                failure_records.append(
                    {
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "task_prompt": task_description,
                    }
                )

            if frames:
                mediapy.write_image(
                    f"franka_libero_pi05FT_task{task_id}_f0.png",
                    frames[0],
                )
                mediapy.write_video(
                    output_dir / f"franka_libero_pi05FT_task{task_id}_ep{episode_idx}.mp4",
                    frames,
                    fps=FRAME_RATE,
                    codec="mpeg4",
                    bps=1_000_000,
                )

            # if wrist_frames:
            #     mediapy.write_video(
            #         output_dir / f"franka_libero_pi05FT_wrist_task{task_id}_ep{episode_idx}.mp4",
            #         wrist_frames,
            #         fps=FRAME_RATE,
            #         codec="mpeg4",
            #         bps=1_000_000,
            #     )

    print(f"Trials success: {trials_success}")
    print(f"Failure records: {failure_records}")
    if trials_success:
        print(f"Trials success rate: {sum(trials_success) / len(trials_success)}")
    else:
        print("Trials success rate: no episodes were run.")
