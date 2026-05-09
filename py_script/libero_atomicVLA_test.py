"""
Run AtomicVLA (`Atomic_libero`) on LIBERO rollouts.

This script follows the actual `Atomic_libero` training contract:
- prompt = task instruction
- `AtomicReasoningPolicy` manages the atomic thought internally
- `prefill(...)` decides whether to think or act
- after a thinking step, we immediately call the policy again on the same
  observation so the updated atomic skill can be converted into actions
"""
import dataclasses
import math
import os
import pathlib


def limit_jax_mem(limit: float) -> None:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"


limit_jax_mem(0.6)

import cv2
import jax.numpy as jnp
import mediapy
import numpy as np
import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.shared import download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config


MODEL_NAME = "Atomic_libero"
TASK_SUITE_NAME = "libero_10"    # "libero_10"   "libero_goal"  "libero_spatial"
TARGET_TASK_ID = None
# Set this to a specific step directory if auto-discovery does not find your run.
CHECKPOINT_DIR = "/work/hdd/bgtb/zhong2/checkpoints/Atomic_libero/Atomic_libero/100000"
TRIALS_PER_TASK = 10
SEED = 7
EPISODE_LENGTH = 800
FRAME_RATE = 20
LIBERO_ENV_RESOLUTION = 224
REPLAN_FRACTION = 2
MAX_INTERNAL_POLICY_CALLS = 4
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

ROUTER_OVERLAY_MODE = 2  # 1: selected expert + weights, 2: selected expert + softmax distribution

print("TASK_SUITE_NAME: ", TASK_SUITE_NAME, "; TRIALS_PER_TASK: ", TRIALS_PER_TASK, "; TARGET_TASK_ID: ", TARGET_TASK_ID)


@dataclasses.dataclass(frozen=True)
class AtomicLiberoInferenceOutputs(_transforms.DataTransformFn):
    """Trim AtomicVLA actions to the 7-D LIBERO action space."""

    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        outputs = dict(data)
        if "actions" in outputs:
            outputs["actions"] = np.asarray(outputs["actions"][..., : self.action_dim])
        return outputs


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def prompt_from_obs(obs, task):
    """Build the raw LIBERO observation dict expected by AtomicReasoningPolicy."""
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
            raise FileNotFoundError(f"Checkpoint path does not contain a valid Atomic checkpoint: {resolved}")
        return step_dir

    searched = []
    for candidate in _candidate_checkpoint_dirs(model_name, exp_name):
        searched.append(str(candidate))
        step_dir = _select_checkpoint_step(candidate)
        if step_dir is not None:
            return step_dir

    raise FileNotFoundError(
        "Could not locate a trained AtomicVLA checkpoint. Checked:\n- " + "\n- ".join(searched)
    )


def _load_atomic_norm_stats(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
    *,
    assets_dir: str | os.PathLike[str] | None = None,
):
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = data_config.norm_stats
    asset_id = data_config.asset_id

    if asset_id is None:
        return data_config, norm_stats

    candidate_roots = []
    if assets_dir is not None:
        candidate_roots.append(pathlib.Path(assets_dir).expanduser().resolve())
    candidate_roots.append(checkpoint_dir / "assets")

    for root in candidate_roots:
        if (root / asset_id).exists():
            norm_stats = _checkpoints.load_norm_stats(root, asset_id)
            break

    return data_config, norm_stats


def create_atomic_policy(model_name, checkpoint_dir=None, assets_dir=None):
    train_config = _config.get_config(model_name)
    checkpoint_dir = resolve_checkpoint_dir(model_name, checkpoint_dir=checkpoint_dir, exp_name=model_name)
    _, norm_stats = _load_atomic_norm_stats(train_config, checkpoint_dir, assets_dir=assets_dir)

    policy = _policy_config.create_trained_atomic_policy(
        train_config,
        checkpoint_dir,
        norm_stats=norm_stats,
    )
    return policy, checkpoint_dir


def _normalize_skill_text(skill: str | None) -> str:
    if not skill:
        return ""
    return " ".join(str(skill).split()).strip()


def _current_atomic_skill(policy: _policy.AtomicReasoningPolicy) -> str:
    return _normalize_skill_text(getattr(policy, "_scene_plan", None) or getattr(policy, "_thought", None))


def _current_router_debug(policy: _policy.AtomicReasoningPolicy) -> dict | None:
    return getattr(policy, "last_router_debug", None)


def _format_router_probs(router_debug: dict | None, *, precision: int = 3) -> str:
    if not router_debug or "router_probs" not in router_debug:
        return "[]"
    return "[" + ", ".join(f"{float(prob):.{precision}f}" for prob in router_debug["router_probs"]) + "]"


def _router_overlay_lines(router_debug: dict | None) -> list[str]:
    if router_debug is None:
        return []

    selected_expert = router_debug["selected_full_expert_idx"]
    if ROUTER_OVERLAY_MODE == 1:
        return [
            f"Shared w: {router_debug['shared_weight']:.3f}",
            f"Expert {selected_expert} w: {router_debug['selected_weight']:.3f}",
        ]

    if ROUTER_OVERLAY_MODE == 2:
        probs = [float(prob) for prob in router_debug.get("router_probs", [])]
        lines = [f"Selected Expert: {selected_expert}"]
        entries = [f"E{idx + 1}={prob:.2f}" for idx, prob in enumerate(probs)]
        chunk_size = 3
        for start in range(0, len(entries), chunk_size):
            lines.append(" ".join(entries[start : start + chunk_size]))
        return lines

    raise ValueError(f"Unsupported ROUTER_OVERLAY_MODE={ROUTER_OVERLAY_MODE!r}. Expected 1 or 2.")


def infer_action_chunk(
    policy: _policy.AtomicReasoningPolicy,
    obs: dict,
    task_description: str,
    *,
    infer_since_last_think: int = 0,
    max_internal_calls: int = MAX_INTERNAL_POLICY_CALLS,
) -> tuple[np.ndarray, str, int, dict | None]:
    """Run AtomicVLA until it emits executable actions for the current observation."""
    current_skill = _current_atomic_skill(policy)

    for _ in range(max_internal_calls):
        infer_since_last_think += 1
        result = policy.infer(prompt_from_obs(obs, task_description))
        current_skill = _current_atomic_skill(policy) or current_skill

        if policy.is_thinking:
            infer_since_last_think = 0
            print(f"[AtomicVLA] Thinking -> current atomic skill: {current_skill or '<empty>'}")
            continue

        actions = np.asarray(result["actions"])
        if actions.ndim != 2:
            raise ValueError(f"Expected an action chunk of shape [horizon, dim], got {actions.shape!r}")
        router_debug = _current_router_debug(policy)
        if router_debug is not None:
            print(
                "[AtomicVLA] Router -> "
                f"shared={router_debug['shared_weight']:.4f}, "
                f"selected={router_debug['selected_weight']:.4f}, "
                f"expert={router_debug['selected_full_expert_idx']}, "
                f"probs={_format_router_probs(router_debug, precision=4)}"
            )
        return actions, current_skill, infer_since_last_think, router_debug

    raise RuntimeError("AtomicReasoningPolicy did not emit actions within the internal retry limit.")


def write_skill_to_frame(obs, current_skill, infer_since_last_think, router_debug):
    """Render debug text at the top-right of the agentview frame."""
    frame = np.copy(obs["agentview_image"][::-1, ::-1, :])
    overlay_lines = [
        f"Skill: {current_skill}" if current_skill else "Skill:",
        f"Infer since think: {infer_since_last_think}",
    ]
    overlay_lines.extend(_router_overlay_lines(router_debug))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.25
    thickness = 1
    margin = 8
    line_metrics = [cv2.getTextSize(line, font, font_scale, thickness) for line in overlay_lines]
    text_w = max(metric[0][0] for metric in line_metrics)
    text_h = max(metric[0][1] for metric in line_metrics)
    baseline = max(metric[1] for metric in line_metrics)
    line_gap = 4
    block_h = len(overlay_lines) * text_h + (len(overlay_lines) - 1) * line_gap

    x = max(margin, frame.shape[1] - text_w - margin)
    y = margin + text_h

    cv2.rectangle(
        frame,
        (max(0, x - 4), max(0, y - text_h - 4)),
        (min(frame.shape[1] - 1, x + text_w + 4), min(frame.shape[0] - 1, y - text_h + block_h + baseline + 4)),
        (0, 0, 0),
        -1,
    )
    for idx, line in enumerate(overlay_lines):
        line_y = y + idx * (text_h + line_gap)
        cv2.putText(frame, line, (x, line_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame


if __name__ == "__main__":
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[TASK_SUITE_NAME]()
    num_tasks_in_suite = task_suite.n_tasks

    openpi_root = pathlib.Path(__file__).resolve().parent.parent / "pace" / "openpi"
    assets_dir = openpi_root / "assets" / MODEL_NAME

    print("Loading AtomicVLA policy...", end="", flush=True)
    policy, checkpoint_dir = create_atomic_policy(
        MODEL_NAME,
        checkpoint_dir=CHECKPOINT_DIR,
        assets_dir=assets_dir,
    )
    print("Done.")
    print(f"Using checkpoint: {checkpoint_dir}")

    trial_image_dir = pathlib.Path("trial_imgs")
    trial_image_dir.mkdir(parents=True, exist_ok=True)

    trials_success = []
    failture_records = []

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)

        if TARGET_TASK_ID is not None and task_id != TARGET_TASK_ID:
            continue

        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, SEED)

        for episode_idx in tqdm.tqdm(range(TRIALS_PER_TASK)):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            if hasattr(policy, "reset"):
                policy.reset()
            elif hasattr(policy, "start"):
                policy.start()

            # wait for the env to settle
            for _ in range(20):
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)

            infer_since_last_think = 0
            actions, current_skill, infer_since_last_think, router_debug = infer_action_chunk(
                policy,
                obs,
                task_description,
                infer_since_last_think=infer_since_last_think,
            )
            print(f"[Task {task_id}] Task: {task_description}")
            print(f"[Task {task_id}] Starting atomic skill: {current_skill or '<empty>'}")

            frames = []
            wrist_frames = []
            trajectory_idx = 0
            replan_interval = max(1, len(actions) // REPLAN_FRACTION)
            done = False

            for step_idx in range(EPISODE_LENGTH):
                act = np.copy(actions[trajectory_idx])
                obs, reward, done, info = env.step(act)
                frames.append(write_skill_to_frame(obs, current_skill, infer_since_last_think, router_debug))
                wrist_frames.append(obs["robot0_eye_in_hand_image"][::-1, ::-1, :])
                trajectory_idx += 1

                if done:
                    break

                if trajectory_idx >= replan_interval:
                    actions, next_skill, infer_since_last_think, router_debug = infer_action_chunk(
                        policy,
                        obs,
                        task_description,
                        infer_since_last_think=infer_since_last_think,
                    )
                    if next_skill:
                        current_skill = next_skill
                    print(f"[Step {step_idx}] Atomic skill: {current_skill or '<empty>'}")

                    mediapy.write_image(
                        trial_image_dir / f"frame_agentview_task{task_id}_step{step_idx}.png",
                        np.copy(obs["agentview_image"][::-1, ::-1, :]),
                    )
                    mediapy.write_image(
                        trial_image_dir / f"frame_wrist_task{task_id}_step{step_idx}.png",
                        np.copy(obs["robot0_eye_in_hand_image"][::-1, ::-1, :]),
                    )

                    trajectory_idx = 0
                    replan_interval = max(1, len(actions) // REPLAN_FRACTION)

            trials_success.append(1 if done else 0)
            if not done:
                failture_records.append(
                    {
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "task_prompt": task_description,
                    }
                )

            if frames:
                mediapy.write_image(f"franka_libero_atomicVLA_task{task_id}_f0.png", frames[0])
                mediapy.write_video(
                    f"{TASK_SUITE_NAME}_benchmark/franka_libero_atomicVLA_task{task_id}_ep{episode_idx}.mp4",
                    frames,
                    fps=FRAME_RATE,
                    codec="mpeg4",
                    bps=1_000_000,
                )

            # if wrist_frames:
            #     mediapy.write_video(
            #         f"franka_libero_atomicVLA_wrist_task{task_id}_ep{episode_idx}.mp4",
            #         wrist_frames,
            #         fps=FRAME_RATE,
            #         codec="mpeg4",
            #         bps=1_000_000,
            #     )

    print(f"Trials success: {trials_success}")
    print(f"Failture records: {failture_records}")
    if trials_success:
        print(f"Trials success rate: {sum(trials_success) / len(trials_success)}")
    else:
        print("Trials success rate: no episodes were run.")
