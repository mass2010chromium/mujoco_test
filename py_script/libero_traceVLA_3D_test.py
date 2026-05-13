"""Run TraceVLA (`trace_vla` / `trace_vla_lora`) on LIBERO rollouts.

Closed-loop deployment that mirrors how the model was trained:

  * an off-the-shelf VLM (OpenRouter Gemini by default) supplies the skill plan and
    a per-skill *semantic target point* — exactly the same two artifacts that the
    annotation pipeline produced for training (`skill_annotations.json`,
    `skill_target_traces.json`).
  * the trained TraceVLA produces a 2-D pixel-space *trace* per skill (the trace
    expert, MoE-routed by the skill id), which we render as an overlay onto the
    base image (cyan polyline, matching training).
  * the action expert consumes the overlay-augmented base image and the wrist
    image and emits an action chunk; in the same forward we also obtain a
    completion-progress prediction.
  * when the predicted progress exceeds ``--completion-threshold`` for
    ``--consecutive-required`` consecutive checks, we advance to the next skill
    (re-querying Gemini for a fresh semantic point and re-sampling the trace).

The current end-effector pixel position is computed from `obs["robot0_eef_pos"]`
and the LIBERO camera matrix via the same projection used in the annotation
pipeline (`project_world_points_to_lerobot_image`), so the inpainting clamp
(`row 0 of x_t = current EE`) inside `sample_trace` is fed the same kind of
signal it was trained on.

The saved per-frame video overlays:

  * the cached trace polyline (cyan)
  * the current EE projection (green dot)
  * the current semantic target point (red dot)
  * the active skill text and the latest progress prediction (top-left text)

Usage examples:

    # Plan via Gemini (requires OPENROUTER_API_KEY in env).
    python py_script/libero_traceVLA_test.py \\
        --task-suite libero_10 --target-task-id 2 \\
        --checkpoint-dir /work/hdd/bgtb/$USER/checkpoints/trace_vla_lora/.../50000

    # Or, hardcode the plan and skip Gemini for plan generation:
    python py_script/libero_traceVLA_test.py \\
        --plan "1. OPEN(top drawer of the cabinet) 2. PICKUP_FROM(black bowl, table) 3. PLACE_IN(black bowl, top drawer) 4. CLOSE(top drawer)"
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

# ---------------------------------------------------------------------------
# Heavy imports — kept after the JAX env tweak above.
# ---------------------------------------------------------------------------

import mediapy
import numpy as np
import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi.models import trace_utils as _trace_utils
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config

sys.path.insert(0, os.path.dirname(__file__) + "/serving")
from inference_common import load_policy
import pi05_trace   # Registers trace models

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 224  # matches the LeRobot training data resolution

# LeRobot LIBERO image convention (EE projection lands in this convention; the
# image we feed the model is also in this convention after [::-1, ::-1, :]).
LEROBOT_IMAGE_CONVENTION = "robosuite_agentview_horizontal_flip"

def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution, "camera_depths": True}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _compute_camera_calibration(env: OffScreenRenderEnv, *, camera_name: str = "agentview",
                                  image_height: int = LIBERO_ENV_RESOLUTION,
                                  image_width: int = LIBERO_ENV_RESOLUTION) -> dict[str, Any]:
    """Pull the world->camera 4x4 from the live LIBERO env.

    The robosuite OffScreenRenderEnv exposes `env.sim`; we use the same
    `get_camera_transform_matrix` that the annotation pipeline uses, so EE
    projections match the convention the trace expert was trained on.
    """
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    sim = env.sim
    transform = get_camera_transform_matrix(sim, camera_name, int(image_height), int(image_width))
    return {
        "world_to_camera_transform": np.asarray(transform, dtype=float),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "image_convention": LEROBOT_IMAGE_CONVENTION,
    }

def _transform_camera_xy_to_lerobot_xy(xy: np.ndarray, *, image_width: int, image_height: int) -> np.ndarray:
    """Apply the LeRobot LIBERO horizontal-flip convention to camera-XY pixel coords."""
    pts = np.asarray(xy, dtype=float)
    out = np.column_stack([(int(image_width) - 1) - pts[:, 0], pts[:, 1]])
    return out

def project_ee_world_to_pixel(ee_world_xyz: np.ndarray, calib: dict[str, Any]) -> tuple[int, int, float]:
    """3D world EE position -> (x_pixel, y_pixel, depth) in LeRobot LIBERO image convention.

    Mirrors `project_world_points_to_lerobot_image` from the annotation pipeline
    inline (no external dependency). Returns x, y in image pixel coords, plus the
    camera-frame depth (z) so callers can sanity-check that the point is in
    front of the camera.
    """
    W = int(calib["image_width"])
    H = int(calib["image_height"])
    transform = np.asarray(calib["world_to_camera_transform"], dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"world_to_camera_transform must have shape (4,4), got {transform.shape}.")

    pts = np.asarray(ee_world_xyz, dtype=float).reshape(1, 3)
    homogeneous = np.concatenate([pts, np.ones((1, 1), dtype=float)], axis=1)
    projected = (transform @ homogeneous.T).T              # (1, 4)
    depth = float(projected[0, 2])
    if not np.isfinite(depth) or depth <= 0:
        # Fall back to image center if the EE is behind the camera or projection failed.
        return W // 2, H // 2, depth
    cam_xy = projected[:, :2] / projected[:, 2:3]         # (1, 2)
    image_xy = _transform_camera_xy_to_lerobot_xy(cam_xy, image_width=W, image_height=H)[0]
    x = int(round(np.clip(image_xy[0], 0, W - 1)))
    y = int(round(np.clip(image_xy[1], 0, H - 1)))
    return x, y, depth


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TraceVLA LIBERO inference + benchmark.")
    p.add_argument("--model-name", default="trace_vla_3d_lora",
                    help="TrainConfig name. Should be a trace policy")
    p.add_argument("--task-suite", default="libero_spatial",
                    help="LIBERO task suite name (libero_spatial / libero_object / libero_goal / libero_10 / libero_90).")
    p.add_argument("--target-task-id", type=int, default=None,
                    help="If set, only run this single task id within the suite.")
    p.add_argument("--trials-per-task", type=int, default=10)
    p.add_argument("--episode-length", type=int, default=800,
                    help="Hard cap on env steps per episode.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--frame-rate", type=int, default=20, help="Output video FPS.")
    p.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("trial_imgs_traceVLA"),
                    help="Where to save individual debug frames + per-trial mp4s.")

    # Plan source.
    p.add_argument("--plan", default="",
                    help='If set, use this plan string instead of querying Gemini, e.g.: '
                         '"1. PICKUP_FROM(black bowl, table) 2. PLACE_IN(black bowl, top drawer)"')

    p.add_argument("--actions-per-chunk", type=int, default=5,
                    help="Number of env steps to consume per action chunk before calling the model again. "
                         "Defaults to action_horizon // 2 = 5 for the standard trace_vla configs.")

    # Misc.
    p.add_argument("--jax-mem-fraction", type=float, default=0.6,
                    help="XLA_PYTHON_CLIENT_MEM_FRACTION (set before JAX init).")
    p.add_argument("--video-codec", default="mpeg4",
                    help="mediapy codec for output video.")
    p.add_argument("--video-bps", type=int, default=1_000_000)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_episode(task_id, episode_idx, *, env, calib: dict[str, Any], policy, task_description: str,
                  args: argparse.Namespace) -> tuple[bool, list[np.ndarray]]:
    """Run one episode of (potentially multi-skill) rollout. Returns (env_done, frames).

    Per-skill cycle:
      1. Query Gemini for a semantic target on the current scene.
      2. Re-plan trace (sample_trace) — repeat every `--replan-every-chunks` action chunks.
      3. Action loop: every action chunk, optionally run predict_completion FIRST to
         decide whether to abort the skill, then run sample_actions_and_completion to
         get the chunk + a fresh progress estimate. Execute the chunk in env, save
         per-step debug frames.
      4. Advance to the next skill when the abort trigger fires.
    """
    frames: list[np.ndarray] = []

    # Settle the env (matches the AtomicVLA reference).
    obs = None
    for _ in range(20):
        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)

    obs['ee_position'] = project_ee_world_to_pixel(obs['robot0_eef_pos'], calib)
    policy.initialize(obs, task_description)
    policy._save_video_frames = frames   # Hijack their pointer
    vla_output = policy.run_vla(obs)
    actions = vla_output['actions']

    print(f"[Task {task_id} ep {episode_idx}] Gemini plan: {policy._plan_raw}", flush=True)

    trajectory_idx = 0
    for total_steps in range(args.episode_length):
        act = np.copy(actions[trajectory_idx])
        obs, reward, done, info = env.step(act)
        obs['ee_position'] = project_ee_world_to_pixel(obs['robot0_eef_pos'], calib)
        policy.record_video_frame(obs)
        if done:
            print(f"    env reports done at step {total_steps}", flush=True)
            break

        trajectory_idx += 1
        if trajectory_idx == args.actions_per_chunk:
            vla_output = policy.run_vla(obs)
            actions = vla_output['actions']
            trajectory_idx = 0

    return done, frames


def main() -> None:
    args = _parse_args()
    if abs(args.jax_mem_fraction - 0.6) > 1e-6:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{args.jax_mem_fraction:.2f}"

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite not in benchmark_dict:
        raise SystemExit(f"Unknown task suite: {args.task_suite!r}. Available: {list(benchmark_dict)!r}")
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TraceVLA policy ({args.model_name})...", end="", flush=True)
    policy = load_policy(args.model_name)
    print(" Done.", flush=True)

    overall_success: list[int] = []
    failure_records: list[dict[str, Any]] = []

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        if args.target_task_id is not None and task_id != args.target_task_id:
            continue
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        calib = _compute_camera_calibration(env)
        print(f"[Task {task_id}] '{task_description}'", flush=True)

        for episode_idx in tqdm.tqdm(range(args.trials_per_task), leave=False):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            initial_scene_lerobot = obs["agentview_image"][::-1, ::-1, :]

            done, frames = run_episode(
                task_id, episode_idx,
                env=env, calib=calib, policy=policy, task_description=task_description,
                args=args
            )
            overall_success.append(1 if done else 0)
            if not done:
                failure_records.append({"task_id": task_id, "episode_idx": episode_idx,
                                          "task_prompt": task_description, "plan": policy._plan_raw})

            if frames:
                video_path = args.output_dir / f"{args.task_suite}_traceVLA_task{task_id}_ep{episode_idx}.mp4"
                still_path = args.output_dir / f"{args.task_suite}_traceVLA_task{task_id}_ep{episode_idx}_first.png"
                mediapy.write_image(still_path, frames[0])
                mediapy.write_video(video_path, frames, fps=args.frame_rate,
                                      codec=args.video_codec, bps=args.video_bps)
                print(f"  -> wrote {video_path} ({len(frames)} frames)", flush=True)

        env.close() if hasattr(env, "close") else None

    print(f"Trials success: {overall_success}", flush=True)
    if overall_success:
        print(f"Success rate: {sum(overall_success) / len(overall_success):.3f}", flush=True)
    if failure_records:
        print(f"Failures ({len(failure_records)}): {failure_records}", flush=True)


if __name__ == "__main__":
    main()
