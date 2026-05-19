"""
Run the LIBERO-finetuned vanilla pi05 model (`pi05_libero_100`) on LIBERO rollouts.

Mirrors the benchmark structure of ``libero_atomicVLA_test.py`` /
``libero_traceVLA_test.py`` but uses the plain :class:`openpi.policies.policy.Policy`
produced by ``create_trained_policy`` — no AtomicVLA reasoning, no router, no overlays.

Data flow (matches the training contract of `pi05_libero_100`):
- prompt = task instruction (LIBERO env's task language)
- observation/image: agentview, after LeRobot 180-deg rotation (``[::-1, ::-1, :]``)
- observation/wrist_image: wrist eye-in-hand, same flip
- observation/state: 8-D = (eef_xyz | eef_axisangle | gripper_qpos[2])

The policy returns an action chunk of shape ``(action_horizon, 7)``
(``LiberoOutputs`` trims to the first 7 channels).

Result logging + resume support
-------------------------------
After every episode the script writes (atomically) a single ``results.json``
file under ``--output-dir``. The file is the source of truth for which
``(task_id, episode_idx)`` pairs have been benchmarked. Re-running with the
same ``--output-dir`` resumes from the last logged state — already-logged
``(task_id, episode_idx)`` pairs are skipped, new ones append to the file.
Pass ``--overwrite-results`` to start fresh.

Default checkpoint:
``/work/hdd/bgtb/zhong2/checkpoints/pi05_libero_100/pi05_libero_100/99999``
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys


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


# Defaults — used as argparse defaults; the actual runtime values come from CLI.
DEFAULT_MODEL_NAME = "pi05_libero_100"
DEFAULT_TASK_SUITE_NAME = "libero_spatial"
# Set this to a specific step directory if auto-discovery doesn't find your run.
DEFAULT_CHECKPOINT_DIR = "/work/hdd/bgtb/zhong2/checkpoints/pi05_libero_100/pi05_libero_100/99999"
DEFAULT_TRIALS_PER_TASK = 10
DEFAULT_SEED = 7
DEFAULT_EPISODE_LENGTH = 800
DEFAULT_FRAME_RATE = 20
DEFAULT_LIBERO_ENV_RESOLUTION = 224
DEFAULT_REPLAN_FRACTION = 2  # re-query policy every action_horizon // REPLAN_FRACTION env steps
DEFAULT_OUTPUT_DIR = pathlib.Path("trial_imgs_pi05FT")
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


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


# ---------------------------------------------------------------------------
# Result persistence + resume support
# ---------------------------------------------------------------------------

RESULTS_FILENAME = "results.json"


def _results_json_path(output_dir: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(output_dir) / RESULTS_FILENAME


def _summarize_episodes(episodes: list[dict]) -> dict:
    n = len(episodes)
    successes = sum(1 for ep in episodes if bool(ep.get("success")))
    return {
        "total": n,
        "successes": successes,
        "failures": n - successes,
        "success_rate": (successes / n) if n > 0 else 0.0,
    }


def _write_results_atomic(
    *,
    output_dir: pathlib.Path,
    task_suite: str,
    model_name: str,
    checkpoint_dir: pathlib.Path | str,
    args: argparse.Namespace,
    completed_episodes: list[dict],
) -> pathlib.Path:
    """Atomic write: tmp file + rename. Kills mid-write never leave a corrupt
    file on disk that would later break resume. The JSON is the source of truth
    for which (task_id, episode_idx) pairs are done; videos / frames are aux."""
    path = _results_json_path(output_dir)
    document = {
        "task_suite": task_suite,
        "model_name": model_name,
        "checkpoint_dir": str(checkpoint_dir),
        "args": {
            "trials_per_task": int(args.trials_per_task),
            "episode_length": int(args.episode_length),
            "seed": int(args.seed),
            "target_task_id": (None if args.target_task_id is None else int(args.target_task_id)),
            "replan_fraction": int(args.replan_fraction),
            "libero_env_resolution": int(args.libero_env_resolution),
            "frame_rate": int(args.frame_rate),
        },
        "summary": _summarize_episodes(completed_episodes),
        "completed_episodes": completed_episodes,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def _load_resume_state(
    output_dir: pathlib.Path,
    *,
    expected_task_suite: str,
    overwrite: bool,
) -> tuple[list[dict], set[tuple[int, int]]]:
    """Return (completed_episodes, done_keys). Both empty if there's nothing to
    resume. If the file is present but the task_suite disagrees (or parse fails)
    we raise SystemExit rather than silently overwrite. Use --overwrite-results
    to discard an existing file."""
    path = _results_json_path(output_dir)
    if overwrite:
        if path.exists():
            print(f"[results] --overwrite-results set; ignoring existing {path}", flush=True)
        return [], set()
    if not path.exists():
        return [], set()
    try:
        doc = json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(
            f"[resume] failed to parse {path}: {exc}. "
            f"Either fix the file or pass --overwrite-results to start fresh."
        )
    file_suite = doc.get("task_suite")
    if file_suite != expected_task_suite:
        raise SystemExit(
            f"[resume] {path} has task_suite={file_suite!r} but this run is "
            f"task_suite={expected_task_suite!r}. Either point --output-dir elsewhere "
            f"or pass --overwrite-results to discard the existing results."
        )
    completed = list(doc.get("completed_episodes", []) or [])
    done_keys: set[tuple[int, int]] = {
        (int(ep["task_id"]), int(ep["episode_idx"])) for ep in completed
    }
    summary = _summarize_episodes(completed)
    print(
        f"[resume] loaded {path}: {summary['total']} episodes already done "
        f"({summary['successes']} success / {summary['failures']} fail; "
        f"running success rate {summary['success_rate']:.3f})",
        flush=True,
    )
    return completed, done_keys


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pi05 LIBERO inference + benchmark.")
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                    help="TrainConfig name (default: pi05_libero_100).")
    p.add_argument("--task-suite", default=DEFAULT_TASK_SUITE_NAME,
                    help="LIBERO task suite name (libero_spatial / libero_object / "
                         "libero_goal / libero_10 / libero_90).")
    p.add_argument("--target-task-id", type=int, default=None,
                    help="If set, only run this single task id within the suite.")
    p.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR,
                    help="Path to a trained pi05 checkpoint step dir (containing params/). "
                         "Pass an empty string to fall back to auto-discovery.")
    p.add_argument("--assets-dir", default=None,
                    help="Optional override for the assets dir (norm_stats). "
                         "Defaults to pace/openpi/assets/<model_name>.")
    p.add_argument("--trials-per-task", type=int, default=DEFAULT_TRIALS_PER_TASK)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--episode-length", type=int, default=DEFAULT_EPISODE_LENGTH,
                    help="Hard cap on env steps per episode.")
    p.add_argument("--frame-rate", type=int, default=DEFAULT_FRAME_RATE,
                    help="Output video FPS.")
    p.add_argument("--libero-env-resolution", type=int, default=DEFAULT_LIBERO_ENV_RESOLUTION,
                    help="Camera resolution (square) the LIBERO env renders at.")
    p.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR,
                    help="Where to save per-trial mp4s, first-frame stills, and results.json.")
    p.add_argument("--replan-fraction", type=int, default=DEFAULT_REPLAN_FRACTION,
                    help="Re-query the policy after consuming len(actions)//N of the current chunk (default 2).")
    p.add_argument("--video-codec", default="mpeg4")
    p.add_argument("--video-bps", type=int, default=1_000_000)
    p.add_argument("--no-video", action="store_true",
                    help="Skip writing the per-trial .mp4 file. The first-frame still PNG "
                         "is still written (cheap, useful for sanity-checking). By default "
                         "the mp4 is always written.")
    p.add_argument("--overwrite-results", action="store_true",
                    help="Ignore any existing results.json in --output-dir and start fresh. "
                         "Without this, the script resumes from the last logged state — "
                         "(task_id, episode_idx) pairs already in results.json are skipped, "
                         "and new episodes append to the same file.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main inference loop — preserves the original per-task / per-episode rollout
# logic exactly; CLI flags + resume + per-episode JSON write are layered around it.
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    print(
        f"TASK_SUITE_NAME: {args.task_suite}; "
        f"TRIALS_PER_TASK: {args.trials_per_task}; "
        f"TARGET_TASK_ID: {args.target_task_id}",
        flush=True,
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite not in benchmark_dict:
        raise SystemExit(
            f"Unknown task suite: {args.task_suite!r}. Available: {list(benchmark_dict)!r}"
        )
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Resume BEFORE the slow model load so suite-mismatch / corrupt-file fails fast.
    completed_episodes, done_keys = _load_resume_state(
        args.output_dir,
        expected_task_suite=args.task_suite,
        overwrite=args.overwrite_results,
    )

    if args.assets_dir is not None:
        assets_dir = pathlib.Path(args.assets_dir)
    else:
        openpi_root = pathlib.Path(__file__).resolve().parent.parent / "pace" / "openpi"
        assets_dir = openpi_root / "assets" / args.model_name

    # An empty --checkpoint-dir means "fall back to auto-discovery".
    checkpoint_dir_arg = args.checkpoint_dir if args.checkpoint_dir else None

    print("Loading pi05_libero_100 policy...", end="", flush=True)
    policy, checkpoint_dir = create_pi05FT_policy(
        args.model_name,
        checkpoint_dir=checkpoint_dir_arg,
        assets_dir=assets_dir,
    )
    print("Done.", flush=True)
    print(f"Using checkpoint: {checkpoint_dir}", flush=True)

    # Rebuild the convenience lists from resume state so end-of-run summaries
    # and per-episode appends include episodes carried over from previous runs.
    trials_success: list[int] = [1 if ep.get("success") else 0 for ep in completed_episodes]
    failure_records: list[dict] = [
        {k: ep[k] for k in ("task_id", "episode_idx", "task_prompt") if k in ep}
        for ep in completed_episodes if not ep.get("success")
    ]

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        if args.target_task_id is not None and task_id != args.target_task_id:
            continue

        # Task-level fast-skip: skip env setup entirely if all episodes are done.
        remaining_episode_indices = [
            ep_idx for ep_idx in range(args.trials_per_task)
            if (task_id, ep_idx) not in done_keys
        ]
        if not remaining_episode_indices:
            print(
                f"[Task {task_id}] all {args.trials_per_task} episodes already done -> skip",
                flush=True,
            )
            continue

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, args.libero_env_resolution, args.seed)
        print(
            f"[Task {task_id}] '{task_description}'  (remaining episodes: {remaining_episode_indices})",
            flush=True,
        )

        for episode_idx in tqdm.tqdm(range(args.trials_per_task), leave=False):
            # Resume: skip individual episodes already logged.
            if (task_id, episode_idx) in done_keys:
                print(
                    f"[Task {task_id} ep {episode_idx}] already in results.json -> skip",
                    flush=True,
                )
                continue

            # -------------------------------------------------------------
            # The original per-episode rollout body (unchanged in logic):
            #   reset env, settle, query initial action chunk, loop until
            #   done / episode_length, replan every action_horizon // K.
            # -------------------------------------------------------------
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            # Settle the env (matches the AtomicVLA / TraceVLA reference scripts).
            for _ in range(20):
                obs, _reward, _done, _info = env.step(LIBERO_DUMMY_ACTION)

            # Initial action chunk.
            actions = np.asarray(policy.infer(prompt_from_obs(obs, task_description))["actions"])
            if actions.ndim != 2:
                raise ValueError(f"Expected an action chunk of shape [horizon, dim], got {actions.shape!r}")

            replan_interval = max(1, len(actions) // int(args.replan_fraction))
            trajectory_idx = 0
            done = False
            frames = []
            wrist_frames = []

            for step_idx in range(args.episode_length):
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
                    replan_interval = max(1, len(actions) // int(args.replan_fraction))

            trials_success.append(1 if done else 0)
            if not done:
                failure_records.append({
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "task_prompt": task_description,
                })

            if frames:
                first_path = args.output_dir / f"franka_libero_pi05FT_task{task_id}_f0.png"
                video_path = args.output_dir / f"franka_libero_pi05FT_task{task_id}_ep{episode_idx}.mp4"
                mediapy.write_image(first_path, frames[0])
                if args.no_video:
                    print(f"  -> wrote {first_path} (--no-video set; skipped mp4)", flush=True)
                else:
                    mediapy.write_video(
                        video_path, frames, fps=args.frame_rate,
                        codec=args.video_codec, bps=args.video_bps,
                    )
                    print(f"  -> wrote {video_path} ({len(frames)} frames)", flush=True)

            # Persist results AFTER per-episode artifacts are on disk, so any
            # episode logged in completed_episodes is guaranteed to have its
            # companion files. The JSON is the source of truth for resume.
            episode_record = {
                "task_id": int(task_id),
                "episode_idx": int(episode_idx),
                "success": bool(done),
                "task_prompt": task_description,
            }
            completed_episodes.append(episode_record)
            done_keys.add((int(task_id), int(episode_idx)))
            result_path = _write_results_atomic(
                output_dir=args.output_dir,
                task_suite=args.task_suite,
                model_name=args.model_name,
                checkpoint_dir=checkpoint_dir,
                args=args,
                completed_episodes=completed_episodes,
            )
            running = _summarize_episodes(completed_episodes)
            print(
                f"  -> {result_path}: {running['total']} episodes done overall, "
                f"running success rate {running['success_rate']:.3f} "
                f"({running['successes']}/{running['total']})",
                flush=True,
            )

        if hasattr(env, "close"):
            env.close()

    # Final summary — read from the JSON-on-disk source of truth.
    final = _summarize_episodes(completed_episodes)
    print(f"Trials success: {trials_success}", flush=True)
    print(f"Failure records: {failure_records}", flush=True)
    print(
        f"Trials success rate: {final['success_rate']:.3f}  ({final['successes']}/{final['total']})",
        flush=True,
    )
    print(f"Task Suite: {args.task_suite}", flush=True)
    print(f"Total trials: {final['total']}", flush=True)
    print(f"Results JSON: {_results_json_path(args.output_dir)}", flush=True)


if __name__ == "__main__":
    main()
