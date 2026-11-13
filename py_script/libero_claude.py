"""Let Claude Code try to solve LIBERO tasks, one delta command per turn.

Each turn the environment:
  1. renders the current agent-view and wrist images to PNG files on disk,
  2. invokes ``claude -p`` with the task string, the robot state and the image paths
     (the agent ``Read``s the images itself — each prompt references only that turn's
     images; every turn's PNGs are kept on disk unless ``--latest-images-only``),
  3. receives a structured 7-number command back (6-DOF pose delta + gripper velocity),
  4. executes it with a pose feedback loop for ``--steps-per-command`` env steps,
and then re-prompts. Conversation persistence across turns uses Claude Code's own
session store: the first turn runs ``claude -p --session-id <uuid>``, later turns run
``claude -p --resume <uuid>`` (re-using ``--session-id`` on an existing session errors).

Usage:
    mujoco_playground/.venv/bin/python py_script/libero_claude.py \\
        --task-suite libero_10 --target-task-id 0 --trials-per-task 1 --model sonnet

    # Headless sanity check of the frame conventions + feedback loop (no Claude, no rendering):
    mujoco_playground/.venv/bin/python py_script/libero_claude.py --self-test

Interactive alternative: ``libero_mcp.py`` exposes the same robot interface as an MCP
server, so an interactive Claude Code session can drive the robot directly.

The control frame, image convention, feedback-loop execution and gripper convention are
documented in ``libero_claude_core.py`` (shared with the MCP server).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
import uuid
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from libero_claude_core import (  # noqa: E402
    DEFAULT_MAX_ROTATION_DEG,
    DEFAULT_MAX_TRANSLATION_CM,
    DEFAULT_STEPS_PER_COMMAND,
    LIBERO_DUMMY_ACTION,
    NUM_SETTLE_STEPS,
    Command,
    LevelCameraFrame,
    LiberoEpisode,
    control_conventions_text,
    eef_pose,
    execute_command,
    format_state,
    load_task,
    make_env,
)

RESULTS_FILENAME = "results.json"


# ---------------------------------------------------------------------------
# Prompts + Claude Code invocation
# ---------------------------------------------------------------------------

COMMAND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "reasoning": {"type": "string", "description": "Brief reasoning about the scene and the next move."},
        "dx": {"type": "number", "description": "cm, + = image right"},
        "dy": {"type": "number", "description": "cm, + = straight down (gravity)"},
        "dz": {"type": "number", "description": "cm, + = into the image / away from the camera"},
        "rx": {"type": "number", "description": "deg, rotation about +x (right-hand rule)"},
        "ry": {"type": "number", "description": "deg, rotation about +y (down axis)"},
        "rz": {"type": "number", "description": "deg, rotation about +z (into-image axis)"},
        "gripper": {"type": "number", "minimum": -1, "maximum": 1,
                    "description": "+1 close, -1 open, 0 hold"},
    },
    "required": ["reasoning", "dx", "dy", "dz", "rx", "ry", "rz", "gripper"],
}


def build_system_prompt(*, steps_per_command: int, max_cm: float, max_deg: float, max_turns: int,
                        drop_line: bool = False) -> str:
    conventions = control_conventions_text(steps_per_command=steps_per_command, max_cm=max_cm, max_deg=max_deg,
                                           drop_line=drop_line)
    return f"""You are controlling a Franka Panda robot arm with a parallel-jaw gripper in a simulated tabletop scene (the LIBERO benchmark). Your goal is to complete the given task.

Every turn you are given the task, the robot state, the result of your previous command, and the paths of two freshly rendered images. Always use the Read tool to look at BOTH images before deciding. Each command runs for exactly {steps_per_command} simulation steps.

{conventions}

You have at most {max_turns} turns. The episode ends automatically when the task is completed. Reply only with the structured command."""


def build_turn_prompt(*, turn: int, max_turns: int, task: str, agentview_path: pathlib.Path,
                      wrist_path: pathlib.Path, state: dict[str, Any], last_report: str | None) -> str:
    lines = [
        f"Turn {turn + 1} of {max_turns}.",
        f"Task: {task}",
        "",
        "Images (Read both):",
        f"- agentview: {agentview_path}",
        f"- wrist: {wrist_path}",
        "",
        "Robot state:",
        format_state(state),
        "",
    ]
    if last_report is None:
        lines.append("This is the first turn.")
    else:
        lines.append("Result of your previous command:")
        lines.append(last_report)
    lines.append("")
    lines.append("Give the next command.")
    return "\n".join(lines)


class ClaudeSession:
    """Persists one Claude Code conversation across turns via --session-id / --resume."""

    def __init__(self, args: argparse.Namespace, system_prompt: str, cwd: pathlib.Path):
        self.args = args
        self.system_prompt = system_prompt
        self.cwd = cwd
        self.session_id = str(uuid.uuid4())
        self.started = False
        self.total_cost_usd = 0.0

    def _cmd(self) -> list[str]:
        a = self.args
        cmd = [
            a.claude_bin, "-p",
            "--model", a.model,
            "--output-format", "json",
            "--json-schema", json.dumps(COMMAND_SCHEMA),
            "--system-prompt", self.system_prompt,
            # Only Read is available, and it's pre-approved. Note: --tools/--allowed-tools are
            # variadic, which is why the prompt goes in on stdin rather than as an argument.
            "--tools", "Read",
            "--allowed-tools", "Read",
        ]
        cmd += ["--resume", self.session_id] if self.started else ["--session-id", self.session_id]
        if a.effort:
            cmd += ["--effort", a.effort]
        if a.bare:
            cmd.append("--bare")
        return cmd

    def ask(self, prompt: str) -> tuple[Command, dict[str, Any]]:
        """Send one turn; returns (command, raw result json). Raises RuntimeError on failure."""
        last_err = "unknown error"
        for attempt in range(1, self.args.claude_retries + 1):
            text = prompt if attempt == 1 else (
                prompt + f"\n\n(Previous reply could not be used: {last_err}. Reply with the structured command.)")
            t0 = time.time()
            try:
                proc = subprocess.run(self._cmd(), input=text, capture_output=True, text=True,
                                      cwd=self.cwd, timeout=self.args.claude_timeout)
            except subprocess.TimeoutExpired:
                last_err = f"timed out after {self.args.claude_timeout}s"
                print(f"    claude: {last_err} (attempt {attempt})", flush=True)
                continue
            dt = time.time() - t0
            try:
                result = json.loads(proc.stdout)
            except json.JSONDecodeError:
                last_err = f"non-JSON output (rc={proc.returncode}): {(proc.stderr or proc.stdout)[-300:]!r}"
                print(f"    claude: {last_err} (attempt {attempt})", flush=True)
                continue
            # The session exists on disk once any call has produced a result for it.
            if result.get("session_id") == self.session_id:
                self.started = True
            self.total_cost_usd += float(result.get("total_cost_usd") or 0.0)
            if result.get("is_error"):
                last_err = f"error result: {str(result.get('result'))[:300]!r}"
                print(f"    claude: {last_err} (attempt {attempt})", flush=True)
                continue
            so = result.get("structured_output")
            if not isinstance(so, dict):
                last_err = "missing structured_output"
                print(f"    claude: {last_err} (attempt {attempt})", flush=True)
                continue
            try:
                cmd = Command(**{k: float(so[k]) for k in ("dx", "dy", "dz", "rx", "ry", "rz", "gripper")})
            except (KeyError, TypeError, ValueError) as exc:
                last_err = f"bad command fields: {exc}"
                continue
            result["_latency_s"] = round(dt, 2)
            return cmd, result
        raise RuntimeError(f"Claude failed after {self.args.claude_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------

def run_episode(*, env, task_description: str, init_state, args, episode_dir: pathlib.Path) -> dict[str, Any]:
    ep = LiberoEpisode(env, task_description, init_state, episode_dir=episode_dir,
                       render_size=args.render_size, annotate=args.annotate,
                       record_video=not args.no_video, keep_all_images=not args.latest_images_only,
                       drop_line=args.annotate and args.drop_line)

    system_prompt = build_system_prompt(steps_per_command=args.steps_per_command,
                                        max_cm=args.max_translation_cm, max_deg=args.max_rotation_deg,
                                        max_turns=args.max_turns, drop_line=args.annotate and args.drop_line)
    session = ClaudeSession(args, system_prompt, cwd=ep.episode_dir)
    print(f"  session {session.session_id}", flush=True)

    last_report: str | None = None
    stop_reason = "max_turns"
    for turn in range(args.max_turns):
        paths = ep.save_observation()
        if paths is None:
            raise RuntimeError("Env has no camera observations; batch mode needs rendering.")
        agent_path, wrist_path = paths
        prompt = build_turn_prompt(turn=turn, max_turns=args.max_turns, task=task_description,
                                   agentview_path=agent_path, wrist_path=wrist_path,
                                   state=ep.state(), last_report=last_report)
        try:
            cmd_raw, result = session.ask(prompt)
        except RuntimeError as exc:
            print(f"  turn {turn}: {exc}", flush=True)
            stop_reason = "claude_error"
            break
        reasoning = str((result.get("structured_output") or {}).get("reasoning", ""))
        cmd, report, last_report = ep.move(
            cmd_raw, steps=args.steps_per_command, gain=args.feedback_gain,
            max_cm=args.max_translation_cm, max_deg=args.max_rotation_deg,
            log_extra={"reasoning": reasoning, "latency_s": result.get("_latency_s"),
                       "cost_usd_cumulative": round(session.total_cost_usd, 5)})
        print(f"  turn {turn}: {cmd.as_dict()}  ({result.get('_latency_s')}s, "
              f"${session.total_cost_usd:.3f} total)  {reasoning[:160]}", flush=True)

        if report.success:
            stop_reason = "success"
            break
        if args.max_cost_usd and session.total_cost_usd >= args.max_cost_usd:
            stop_reason = "cost_cap"
            print(f"  cost cap ${args.max_cost_usd} reached", flush=True)
            break

    ep.write_video(fps=args.frame_rate, codec=args.video_codec, bps=args.video_bps)
    return {
        "success": ep.success, "turns": ep.moves, "stop_reason": stop_reason,
        "session_id": session.session_id, "cost_usd": round(session.total_cost_usd, 5),
    }


# ---------------------------------------------------------------------------
# Results persistence (resume support; same scheme as libero_traceVLA_test.py)
# ---------------------------------------------------------------------------

def _summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(episodes)
    s = sum(1 for e in episodes if e.get("success"))
    return {"total": n, "successes": s, "failures": n - s, "success_rate": s / n if n else 0.0,
            "cost_usd": round(sum(float(e.get("cost_usd", 0.0)) for e in episodes), 4)}


def _write_results(path: pathlib.Path, *, task_suite: str, args, episodes: list[dict[str, Any]]) -> None:
    doc = {
        "task_suite": task_suite,
        "args": {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()},
        "summary": _summarize(episodes),
        "completed_episodes": episodes,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    tmp.replace(path)


def _load_results(path: pathlib.Path, *, task_suite: str, overwrite: bool) -> list[dict[str, Any]]:
    if overwrite or not path.exists():
        return []
    doc = json.loads(path.read_text())
    if doc.get("task_suite") != task_suite:
        raise SystemExit(f"{path} is for task_suite={doc.get('task_suite')!r}, not {task_suite!r}. "
                         f"Use another --output-dir or pass --overwrite-results.")
    eps = list(doc.get("completed_episodes") or [])
    print(f"[resume] {len(eps)} episodes already done: {_summarize(eps)}", flush=True)
    return eps


# ---------------------------------------------------------------------------
# Self-test (headless: no rendering, no Claude)
# ---------------------------------------------------------------------------

def self_test(args) -> None:
    task, init_states, _ = load_task(args.task_suite, args.target_task_id or 0)
    env = make_env(task, resolution=args.render_size, seed=args.seed, headless=True)
    env.reset()
    obs = env.set_init_state(init_states[0])
    for _ in range(NUM_SETTLE_STEPS):
        obs, *_ = env.step(LIBERO_DUMMY_ACTION)

    frame = LevelCameraFrame.from_sim(env.sim)
    print("Level frame axes in world (columns x, y, z):\n", frame.R_wl.round(3))
    _, R = eef_pose(env)
    print("approach dir (level frame):", frame.to_level(R[:, 2]).round(3), "(expect ~[0, 1, 0] = down)")

    tests = [
        ("right 5cm", Command(dx=5)),
        ("up 5cm", Command(dy=-5)),
        ("into 5cm", Command(dz=5)),
        ("twist +30deg about y (down)", Command(ry=30)),
        ("twist back", Command(ry=-30)),
        ("close gripper", Command(gripper=1)),
        ("open gripper", Command(gripper=-1)),
        ("diag 6cm right + 3cm down", Command(dx=6, dy=3)),
    ]
    for name, cmd in tests:
        obs, rep = execute_command(env, obs, frame, cmd, steps=args.steps_per_command, gain=args.feedback_gain)
        print(f"\n== {name}\n{rep.summary()}")
    env.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Claude Code as a LIBERO policy (turn-based delta commands).")
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--target-task-id", type=int, default=None, help="Only run this task id.")
    p.add_argument("--trials-per-task", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("trial_imgs_claude"))
    p.add_argument("--overwrite-results", action="store_true")

    # Turn / execution.
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--steps-per-command", type=int, default=DEFAULT_STEPS_PER_COMMAND,
                   help="Env steps per command (20 Hz control -> 1 s). >=15 lets the gripper fully open/close.")
    p.add_argument("--feedback-gain", type=float, default=1.0,
                   help="Proportional gain of the pose feedback loop (1.0 = send the full remaining error).")
    p.add_argument("--max-translation-cm", type=float, default=DEFAULT_MAX_TRANSLATION_CM)
    p.add_argument("--max-rotation-deg", type=float, default=DEFAULT_MAX_ROTATION_DEG)

    # Observation.
    p.add_argument("--render-size", type=int, default=512, help="Camera render resolution (square).")
    p.add_argument("--no-annotate", dest="annotate", action="store_false",
                   help="Don't draw the gripper marker / command axes / wrist grasp-point overlays.")
    p.add_argument("--drop-line", action="store_true",
                   help="Overlay a vertical drop line from the gripper centre to the table (both images) and "
                        "report the gripper's height above the table. Uses the static scene geometry below the "
                        "gripper, i.e. a small amount of privileged information. Off by default.")
    p.add_argument("--latest-images-only", action="store_true",
                   help="Delete the previous turn's PNGs each turn (default: keep every turn's images).")

    # Claude Code.
    p.add_argument("--claude-bin", default="claude")
    p.add_argument("--model", default="sonnet")
    p.add_argument("--effort", default=None, choices=[None, "low", "medium", "high", "xhigh", "max"])
    p.add_argument("--bare", action="store_true",
                   help="Pass --bare (faster startup, no user hooks/CLAUDE.md). Requires ANTHROPIC_API_KEY.")
    p.add_argument("--claude-timeout", type=float, default=300.0, help="Seconds per claude call.")
    p.add_argument("--claude-retries", type=int, default=3)
    p.add_argument("--max-cost-usd", type=float, default=None, help="Per-episode cost cap.")

    # Video.
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--frame-rate", type=int, default=20)
    p.add_argument("--video-codec", default="mpeg4")
    p.add_argument("--video-bps", type=int, default=2_000_000)

    p.add_argument("--self-test", action="store_true",
                   help="Headless check of the control frame + feedback loop (no rendering, no Claude).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_test:
        self_test(args)
        return

    import tqdm

    try:
        _, _, n_tasks = load_task(args.task_suite, 0)
    except ValueError as exc:
        raise SystemExit(str(exc))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / RESULTS_FILENAME
    episodes = _load_results(results_path, task_suite=args.task_suite, overwrite=args.overwrite_results)
    done_keys = {(int(e["task_id"]), int(e["episode_idx"])) for e in episodes}

    for task_id in range(n_tasks):
        if args.target_task_id is not None and task_id != args.target_task_id:
            continue
        todo = [i for i in range(args.trials_per_task) if (task_id, i) not in done_keys]
        if not todo:
            continue
        task, init_states, _ = load_task(args.task_suite, task_id)
        env = make_env(task, resolution=args.render_size, seed=args.seed)
        print(f"[Task {task_id}] {task.language!r}  episodes {todo}", flush=True)

        for ep in tqdm.tqdm(todo, leave=False):
            ep_dir = (args.output_dir / f"{args.task_suite}_task{task_id}_ep{ep}").resolve()
            ep_dir.mkdir(parents=True, exist_ok=True)
            out = run_episode(env=env, task_description=task.language, init_state=init_states[ep],
                              args=args, episode_dir=ep_dir)
            episodes.append({"task_id": task_id, "episode_idx": ep, "task_prompt": task.language,
                             "episode_dir": str(ep_dir), **out})
            done_keys.add((task_id, ep))
            _write_results(results_path, task_suite=args.task_suite, args=args, episodes=episodes)
            s = _summarize(episodes)
            print(f"  -> {'SUCCESS' if out['success'] else 'fail'} ({out['stop_reason']}, {out['turns']} turns, "
                  f"${out['cost_usd']:.3f}); running {s['successes']}/{s['total']} = {s['success_rate']:.3f}",
                  flush=True)

            break
        env.close()
        
        break

    print(f"Final: {_summarize(episodes)}\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
