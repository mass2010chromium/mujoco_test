"""MCP server exposing a LIBERO robot to an interactive Claude Code session.

The interactive session *is* the agent: it calls ``reset`` to start an episode, looks at the
images returned by each tool call, and calls ``move`` with 6-DOF deltas + a gripper command.
The control interface (level camera frame, feedback-loop execution, gripper convention) is
identical to the batch entrypoint ``libero_claude.py``; both share ``libero_claude_core.py``.

Register with Claude Code (from the repo root; local scope = this project, only you):

    claude mcp add libero --scope local -e MUJOCO_GL=egl -- \\
        $PWD/mujoco_playground/.venv/bin/python $PWD/py_script/libero_mcp.py \\
        --output-dir $PWD/py_script/trial_imgs_claude_mcp

then, in a new ``claude`` session: "Use the libero tools to solve libero_10 task 0."
Server flags (after the script path) set defaults such as --render-size / --steps-per-command;
``--headless`` runs without rendering (state only) for testing on machines without EGL.

Tools: list_tasks, reset, observe, move_relative, move_absolute, finish.

Files per episode (under --output-dir/<suite>_task<id>_init<k>_<timestamp>/):
  obs/turn_XXX_{agentview,wrist}.png   every observation (after reset and after each move)
  turns.jsonl                          one line per move (state, command, report, reasoning)
  video.mp4, first_frame.png           written by ``finish`` (or when a new episode is reset)
and --output-dir/episodes.jsonl gets one summary line per finished episode.

Threading: MuJoCo's EGL rendering context is bound to the thread that created it, while the
MCP SDK would run sync tools on arbitrary worker threads — so every env call is funnelled
through a single dedicated thread.

stdout is the MCP wire, and LIBERO/robosuite print to stdout (e.g. "[info] using task
orders ..."), which the SDK's own fd diversion did not catch in testing. So before anything
else is imported we keep a private duplicate of fd 1 for the wire, point fd 1 and
``sys.stdout`` at stderr, and hand the private fd to ``stdio_server`` explicitly.
"""
from __future__ import annotations

import os
import sys

_WIRE_FD = os.dup(1)          # the real stdout: reserved for MCP JSON-RPC
os.dup2(2, 1)                 # C-level / subprocess writes to fd 1 -> stderr
sys.stdout = sys.stderr       # Python-level print() -> stderr

import argparse  # noqa: E402
import asyncio
import concurrent.futures
import datetime
import io
import json
import pathlib
from typing import Annotated, Any

import anyio
import numpy as np
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image
from mcp.server.stdio import stdio_server
from mcp_types import TextContent
from pydantic import Field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from libero_claude_core import (  # noqa: E402
    DEFAULT_MAX_ROTATION_DEG,
    DEFAULT_MAX_TRANSLATION_CM,
    DEFAULT_STEPS_PER_COMMAND,
    Command,
    LiberoEpisode,
    control_conventions_text,
    format_state,
    load_task,
    make_env,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LIBERO MCP server for interactive Claude Code sessions.")
    p.add_argument("--output-dir", type=pathlib.Path,
                   default=pathlib.Path(__file__).resolve().parent / "trial_imgs_claude_mcp")
    p.add_argument("--render-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--steps-per-command", type=int, default=DEFAULT_STEPS_PER_COMMAND)
    p.add_argument("--feedback-gain", type=float, default=1.0)
    p.add_argument("--max-translation-cm", type=float, default=DEFAULT_MAX_TRANSLATION_CM)
    p.add_argument("--max-rotation-deg", type=float, default=DEFAULT_MAX_ROTATION_DEG)
    p.add_argument("--max-steps-per-command", type=int, default=100,
                   help="Upper bound for move_relative's per-call `steps` override.")
    p.add_argument("--max-steps-absolute", type=int, default=200,
                   help="Default (and upper bound) timeout in sim steps for move_absolute.")
    p.add_argument("--no-annotate", dest="annotate", action="store_false")
    p.add_argument("--drop-line", action="store_true",
                   help="Overlay a vertical drop line from the gripper centre to the table (both images) and "
                        "report the gripper's height above the table. Uses the static scene geometry below the "
                        "gripper, i.e. a small amount of privileged information. Off by default.")
    p.add_argument("--no-inline-images", dest="inline_images", action="store_false",
                   help="Return image file paths instead of inline images (agent must Read them).")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--frame-rate", type=int, default=20)
    p.add_argument("--headless", action="store_true",
                   help="No rendering / camera observations (state only). For testing without EGL.")
    return p.parse_args(argv)


ARGS = _parse_args()


# ---------------------------------------------------------------------------
# Robot state (touched only from the env thread)
# ---------------------------------------------------------------------------

class _Robot:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.env = None
        self.env_key: tuple[str, int] | None = None
        self.task_suite: str | None = None
        self.task_id: int | None = None
        self.episode: LiberoEpisode | None = None
        self.finished = True

    # -- helpers ---------------------------------------------------------------

    def _require_episode(self) -> LiberoEpisode:
        if self.episode is None:
            raise RuntimeError("No active episode. Call `reset` first.")
        return self.episode

    def _content(self, header: str, images: str = "both") -> list[TextContent | Image]:
        """Header text + state + the requested current images (inline or as paths).
        All images are always saved to disk (see save_observation); ``images`` only controls
        what is sent back."""
        if images not in IMAGE_CHOICES:
            raise ValueError(f"images must be one of {IMAGE_CHOICES}")
        ep = self._require_episode()
        text = (f"{header}\n\nTask: {ep.task_description}\nMoves so far: {ep.moves}"
                f"{'  (task already SUCCEEDED)' if ep.success else ''}\n\nRobot state:\n{format_state(ep.state())}")
        out: list[TextContent | Image] = []
        if not ep.has_cameras:
            out.append(TextContent(type="text", text=text + "\n\n(headless server: no images)"))
            return out
        wanted = {"both": ["agentview", "wrist"], "agentview": ["agentview"], "wrist": ["wrist"], "none": []}[images]
        if not wanted:
            out.append(TextContent(type="text", text=text + "\n\n(no images requested; call `observe` to see the scene)"))
            return out
        paths = {name: ep.obs_dir / f"turn_{ep.moves:03d}_{name}.png" for name in wanted}
        if not self.args.inline_images:
            text += "\n\nImages (Read them):\n" + "\n".join(f"- {n}: {p}" for n, p in paths.items())
            out.append(TextContent(type="text", text=text))
            return out
        out.append(TextContent(type="text", text=text + f"\n\nImages follow: {', then '.join(wanted)} (annotated)."))
        for name in wanted:
            img = ep.agentview_image() if name == "agentview" else ep.wrist_image()
            out.append(Image(data=_png(img), format="png"))
        return out

    def _finish_current(self) -> dict[str, Any] | None:
        ep = self.episode
        if ep is None or self.finished:
            return None
        video = None if self.args.no_video else ep.write_video(fps=self.args.frame_rate)
        summary = {
            "task_suite": self.task_suite, "task_id": self.task_id, "task": ep.task_description,
            "success": ep.success, "moves": ep.moves, "episode_dir": str(ep.episode_dir),
            "video": str(video) if video else None,
            "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        (ep.episode_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        with (self.args.output_dir / "episodes.jsonl").open("a") as f:
            f.write(json.dumps(summary) + "\n")
        self.finished = True
        return summary

    # -- tools -----------------------------------------------------------------

    def list_tasks(self, task_suite: str) -> str:
        _, _, n = load_task(task_suite, 0)
        lines = [f"{task_suite}: {n} tasks"]
        for i in range(n):
            task, init_states, _ = load_task(task_suite, i)
            lines.append(f"  {i}: {task.language}  ({len(init_states)} initial states)")
        return "\n".join(lines)

    def reset(self, task_suite: str, task_id: int, init_state_idx: int) -> list[TextContent | Image]:
        prev = self._finish_current()
        task, init_states, _ = load_task(task_suite, task_id)
        if not 0 <= init_state_idx < len(init_states):
            raise ValueError(f"init_state_idx must be in 0..{len(init_states) - 1}")
        key = (task_suite, task_id)
        if self.env is None or self.env_key != key:
            if self.env is not None:
                self.env.close()
            self.env = make_env(task, resolution=self.args.render_size, seed=self.args.seed,
                                headless=self.args.headless)
            self.env_key = key
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ep_dir = self.args.output_dir / f"{task_suite}_task{task_id}_init{init_state_idx}_{stamp}"
        self.episode = LiberoEpisode(self.env, task.language, init_states[init_state_idx], episode_dir=ep_dir,
                                     render_size=self.args.render_size, annotate=self.args.annotate,
                                     record_video=not self.args.no_video, keep_all_images=True,
                                     drop_line=self.args.annotate and self.args.drop_line)
        self.task_suite, self.task_id, self.finished = task_suite, task_id, False
        self.episode.save_observation()
        header = f"New episode: {task_suite} task {task_id}, initial state {init_state_idx}.\n" \
                 f"Episode files: {self.episode.episode_dir}"
        if prev:
            header = f"(previous episode saved: success={prev['success']}, {prev['moves']} moves)\n" + header
        return self._content(header)

    def observe(self, images: str) -> list[TextContent | Image]:
        return self._content("Current observation.", images)

    def _active_episode(self) -> LiberoEpisode:
        ep = self._require_episode()
        if self.finished:
            raise RuntimeError("This episode was finished. Call `reset` to start a new one.")
        return ep

    def move_relative(self, cmd: Command, steps: int | None, reasoning: str,
                      images: str) -> list[TextContent | Image]:
        ep = self._active_episode()
        n = self.args.steps_per_command if steps is None else int(steps)
        if not 1 <= n <= self.args.max_steps_per_command:
            raise ValueError(f"steps must be in 1..{self.args.max_steps_per_command}")
        _, report, text = ep.move(cmd, steps=n, gain=self.args.feedback_gain,
                                  max_cm=self.args.max_translation_cm, max_deg=self.args.max_rotation_deg,
                                  log_extra={"tool": "move_relative", "reasoning": reasoning, "steps": n})
        ep.save_observation()
        return self._content(f"Move {ep.moves} (relative) executed ({report.steps_run} sim steps).\n{text}", images)

    def move_absolute(self, x: float, y: float, z: float, rot: tuple[float, float, float] | None,
                      gripper: float, max_steps: int | None, reasoning: str,
                      images: str) -> list[TextContent | Image]:
        ep = self._active_episode()
        n = self.args.max_steps_absolute if max_steps is None else int(max_steps)
        if not 1 <= n <= self.args.max_steps_absolute:
            raise ValueError(f"max_steps must be in 1..{self.args.max_steps_absolute}")
        _, text = ep.move_absolute(x, y, z, rot=rot, gripper=gripper, max_steps=n, gain=self.args.feedback_gain,
                                   log_extra={"tool": "move_absolute", "reasoning": reasoning})
        ep.save_observation()
        return self._content(f"Move {ep.moves} (absolute) {text}", images)

    def finish(self) -> str:
        summary = self._finish_current()
        if summary is None:
            return "No unfinished episode."
        return json.dumps(summary, indent=2)


def _png(img: np.ndarray) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


ROBOT = _Robot(ARGS)
_ENV_THREAD = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="libero-env")


async def _on_env_thread(fn, *a, **kw):
    try:
        return await asyncio.wrap_future(_ENV_THREAD.submit(fn, *a, **kw))
    except Exception as exc:
        # ToolError (unlike other exceptions) delivers the message to the model instead of a
        # generic "Error executing tool"; this is a local single-user tool, so expose it.
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise ToolError(f"{type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

INSTRUCTIONS = f"""Controls a simulated Franka Panda robot arm in the LIBERO tabletop benchmark.

Workflow: `reset` starts an episode (returns the task string, robot state and images); then repeatedly look at the images and call `move_relative` (small relative 6-DOF step) or `move_absolute` (go to a position in episode coordinates) with a gripper command; each move returns what it achieved plus fresh images. The episode reports TASK SUCCEEDED when the goal is reached. Call `finish` when done to save the video. `observe` re-sends the current observation without moving.

{control_conventions_text(steps_per_command=ARGS.steps_per_command, max_cm=ARGS.max_translation_cm,
                          max_deg=ARGS.max_rotation_deg, mcp_tools=True,
                          max_steps_absolute=ARGS.max_steps_absolute,
                          drop_line=ARGS.annotate and ARGS.drop_line)}"""

IMAGE_CHOICES = ("both", "agentview", "wrist", "none")
ImagesArg = Annotated[str, Field(description="Which images to return after the call: "
                                             "'both' (default), 'agentview', 'wrist' or 'none'. "
                                             "Skipping images saves context on transit moves.")]
GripperArg = Annotated[float, Field(ge=-1, le=1, description="+1 close, -1 open, 0 hold; applied for the whole move")]
ReasoningArg = Annotated[str, Field(description="Optional short note on why; logged to turns.jsonl")]

mcp = MCPServer("libero", instructions=INSTRUCTIONS)

@mcp.tool()
async def list_tasks(
    task_suite: Annotated[str, Field(description="libero_spatial, libero_object, libero_goal, libero_10, libero_90")]
    = "libero_10",
) -> str:
    """List the tasks (ids and language instructions) in a LIBERO task suite."""
    return await _on_env_thread(ROBOT.list_tasks, task_suite)


@mcp.tool()
async def reset(
    task_suite: str = "libero_10",
    task_id: int = 0,
    init_state_idx: Annotated[int, Field(description="Which of the task's fixed initial states (0..49)")] = 0,
) -> list[TextContent | Image]:
    """Start a new episode (saves the previous one). Returns the task, robot state, and the
    agentview + wrist images."""
    return await _on_env_thread(ROBOT.reset, task_suite, task_id, init_state_idx)


@mcp.tool()
async def observe(images: ImagesArg = "both") -> list[TextContent | Image]:
    """Return the current robot state and images without moving."""
    return await _on_env_thread(ROBOT.observe, images)


@mcp.tool()
async def move_relative(
    dx: Annotated[float, Field(description="cm, + = right in the agentview image")] = 0.0,
    dy: Annotated[float, Field(description="cm, + = straight DOWN (gravity); negative lifts")] = 0.0,
    dz: Annotated[float, Field(description="cm, + = into the agentview image (away from camera, toward robot base)")] = 0.0,
    rx: Annotated[float, Field(description="deg about +x, right-hand rule, pivot at gripper centre")] = 0.0,
    ry: Annotated[float, Field(description="deg about +y (down axis); twists a downward-pointing gripper")] = 0.0,
    rz: Annotated[float, Field(description="deg about +z (into-image axis)")] = 0.0,
    gripper: GripperArg = 0.0,
    steps: Annotated[int | None, Field(description="Simulation steps for this move (20 = 1 s). Default: server setting.")] = None,
    reasoning: ReasoningArg = "",
    images: ImagesArg = "both",
) -> list[TextContent | Image]:
    """Execute one RELATIVE move of the gripper (level camera frame; translation clipped per call)
    with a feedback controller for a fixed number of simulation steps, then return the execution
    report, robot state and new images. Use for small corrections and gripper open/close."""
    cmd = Command(dx=dx, dy=dy, dz=dz, rx=rx, ry=ry, rz=rz, gripper=gripper)
    return await _on_env_thread(ROBOT.move_relative, cmd, steps, reasoning, images)


@mcp.tool()
async def move_absolute(
    x: Annotated[float, Field(description="target cm from the episode-start gripper position; + = right in the agentview image")],
    y: Annotated[float, Field(description="target cm from the episode-start gripper position; + = DOWN (smaller = higher)")],
    z: Annotated[float, Field(description="target cm from the episode-start gripper position; + = into the agentview image")],
    rx: Annotated[float | None, Field(description="target rotation from the episode-start orientation, rotation-vector "
                                                  "x component in deg. Give all three of rx/ry/rz or none (none = keep current orientation)")] = None,
    ry: Annotated[float | None, Field(description="target rotation-vector y component in deg (see rx)")] = None,
    rz: Annotated[float | None, Field(description="target rotation-vector z component in deg (see rx)")] = None,
    gripper: GripperArg = 0.0,
    max_steps: Annotated[int | None, Field(description="Timeout in simulation steps (20 = 1 s). Default: server maximum.")] = None,
    reasoning: ReasoningArg = "",
    images: ImagesArg = "both",
) -> list[TextContent | Image]:
    """Move the gripper to an ABSOLUTE position given in the coordinates the robot state reports
    ("gripper position relative to its position at the start of the episode"). No distance limit:
    the controller runs until the target is reached and the gripper has settled, progress stalls
    (e.g. blocked by contact), the task succeeds, or the timeout elapses; the stop reason is
    reported. Use for transit (lift, carry, return to a remembered position)."""
    rots = (rx, ry, rz)
    if any(r is not None for r in rots) and any(r is None for r in rots):
        raise ToolError("Give all three of rx, ry, rz, or none of them (keep current orientation).")
    rot = None if rx is None else (float(rx), float(ry), float(rz))
    return await _on_env_thread(ROBOT.move_absolute, x, y, z, rot, gripper, max_steps, reasoning, images)


@mcp.tool()
async def finish() -> str:
    """End the current episode: write video.mp4 + summary.json and log it to episodes.jsonl."""
    return await _on_env_thread(ROBOT.finish)


def main() -> None:
    ARGS.output_dir = ARGS.output_dir.resolve()
    ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    async def serve() -> None:
        wire = anyio.wrap_file(io.TextIOWrapper(os.fdopen(_WIRE_FD, "wb", closefd=False), encoding="utf-8"))
        async with stdio_server(stdout=wire) as (read_stream, write_stream):
            server = mcp._lowlevel_server
            await server.run(read_stream, write_stream, server.create_initialization_options())

    try:
        anyio.run(serve)
    finally:
        try:
            _ENV_THREAD.submit(ROBOT._finish_current).result(timeout=120)
        except Exception as exc:  # noqa: BLE001 — best effort on shutdown
            print(f"[libero_mcp] could not save episode on shutdown: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
