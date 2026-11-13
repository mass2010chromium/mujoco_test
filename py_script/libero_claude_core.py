"""Shared LIBERO control core for the Claude-driven entrypoints.

Used by:
  - ``libero_claude.py``  — batch mode: the script drives ``claude -p`` turn by turn.
  - ``libero_mcp.py``     — interactive mode: an MCP server; a Claude Code session drives the robot.

-----------------------------------------------------------------------------------------
Control frame — DESIGN DECISION: "level camera frame"
-----------------------------------------------------------------------------------------
Commands are expressed in a frame derived from the agent-view camera, but *levelled*:
only the camera's heading (yaw) is kept, its pitch/roll are discarded.

  +x : image-right, horizontal                      (agentview in LIBERO: +y world)
  +y : straight DOWN, i.e. along gravity            (-z world)
  +z : into the image, horizontal, toward the robot (agentview in LIBERO: -x world)

This is OpenCV axis order (x-right, y-down, z-forward) and is right-handed
(x × y = z), so rotations follow the right-hand rule about each axis.

Why levelled rather than the true camera axes: the agent-view camera is pitched ~32°
down. In the true camera frame "up in the image" is not "up in the world", so lifting an
object straight up would need a mix of y and z that a language model will often get
wrong. Levelling keeps "left/right in the image" and "toward/away from the camera"
intuitive while making y exactly vertical.

Rotations use the same three axes, but pivot about the gripper centre (the ``grip_site``,
between the fingertips), not about the camera. A command's rotation (rx, ry, rz) in
degrees is a rotation vector in this frame: R_target = Rot(rx, ry, rz) · R_current.
For a single non-zero component this is simply "rotate the gripper by that many degrees
about that axis, right-hand rule".

Images: we show Claude the TRUE camera view, i.e. ``raw[::-1]`` (MuJoCo renders bottom-up).
This differs from openpi / LeRobot LIBERO, which use ``raw[::-1, ::-1]`` — a horizontally
mirrored image (see LEROBOT_IMAGE_CONVENTION = "robosuite_agentview_horizontal_flip" in
libero_traceVLA_test.py). A mirrored image would make the image-defined frame
left-handed relative to the world, so rotations would turn the wrong way.

-----------------------------------------------------------------------------------------
Command execution
-----------------------------------------------------------------------------------------
LIBERO uses robosuite's OSC_POSE controller in *delta* mode (goal = current + 0.05·a_pos,
R_goal = Rot(0.5·a_rot)·R_current, recomputed every env step). Holding a constant action
therefore integrates; it is not a position command. Instead of changing the env
definition, we fix an absolute target pose once per command and run a proportional
feedback loop on top of the delta controller for exactly ``steps`` env steps:
each step sends the remaining error (scaled to the controller's units, then uniformly
shrunk so no component exceeds ±1). If the target is not reached in time, that's
accepted; the shortfall is reported back to the agent.

Gripper: kept consistent with LIBERO — a *velocity-like* command in [-1, 1] held for the
whole command. Verified empirically (libero_10 task 0): +1 CLOSES, -1 OPENS, 0 holds.
A full open<->close takes ~10–13 env steps.
"""
from __future__ import annotations

import dataclasses
import json
import math
import pathlib
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OSC_POSE output scaling in LIBERO's default controller config (robosuite 1.4
# controllers/config/osc_pose.json): an action of 1.0 means 5 cm / 0.5 rad per step.
CTRL_POS_SCALE_M = 0.05
CTRL_ROT_SCALE_RAD = 0.5

# Panda finger joint range observed in LIBERO (per finger, metres): ~0.001 closed, ~0.039 open.
FINGER_CLOSED_M = 0.001
FINGER_OPEN_M = 0.039

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]   # hold pose, open gripper
NUM_SETTLE_STEPS = 20                      # matches the other LIBERO scripts

AGENTVIEW_CAMERA = "agentview"
WRIST_CAMERA = "robot0_eye_in_hand"

# Per-command limits exposed to the agent (also enforced by clipping).
DEFAULT_MAX_TRANSLATION_CM = 10.0
DEFAULT_MAX_ROTATION_DEG = 45.0
DEFAULT_STEPS_PER_COMMAND = 20


# ---------------------------------------------------------------------------
# Small rotation helpers (numpy only; robosuite's transform_utils use xyzw quats)
# ---------------------------------------------------------------------------

def rotvec_to_mat(rv: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, radians) -> 3x3 rotation matrix (Rodrigues)."""
    rv = np.asarray(rv, dtype=float)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def mat_to_rotvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> rotation vector (radians), robust near 0 and pi."""
    R = np.asarray(R, dtype=float)
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = math.acos(cos_t)
    if theta < 1e-8:
        return np.zeros(3)
    if math.pi - theta < 1e-4:
        # Near pi: axis from the diagonal of (R + I) / 2.
        B = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(B), 0.0, None))
        i = int(np.argmax(axis))
        axis[i] = math.sqrt(max(B[i, i], 0.0))
        for j in range(3):
            if j != i:
                axis[j] = B[i, j] / axis[i]
        axis /= np.linalg.norm(axis)
        return axis * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (theta / (2.0 * math.sin(theta)))


# ---------------------------------------------------------------------------
# Level camera frame
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class LevelCameraFrame:
    """Gravity-aligned frame with the agent-view camera's heading (see module docstring).

    ``R_wl`` maps level-frame vectors to world vectors (columns = level axes in world).
    """

    R_wl: np.ndarray

    @classmethod
    def from_sim(cls, sim, camera_name: str = AGENTVIEW_CAMERA) -> "LevelCameraFrame":
        cid = sim.model.camera_name2id(camera_name)
        cam_R = np.asarray(sim.data.cam_xmat[cid], dtype=float).reshape(3, 3)
        # MuJoCo camera convention: looks along -z_cam, x_cam = image right, y_cam = image up.
        cam_right = cam_R[:, 0]
        cam_fwd = -cam_R[:, 2]
        fwd_h = np.array([cam_fwd[0], cam_fwd[1], 0.0])
        if np.linalg.norm(fwd_h) < 1e-6:
            raise ValueError(f"Camera {camera_name!r} looks straight up/down; heading undefined.")
        z_l = fwd_h / np.linalg.norm(fwd_h)            # into the scene, horizontal
        y_l = np.array([0.0, 0.0, -1.0])                # down
        x_l = np.cross(y_l, z_l)                        # right (x = y × z for right-handed)
        if float(np.dot(x_l, cam_right)) <= 0.0:
            raise AssertionError("Level-frame +x disagrees with the camera's image-right axis.")
        return cls(R_wl=np.column_stack([x_l, y_l, z_l]))

    def to_world(self, v_level: np.ndarray) -> np.ndarray:
        return self.R_wl @ np.asarray(v_level, dtype=float)

    def to_level(self, v_world: np.ndarray) -> np.ndarray:
        return self.R_wl.T @ np.asarray(v_world, dtype=float)


# ---------------------------------------------------------------------------
# Robot state access (uses the controller's grip_site, so targets and the
# OSC controller agree on what "end-effector pose" means)
# ---------------------------------------------------------------------------

def eef_pose(env) -> tuple[np.ndarray, np.ndarray]:
    robot = env.robots[0]
    site = robot.eef_site_id
    pos = np.array(env.sim.data.site_xpos[site], dtype=float)
    R = np.array(env.sim.data.site_xmat[site], dtype=float).reshape(3, 3)
    return pos, R


def gripper_opening(obs: dict) -> float:
    """0 = fully closed, 1 = fully open (mean of the two finger joints)."""
    q = np.asarray(obs["robot0_gripper_qpos"], dtype=float)
    per_finger = 0.5 * (abs(q[0]) + abs(q[1]))
    return float(np.clip((per_finger - FINGER_CLOSED_M) / (FINGER_OPEN_M - FINGER_CLOSED_M), 0.0, 1.0))


def fingerpad_positions(env) -> tuple[np.ndarray, np.ndarray]:
    """World positions of the left / right finger pads (robot geometry, not scene knowledge)."""
    g = env.robots[0].gripper
    out = []
    for key in ("left_fingerpad", "right_fingerpad"):
        gid = env.sim.model.geom_name2id(g.important_geoms[key][0])
        out.append(np.array(env.sim.data.geom_xpos[gid], dtype=float))
    return out[0], out[1]


def static_surface_below(env, p: np.ndarray, max_hits: int = 32) -> np.ndarray | None:
    """First STATIC surface (table / floor / fixtures welded to the world) straight below ``p``.

    Movable objects and the robot itself are skipped, so this is fixed scene geometry (like a
    calibrated table height), not object perception. Returns the world hit point or None.
    """
    import mujoco

    m, d = env.sim.model._model, env.sim.data._data
    down = np.array([0.0, 0.0, -1.0])
    pnt = np.asarray(p, dtype=float).copy()
    geomid = np.array([-1], dtype=np.int32)
    for _ in range(max_hits):
        dist = mujoco.mj_ray(m, d, pnt, down, None, 1, -1, geomid)
        if geomid[0] < 0 or dist < 0:
            return None
        hit = pnt + dist * down
        if m.body_weldid[m.geom_bodyid[geomid[0]]] == 0:
            return hit
        pnt = hit + 1e-4 * down
    return None


# ---------------------------------------------------------------------------
# Commands + feedback-loop execution
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Command:
    """One agent command in the level camera frame. Units: cm and degrees."""

    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    gripper: float = 0.0   # +1 close, -1 open, 0 hold (velocity-like, LIBERO convention)

    def clipped(self, max_cm: float, max_deg: float) -> "Command":
        t = np.array([self.dx, self.dy, self.dz], dtype=float)
        r = np.array([self.rx, self.ry, self.rz], dtype=float)
        tn, rn = float(np.linalg.norm(t)), float(np.linalg.norm(r))
        if tn > max_cm:
            t *= max_cm / tn
        if rn > max_deg:
            r *= max_deg / rn
        g = float(np.clip(self.gripper, -1.0, 1.0))
        return Command(*t.tolist(), *r.tolist(), g)

    def as_dict(self) -> dict[str, float]:
        return {k: round(float(v), 4) for k, v in dataclasses.asdict(self).items()}


@dataclasses.dataclass
class ExecutionReport:
    steps_run: int
    success: bool
    commanded_translation_cm: list[float]
    achieved_translation_cm: list[float]
    remaining_translation_cm: list[float]
    commanded_rotation_deg: list[float]
    remaining_rotation_deg: float
    gripper_before: float
    gripper_after: float
    steps_requested: int = 0
    # "steps exhausted" | "target reached" | "stalled" | "task succeeded"
    stop_reason: str = "steps exhausted"

    def summary(self) -> str:
        cmd_n = float(np.linalg.norm(self.commanded_translation_cm))
        rem_n = float(np.linalg.norm(self.remaining_translation_cm))
        fmt = lambda v: "[" + ", ".join(f"{x:+.1f}" for x in v) + "]"
        lines = [
            f"commanded translation {fmt(self.commanded_translation_cm)} cm "
            f"(|{cmd_n:.1f}| cm); achieved {fmt(self.achieved_translation_cm)} cm; "
            f"remaining error {fmt(self.remaining_translation_cm)} cm (|{rem_n:.1f}| cm)",
            f"commanded rotation {fmt(self.commanded_rotation_deg)} deg; "
            f"remaining rotation error {self.remaining_rotation_deg:.1f} deg",
            f"gripper opening {self.gripper_before:.2f} -> {self.gripper_after:.2f} (0=closed, 1=open)",
        ]
        if self.stop_reason == "task succeeded" and self.steps_run < self.steps_requested:
            lines.append(f"NOTE: the episode ended (task succeeded) after {self.steps_run} of "
                         f"{self.steps_requested} steps; the rest of this command was NOT executed "
                         f"(e.g. a commanded gripper open/close may be incomplete).")
        elif self.stop_reason == "stalled":
            lines.append(f"NOTE: stopped after {self.steps_run} steps because the gripper stopped making "
                         f"progress (likely blocked by contact or a joint limit).")
        elif rem_n > 1.0:
            lines.append("NOTE: the target position was NOT reached (possibly blocked by contact, "
                         "a joint limit, or too few steps).")
        return "\n".join(lines)


def drive_to_pose(
    env,
    obs: dict,
    frame: LevelCameraFrame,
    p_target: np.ndarray,
    R_target: np.ndarray,
    gripper: float,
    *,
    max_steps: int,
    gain: float = 1.0,
    stop_when_reached: bool = False,
    pos_tol_cm: float = 0.5,
    rot_tol_deg: float = 2.0,
    stall_window: int = 15,
    stall_progress_cm: float = 0.2,
    on_step: Callable[[dict], None] | None = None,
) -> tuple[dict, ExecutionReport]:
    """Drive the delta OSC controller toward a fixed absolute target pose (world frame).

    Each step sends a = gain · error / controller_scale, uniformly shrunk so that
    max|a_i| <= 1 (direction preserved), plus the gripper command. Always stops on task
    success. With ``stop_when_reached`` it also stops once the pose is within tolerance and
    the gripper has settled, or when progress stalls (no pose / gripper progress over
    ``stall_window`` steps, e.g. blocked by contact); otherwise it runs exactly ``max_steps``.
    """
    p0, R0 = eef_pose(env)
    grip_before = gripper_opening(obs)
    pos_errs: list[float] = []      # cm
    rot_errs: list[float] = []      # deg
    grips: list[float] = [grip_before]
    p_prev: np.ndarray | None = None

    stop_reason = "steps exhausted"
    k = 0
    for k in range(1, max_steps + 1):
        p, R = eef_pose(env)
        a_pos = gain * (p_target - p) / CTRL_POS_SCALE_M
        a_rot = gain * mat_to_rotvec(R_target @ R.T) / CTRL_ROT_SCALE_RAD
        for a in (a_pos, a_rot):
            m = float(np.max(np.abs(a)))
            if m > 1.0:
                a /= m
        action = np.concatenate([a_pos, a_rot, [gripper]]).astype(np.float32)
        obs, _reward, done, _info = env.step(action)
        if on_step is not None:
            on_step(obs)
        if done:
            stop_reason = "task succeeded"
            break
        if not stop_when_reached:
            continue
        p, R = eef_pose(env)
        pos_errs.append(float(np.linalg.norm(p_target - p)) * 100)
        rot_errs.append(float(np.rad2deg(np.linalg.norm(mat_to_rotvec(R_target @ R.T)))))
        grips.append(gripper_opening(obs))
        grip_settled = gripper == 0 or (len(grips) >= 4 and max(grips[-4:]) - min(grips[-4:]) < 0.005)
        # "Reached" also requires the arm to have (nearly) stopped (< 1 mm per step), so it does
        # not drift afterwards.
        still = p_prev is not None and float(np.linalg.norm(p - p_prev)) * 100 < 0.1
        p_prev = p
        if pos_errs[-1] < pos_tol_cm and rot_errs[-1] < rot_tol_deg and grip_settled and still:
            stop_reason = "target reached"
            break
        if len(pos_errs) > stall_window and grip_settled:
            j = -stall_window - 1
            if (pos_errs[j] - pos_errs[-1] < stall_progress_cm
                    and rot_errs[j] - rot_errs[-1] < 0.5):
                # Converging slowly right next to the target counts as reached.
                near = pos_errs[-1] < 2 * pos_tol_cm and rot_errs[-1] < 2 * rot_tol_deg
                stop_reason = "target reached" if near else "stalled"
                break

    p1, R1 = eef_pose(env)
    report = ExecutionReport(
        steps_run=k,
        success=stop_reason == "task succeeded",
        commanded_translation_cm=(frame.to_level(p_target - p0) * 100).round(2).tolist(),
        achieved_translation_cm=(frame.to_level(p1 - p0) * 100).round(2).tolist(),
        remaining_translation_cm=(frame.to_level(p_target - p1) * 100).round(2).tolist(),
        commanded_rotation_deg=np.rad2deg(frame.to_level(mat_to_rotvec(R_target @ R0.T))).round(2).tolist(),
        remaining_rotation_deg=round(float(np.rad2deg(np.linalg.norm(mat_to_rotvec(R_target @ R1.T)))), 2),
        gripper_before=round(grip_before, 3),
        gripper_after=round(gripper_opening(obs), 3),
        steps_requested=max_steps,
        stop_reason=stop_reason,
    )
    return obs, report


def execute_command(
    env,
    obs: dict,
    frame: LevelCameraFrame,
    cmd: Command,
    *,
    steps: int,
    gain: float = 1.0,
    on_step: Callable[[dict], None] | None = None,
) -> tuple[dict, ExecutionReport]:
    """RELATIVE command: fix the target once from the pose at command start, then run
    ``drive_to_pose`` for exactly ``steps`` steps (early stop only on task success).
        p* = p0 + R_wl · (dx, dy, dz)
        R* = Rot(R_wl · (rx, ry, rz)) · R0        (pivot = gripper centre)
    """
    p0, R0 = eef_pose(env)
    p_target = p0 + frame.to_world(np.array([cmd.dx, cmd.dy, cmd.dz], dtype=float) / 100.0)
    R_target = rotvec_to_mat(frame.to_world(np.deg2rad([cmd.rx, cmd.ry, cmd.rz]))) @ R0
    obs, report = drive_to_pose(env, obs, frame, p_target, R_target, cmd.gripper, max_steps=steps,
                                gain=gain, stop_when_reached=False, on_step=on_step)
    # Report exactly what was commanded (avoids rotvec round-trip noise).
    report.commanded_translation_cm = np.round([cmd.dx, cmd.dy, cmd.dz], 2).tolist()
    report.commanded_rotation_deg = np.round([cmd.rx, cmd.ry, cmd.rz], 2).tolist()
    return obs, report


# ---------------------------------------------------------------------------
# Images + annotation
# ---------------------------------------------------------------------------

def true_view(raw: np.ndarray) -> np.ndarray:
    """MuJoCo renders bottom-up; a vertical flip gives the true (un-mirrored) camera view.
    NOT the openpi/LeRobot ``[::-1, ::-1]`` convention, which is horizontally mirrored."""
    return np.ascontiguousarray(raw[::-1])


def project_world_to_pixel(points_world: np.ndarray, world_to_pixel: np.ndarray) -> np.ndarray:
    """(N, 3) world points -> (N, 2) float (col, row) in the true-view image.
    robosuite's camera transform uses OpenCV axes (x right, y down) for the upright image."""
    pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    proj = (world_to_pixel @ hom.T).T
    return proj[:, :2] / proj[:, 2:3]


CYAN = (0, 255, 255)        # images are RGB
MAGENTA = (255, 0, 255)
DROP_MARK_RADIUS_M = 0.02   # the table mark is a 2 cm-radius circle -> also a scale reference


def _ipt(v) -> tuple[int, int]:
    return int(round(float(v[0]))), int(round(float(v[1])))


def _dashed_line(img, a, b, color, thickness, dash_px: int = 6) -> None:
    import cv2

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = int(np.linalg.norm(b - a) // dash_px)
    if n < 1:
        cv2.line(img, _ipt(a), _ipt(b), color, thickness, cv2.LINE_AA)
        return
    for i in range(0, n, 2):
        cv2.line(img, _ipt(a + (b - a) * i / n), _ipt(a + (b - a) * min(i + 1, n) / n), color, thickness,
                 cv2.LINE_AA)


def _draw_drop_mark(img, p_eef: np.ndarray, surface: np.ndarray, world_to_pixel: np.ndarray, scale: int) -> None:
    """Dashed vertical line from the gripper centre down to the static surface below it, ending
    in a horizontal circle of radius DROP_MARK_RADIUS_M lying on that surface."""
    import cv2

    ang = np.linspace(0, 2 * np.pi, 33)
    ring = surface + DROP_MARK_RADIUS_M * np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1)
    px = project_world_to_pixel(np.vstack([p_eef[None], surface[None], ring]), world_to_pixel)
    _dashed_line(img, px[0], px[1], MAGENTA, scale)
    cv2.polylines(img, [np.round(px[2:]).astype(np.int32)], True, MAGENTA, scale, cv2.LINE_AA)
    cv2.drawMarker(img, _ipt(px[1]), MAGENTA, cv2.MARKER_CROSS, 4 * scale, scale, cv2.LINE_AA)


def annotate_agentview(img: np.ndarray, env, frame: LevelCameraFrame, world_to_pixel: np.ndarray,
                       axis_len_m: float = 0.06, drop_line: bool = False) -> np.ndarray:
    """Gripper-centre marker, level-frame axes (+x red, +y green, +z blue) and, with ``drop_line``,
    a drop line to the table directly below the gripper centre."""
    import cv2

    out = np.array(img, dtype=np.uint8, copy=True)
    p, _ = eef_pose(env)
    scale = max(1, out.shape[0] // 256)
    surface = static_surface_below(env, p) if drop_line else None
    if surface is not None:
        _draw_drop_mark(out, p, surface, world_to_pixel, scale)
    pts = [p] + [p + frame.to_world(axis_len_m * e) for e in np.eye(3)]
    px = project_world_to_pixel(np.array(pts), world_to_pixel)
    o = _ipt(px[0])
    for i, color in enumerate([(255, 0, 0), (0, 200, 0), (0, 100, 255)]):
        tip = _ipt(px[i + 1])
        cv2.arrowedLine(out, o, tip, color, scale, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(out, "xyz"[i], tip, cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, color, scale, cv2.LINE_AA)
    cv2.circle(out, o, 3 * scale, CYAN, -1, cv2.LINE_AA)
    return out


def annotate_wrist(img: np.ndarray, env, drop_line: bool = False) -> np.ndarray:
    """Mark where the gripper actually grasps in the (offset) wrist camera: the two finger pads
    joined by a cyan closing line with the gripper centre as a cyan dot, plus (with ``drop_line``)
    the dashed drop line from the gripper centre down to the table. An object is centred for a
    grasp when it straddles the cyan dot/line at the object's height — the drop line shows where
    'directly below the gripper centre' falls at every height."""
    import cv2
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    out = np.array(img, dtype=np.uint8, copy=True)
    h, w = out.shape[:2]
    w2p = get_camera_transform_matrix(env.sim, WRIST_CAMERA, h, w)
    scale = max(1, h // 256)
    p, _ = eef_pose(env)
    surface = static_surface_below(env, p) if drop_line else None
    if surface is not None:
        _draw_drop_mark(out, p, surface, w2p, scale)
    left, right = fingerpad_positions(env)
    px = project_world_to_pixel(np.array([p, left, right]), w2p)
    cv2.line(out, _ipt(px[1]), _ipt(px[2]), CYAN, scale, cv2.LINE_AA)
    for q in px[1:]:
        cv2.circle(out, _ipt(q), 3 * scale, CYAN, scale, cv2.LINE_AA)
    cv2.circle(out, _ipt(px[0]), 3 * scale, CYAN, -1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Shared agent-facing description of the control interface
# ---------------------------------------------------------------------------

def control_conventions_text(*, steps_per_command: int, max_cm: float, max_deg: float,
                            mcp_tools: bool = False, max_steps_absolute: int = 200,
                            drop_line: bool = False) -> str:
    """Frame / gripper / execution conventions, shared by the batch system prompt and the MCP server.
    ``mcp_tools=True`` describes the MCP server's move_relative / move_absolute tools;
    ``drop_line=True`` describes the (optional, slightly privileged) table drop-line overlay."""
    agent_drop = wrist_drop = wrist_drop_tip = ""
    if drop_line:
        agent_drop = (" A dashed magenta line drops straight down from the gripper centre to the table, ending in a "
                      "magenta circle of 2 cm radius lying on the table: compare that circle with an object's base to "
                      "judge left/right and near/far alignment (the cyan dot alone mixes up height and depth), and use "
                      "its size as a scale reference. The robot state also reports the gripper centre's height above "
                      "the table.")
        wrist_drop = (" The same dashed magenta drop line and 2 cm circle show the point on the table directly below "
                      "the gripper centre.")
        wrist_drop_tip = " When descending onto an object, its top should lie on the magenta drop line."
    if mcp_tools:
        execution = f"""Execution: two move tools; both apply the gripper command for the whole move.
- `move_relative(dx, dy, dz, rx, ry, rz, gripper)`: a move RELATIVE to the current pose, run for a fixed number of simulation steps (default {steps_per_command}; 20 steps = 1 second). Translation is clipped to {max_cm:g} cm and rotation to {max_deg:g} degrees per call. Best for small corrective moves near objects and for pure gripper open/close.
- `move_absolute(x, y, z, [rx, ry, rz], gripper)`: go to an ABSOLUTE position given in the same coordinates the robot state reports (cm from the gripper's position at the start of the episode, command-frame axes). No distance limit; it runs until the target is reached and the gripper has settled, until progress stalls (e.g. blocked by contact), or up to {max_steps_absolute} steps. Omit rx/ry/rz to keep the current orientation. The gripper travels in a STRAIGHT line to the target, so lift first and only descend once above the destination, otherwise it can sweep through objects on the way. Best for transit: lifting, carrying an object across the table, returning to a remembered position.
If a target is not reached, the remaining error is reported. Moves that do not need visual feedback can skip images with `images="none"` (or request only "agentview" / "wrist")."""
    else:
        execution = f"""Execution: each command is a RELATIVE move from the current pose. A feedback controller tries to reach that target for a fixed number of simulation steps (default {steps_per_command}; 20 steps = 1 second), while applying the gripper command the whole time. If it cannot reach the target in time (e.g. blocked by contact), the remaining error is reported. Per command, the translation is clipped to {max_cm:g} cm and the rotation to {max_deg:g} degrees."""
    return f"""Images:
- agentview: a fixed camera facing the robot across the table. The robot base is at the far side of the table, facing the camera.
- wrist: a camera mounted on the gripper, looking along the fingers. It turns with the gripper, so its image axes do NOT match the agentview: with the initial downward-pointing gripper, wrist-image right is -x and wrist-image down is +z (i.e. it is rotated ~180 degrees relative to the agentview). The current mapping is given in the robot state every turn.
On the agentview image, the gripper centre is marked with a cyan dot, with short arrows showing the command axes from that point: x (red), y (green), z (blue).{agent_drop}
On the wrist image, the wrist camera is mounted OFF-CENTRE (above and to one side of the fingers), so the image centre is NOT where the gripper grasps. The grasp location is drawn instead: two cyan circles mark the finger pads, joined by a cyan closing line with the gripper centre as a cyan dot.{wrist_drop} Before closing, the object should straddle the cyan line between the finger pads.{wrist_drop_tip}

Command frame (metric, gravity-aligned, derived from the agentview camera's heading):
- +x = to the RIGHT in the agentview image (horizontal)
- +y = straight DOWN (towards the table). Use negative dy to lift.
- +z = INTO the agentview image, horizontally away from the camera and toward the robot base. Negative dz brings the gripper toward the camera.
Directions in the task text such as "left"/"right"/"front"/"back" usually refer to the ROBOT's point of view. The robot faces the camera, so the robot's left appears on the RIGHT side of the agentview image, and the robot's front is toward the camera (negative z).

Rotations (rx, ry, rz) are in degrees about those same three axes, pivoting about the gripper centre, right-hand rule (thumb along the + axis, fingers curl in the positive direction). Example: ry rotates the gripper about the vertical axis (twisting a downward-pointing gripper).

Gripper: +1 closes, -1 opens, 0 holds the current opening. It takes ~10-13 simulation steps to fully open or close. A gripper that stays partly open after closing is usually holding an object.

{execution}

Strategy tips: approach objects from above, then descend; open the gripper before descending onto an object; close it, then lift before moving to the destination; make small corrective moves when close to an object. Keep rotations at zero unless the object or handle needs it."""


def format_state(state: dict[str, Any]) -> str:
    fmt = lambda v: "[" + ", ".join(f"{x:+.1f}" for x in v) + "]"
    lines = [
        f"- gripper position relative to its position at the start of the episode (x, y, z) cm: "
        f"{fmt(state['eef_from_start_cm'])}",
        f"- gripper rotation relative to the start of the episode, as a rotation vector (x, y, z) deg: "
        f"{fmt(state['rot_from_start_deg'])}",
        f"- gripper pointing direction (unit vector along the fingers, in the command frame): "
        f"{fmt(state['approach_dir'])}",
        f"- gripper opening: {state['gripper_opening']:.2f} (0 = closed, 1 = fully open)",
    ]
    if state.get("height_above_surface_cm") is not None:
        lines.append(f"- gripper centre height above the table/floor directly below it (ignoring objects): "
                     f"{state['height_above_surface_cm']:.1f} cm")
    if state.get("wrist_image_right") is not None:
        lines.append(f"- wrist image axes in the command frame: image-right = {fmt(state['wrist_image_right'])}, "
                     f"image-down = {fmt(state['wrist_image_down'])}")
    if state.get("eef_pixel") is not None:
        lines.append(f"- gripper centre in the agentview image (col, row) px: {state['eef_pixel']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------

def load_task(task_suite: str, task_id: int):
    """Returns (task, init_states, n_tasks)."""
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    if task_suite not in suites:
        raise ValueError(f"Unknown task suite {task_suite!r}; available: {list(suites)}")
    suite = suites[task_suite]()
    if not 0 <= task_id < suite.n_tasks:
        raise ValueError(f"task_id {task_id} out of range for {task_suite} (0..{suite.n_tasks - 1})")
    return suite.get_task(task_id), suite.get_task_init_states(task_id), suite.n_tasks


def make_env(task, *, resolution: int, seed: int, headless: bool = False):
    """``headless=True`` builds a ControlEnv without any renderer / camera obs (for machines
    without working EGL — useful for testing the control plumbing only)."""
    from libero.libero import get_libero_path

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if headless:
        from libero.libero.envs.env_wrapper import ControlEnv
        env = ControlEnv(bddl_file_name=bddl, has_renderer=False, has_offscreen_renderer=False,
                         use_camera_obs=False)
    else:
        from libero.libero.envs import OffScreenRenderEnv
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env


# ---------------------------------------------------------------------------
# Episode: env state + observation saving + command execution + logging
# ---------------------------------------------------------------------------

class LiberoEpisode:
    """One episode on an existing env. Shared by the batch script and the MCP server.

    Files written under ``episode_dir``:
      obs/turn_XXX_{agentview,wrist}.png   observation before move XXX (all kept unless keep_all_images=False)
      turns.jsonl                          one line per executed move
      first_frame.png, video.mp4           from ``write_video``
    """

    def __init__(self, env, task_description: str, init_state, *, episode_dir: pathlib.Path,
                 render_size: int, annotate: bool = True, record_video: bool = True,
                 keep_all_images: bool = True, drop_line: bool = False):
        self.env = env
        self.task_description = task_description
        self.episode_dir = pathlib.Path(episode_dir).resolve()
        self.obs_dir = self.episode_dir / "obs"
        self.obs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.episode_dir / "turns.jsonl"
        self.log_path.unlink(missing_ok=True)
        self.render_size = render_size
        self.annotate = annotate
        # Table drop line + height-above-table state: uses the static scene geometry below the
        # gripper (a raycast), i.e. a small amount of privileged information. Off by default.
        self.drop_line = drop_line
        self.keep_all_images = keep_all_images
        self.video: list[np.ndarray] = []
        self.moves = 0
        self.success = False

        env.reset()
        obs = env.set_init_state(init_state)
        for _ in range(NUM_SETTLE_STEPS):
            obs, _r, _d, _i = env.step(LIBERO_DUMMY_ACTION)
        self.obs = obs
        self.has_cameras = "agentview_image" in obs
        self.record_video = record_video and self.has_cameras

        from robosuite.utils.camera_utils import get_camera_transform_matrix
        self.frame = LevelCameraFrame.from_sim(env.sim)
        self.world_to_pixel = get_camera_transform_matrix(env.sim, AGENTVIEW_CAMERA, render_size, render_size)
        self.p_start, self.R_start = eef_pose(env)
        self._record(obs)

    # -- observation ---------------------------------------------------------

    def _record(self, obs: dict) -> None:
        if self.record_video:
            self.video.append(true_view(obs["agentview_image"]))

    def state(self) -> dict[str, Any]:
        p, R = eef_pose(self.env)
        st: dict[str, Any] = {
            "eef_from_start_cm": (self.frame.to_level(p - self.p_start) * 100).round(1).tolist(),
            "rot_from_start_deg": np.rad2deg(self.frame.to_level(mat_to_rotvec(R @ self.R_start.T))).round(1).tolist(),
            # grip_site z-axis points along the fingers (out of the hand).
            "approach_dir": self.frame.to_level(R[:, 2]).round(2).tolist(),
            "gripper_opening": round(gripper_opening(self.obs), 3),
            "eef_pixel": None,
            "height_above_surface_cm": None,
        }
        surface = static_surface_below(self.env, p) if self.drop_line else None
        if surface is not None:
            st["height_above_surface_cm"] = round(float(p[2] - surface[2]) * 100, 1)
        # Wrist camera axes for the true-view image (raw[::-1]): right = cam +x, down = cam -y.
        cid = self.env.sim.model.camera_name2id(WRIST_CAMERA)
        cam_R = np.asarray(self.env.sim.data.cam_xmat[cid], dtype=float).reshape(3, 3)
        st["wrist_image_right"] = self.frame.to_level(cam_R[:, 0]).round(2).tolist()
        st["wrist_image_down"] = self.frame.to_level(-cam_R[:, 1]).round(2).tolist()
        if self.has_cameras:
            px = project_world_to_pixel(p[None], self.world_to_pixel)[0]
            st["eef_pixel"] = [int(round(px[0])), int(round(px[1]))]
        return st

    def agentview_image(self) -> np.ndarray | None:
        if not self.has_cameras:
            return None
        img = true_view(self.obs["agentview_image"])
        if self.annotate:
            img = annotate_agentview(img, self.env, self.frame, self.world_to_pixel, drop_line=self.drop_line)
        return img

    def wrist_image(self) -> np.ndarray | None:
        if not self.has_cameras:
            return None
        img = true_view(self.obs["robot0_eye_in_hand_image"])
        if self.annotate:
            img = annotate_wrist(img, self.env, drop_line=self.drop_line)
        return img

    def save_observation(self) -> tuple[pathlib.Path, pathlib.Path] | None:
        """Write the current images as obs/turn_{moves:03d}_*.png; returns absolute paths."""
        import mediapy

        if not self.has_cameras:
            return None
        if not self.keep_all_images:
            for old in self.obs_dir.glob("*.png"):
                old.unlink()
        agent_path = self.obs_dir / f"turn_{self.moves:03d}_agentview.png"
        wrist_path = self.obs_dir / f"turn_{self.moves:03d}_wrist.png"
        mediapy.write_image(agent_path, self.agentview_image())
        mediapy.write_image(wrist_path, self.wrist_image())
        return agent_path, wrist_path

    # -- action --------------------------------------------------------------

    def move(self, cmd_raw: Command, *, steps: int, gain: float, max_cm: float, max_deg: float,
             log_extra: dict[str, Any] | None = None) -> tuple[Command, ExecutionReport, str]:
        """RELATIVE move: clip + execute one command; logs to turns.jsonl.
        Returns (clipped cmd, report, report text)."""
        state_before = self.state()
        cmd = cmd_raw.clipped(max_cm, max_deg)
        self.obs, report = execute_command(self.env, self.obs, self.frame, cmd, steps=steps, gain=gain,
                                           on_step=self._record)
        text = report.summary()
        if cmd.as_dict() != cmd_raw.as_dict():
            text = f"(command was clipped to the per-command limits: {cmd.as_dict()})\n" + text
        return cmd, report, self._finish_move(text, report, state_before, {
            "kind": "relative", "command_raw": cmd_raw.as_dict(), "command": cmd.as_dict(), **(log_extra or {})})

    def move_absolute(self, x: float, y: float, z: float, *, rot: tuple[float, float, float] | None,
                      gripper: float, max_steps: int, gain: float,
                      log_extra: dict[str, Any] | None = None) -> tuple[ExecutionReport, str]:
        """ABSOLUTE move to (x, y, z) cm in the command frame, measured from the gripper's position
        at the start of the episode (the same numbers the state reports). ``rot`` is the target
        rotation from the start orientation as a rotation vector in degrees (None = keep the
        current orientation). Runs until the pose is reached and the gripper has settled, progress
        stalls, the task succeeds, or ``max_steps`` elapse."""
        state_before = self.state()
        p_target = self.p_start + self.frame.to_world(np.array([x, y, z], dtype=float) / 100.0)
        if rot is None:
            R_target = eef_pose(self.env)[1]
        else:
            R_target = rotvec_to_mat(self.frame.to_world(np.deg2rad(rot))) @ self.R_start
        g = float(np.clip(gripper, -1.0, 1.0))
        self.obs, report = drive_to_pose(self.env, self.obs, self.frame, p_target, R_target, g,
                                         max_steps=max_steps, gain=gain, stop_when_reached=True,
                                         on_step=self._record)
        text = (f"stopped after {report.steps_run} sim steps: {report.stop_reason}"
                f"{' (timeout)' if report.stop_reason == 'steps exhausted' else ''}\n" + report.summary())
        target = {"x": x, "y": y, "z": z, "rot": None if rot is None else list(rot), "gripper": g}
        return report, self._finish_move(text, report, state_before, {
            "kind": "absolute", "target": target, "max_steps": max_steps, **(log_extra or {})})

    def _finish_move(self, text: str, report: ExecutionReport, state_before: dict[str, Any],
                     log: dict[str, Any]) -> str:
        if report.success:
            self.success = True
            text += "\nTASK SUCCEEDED."
        with self.log_path.open("a") as f:
            f.write(json.dumps({"turn": self.moves, "state": state_before, **log,
                                "report": dataclasses.asdict(report)}) + "\n")
        self.moves += 1
        return text

    # -- output --------------------------------------------------------------

    def write_video(self, *, fps: int = 20, codec: str = "mpeg4", bps: int = 2_000_000) -> pathlib.Path | None:
        import mediapy

        if not self.video:
            return None
        mediapy.write_image(self.episode_dir / "first_frame.png", self.video[0])
        path = self.episode_dir / "video.mp4"
        mediapy.write_video(path, self.video, fps=fps, codec=codec, bps=bps)
        return path
