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
import dataclasses
import json
import math
import os
import pathlib
import re
import sys
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------------
# JAX memory tweaks must precede `import jax` (and anything pulling JAX in).
# ---------------------------------------------------------------------------

def _limit_jax_mem(limit: float) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", f"{limit:.2f}")


_limit_jax_mem(0.6)


# ---------------------------------------------------------------------------
# Heavy imports — kept after the JAX env tweak above.
# ---------------------------------------------------------------------------

import base64
import io

import cv2
import mediapy
import numpy as np
import tqdm
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi.models import trace_utils as _trace_utils
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "google/gemini-3.1-pro-preview"
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 224  # matches the LeRobot training data resolution
TRACE_OVERLAY_COLOR = (0, 255, 255)        # cyan, matches data_config.overlay_color
TRACE_OVERLAY_THICKNESS = 2                  # matches data_config.overlay_thickness
TRACE_OVERLAY_ENDPOINT_RADIUS = 2.5          # matches data_config.overlay_endpoint_radius
EE_DOT_COLOR = (0, 255, 0)                   # green
SEM_DOT_COLOR = (255, 0, 0)                  # red
ALLOWED_SKILLS = {"PLACE_ON", "PLACE_IN", "PICKUP_FROM", "OPEN", "CLOSE", "TURN_ON", "TURN_OFF"}
SKILL_EXPR_RE = re.compile(r"^([A-Z_]+)\((.*)\)$")
PLAN_ITEM_RE = re.compile(r"\d+\.\s*(?=[A-Z_]+\()")

# Skill-list definitions, copied from py_script/libero_skill_augmentation/common.py
# (stable, small text; bringing it inline avoids a sys.path dependency on the
# annotation package for users who only want inference.)
SKILL_DEFINITIONS_TEXT = """Allowed skill set and exact syntax:
1) PLACE_ON(object1, object2)
- description: place object1 onto object2
- object1: the object being placed
- object2: object that will support object1
- both object1 and object2 must be a single object with no commas in the description
- Example: PLACE_ON(black bowl, plate), PLACE_ON(butter, top of the cabinet)

2) PLACE_IN(object1, object2)
- description: place object1 into object2
- object1: the object being placed
- object2: object that will contain object1
- both object1 and object2 must be a single object with no commas in the description
- Example: PLACE_IN(black bowl, top drawer), PLACE_IN(butter, basket)

3) PICKUP_FROM(object1, object2)
- description: pick up object1 from object2
- object1: the object being picked up. This must be a movable object that can be picked up.
- object2: object that supports object1 originally
- both object1 and object2 must be a single object with no commas in the description
- Example: PICKUP_FROM(red and yellow mug, table), PICKUP_FROM(black bowl, table surface)

4) OPEN(object)
- object: the object being opened
- should be a single object with no commas in the description
- Example: OPEN(middle drawer)

5) CLOSE(object)
- object: the object being closed
- should be a single object with no commas in the description
- Example: CLOSE(drawer)

6) TURN_ON(object)
- object: the object being turned on
- should be a single object with no commas in the description
- Example: TURN_ON(stove)

7) TURN_OFF(object)
- object: the object being turned off
- should be a single object with no commas in the description
- Example: TURN_OFF(stove)

Notes:
- PICKUP_FROM is only for movable objects such as mugs, bowls, and pots that can be picked up and placed.
- Prefer OPEN/CLOSE for drawers or doors when the object is being opened or closed.
- Prefer PLACE_ON for support surfaces and PLACE_IN for containment.
- Only use OPEN if the object is currently closed.
- Only use CLOSE if the object is currently open.
"""

# Coordinate grid Gemini uses to return point_2d. Matches the annotation pipeline.
GEMINI_COORDINATE_GRID = 1024

# LeRobot LIBERO image convention (EE projection lands in this convention; the
# image we feed the model is also in this convention after [::-1, ::-1, :]).
LEROBOT_IMAGE_CONVENTION = "robosuite_agentview_horizontal_flip"


# ---------------------------------------------------------------------------
# Per-trial state / config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SkillContext:
    skill_text: str                     # full parameterized form, e.g. "PICKUP_FROM(black bowl, table)"
    skill_name: str                     # bare verb, e.g. "PICKUP_FROM"
    skill_id: int                       # MoE expert id, 0-4
    semantic_target_pixel: tuple[int, int]   # (x, y) pixel
    semantic_target_xy_norm: tuple[float, float]  # (x, y) in [0, 1]
    cached_trace_xy_norm: np.ndarray | None = None  # (N, 2) in [0, 1]


# ---------------------------------------------------------------------------
# LIBERO env helpers (mirror the existing reference scripts)
# ---------------------------------------------------------------------------

def _get_libero_env(task, resolution: int, seed: int):
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


def _to_lerobot_image(raw_agentview: np.ndarray) -> np.ndarray:
    """Robosuite agentview (raw) -> LeRobot dataset convention used in training.

    Robosuite's agentview is rendered upside-down (OpenGL); the LeRobot dataset
    additionally applies a horizontal flip relative to robosuite raw. The net
    transform is a 180-degree rotation: ``[::-1, ::-1, :]``. This matches what
    the existing libero_atomicVLA_test.py / libero_loraFT_skill_test_v3.py do.
    """
    return np.copy(raw_agentview[::-1, ::-1, :])


def _to_lerobot_wrist(raw_wrist: np.ndarray) -> np.ndarray:
    return np.copy(raw_wrist[::-1, ::-1, :])


def _build_state_vec(obs: dict) -> np.ndarray:
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Camera calibration + EE projection (mirror the annotation pipeline)
# ---------------------------------------------------------------------------

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


def pixel_to_normalized_xy(x: int, y: int, image_width: int, image_height: int) -> tuple[float, float]:
    return (float(x) / max(int(image_width) - 1, 1), float(y) / max(int(image_height) - 1, 1))


# ---------------------------------------------------------------------------
# Checkpoint resolution + policy creation
# ---------------------------------------------------------------------------

def _select_checkpoint_step(path: pathlib.Path) -> pathlib.Path | None:
    path = path.expanduser().resolve()
    if not path.exists():
        return None
    if (path / "params").is_dir():
        return path
    step_dirs = [
        child for child in path.iterdir()
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


def resolve_checkpoint_dir(model_name: str, *, checkpoint_dir: str | os.PathLike[str] | None = None,
                           exp_name: str | None = None) -> pathlib.Path:
    exp_name = exp_name or model_name
    if checkpoint_dir is not None:
        s = os.fspath(checkpoint_dir)
        resolved = (pathlib.Path(download.maybe_download(s)).resolve()
                    if s.startswith("gs://") else pathlib.Path(s).expanduser().resolve())
        step = _select_checkpoint_step(resolved)
        if step is None:
            raise FileNotFoundError(f"Checkpoint path does not contain a valid step dir: {resolved}")
        return step
    searched = []
    for cand in _candidate_checkpoint_dirs(model_name, exp_name):
        searched.append(str(cand))
        step = _select_checkpoint_step(cand)
        if step is not None:
            return step
    raise FileNotFoundError("No TraceVLA checkpoint found. Searched:\n- " + "\n- ".join(searched))


def _load_norm_stats(train_config: _config.TrainConfig, checkpoint_dir: pathlib.Path,
                      assets_dir: str | os.PathLike[str] | None = None):
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


def create_trace_vla_policy(model_name: str, checkpoint_dir: str | os.PathLike[str] | None = None,
                              assets_dir: str | os.PathLike[str] | None = None):
    train_config = _config.get_config(model_name)
    checkpoint_dir = resolve_checkpoint_dir(model_name, checkpoint_dir=checkpoint_dir, exp_name=model_name)
    _, norm_stats = _load_norm_stats(train_config, checkpoint_dir, assets_dir=assets_dir)
    policy = _policy_config.create_trained_trace_vla_policy(
        train_config,
        checkpoint_dir,
        norm_stats=norm_stats,
    )
    return policy, checkpoint_dir, train_config


# ---------------------------------------------------------------------------
# Plan parsing + skill bookkeeping
# ---------------------------------------------------------------------------

def parse_plan(plan_str: str) -> list[str]:
    """Parse a numbered plan string like ``"1. PICKUP_FROM(...) 2. PLACE_ON(...)"`` into
    a list of skill expressions. Mirrors the format the annotation pipeline emits in
    ``skill_annotations.json`` so the trained model sees the same skill strings it
    saw during training.
    """
    cleaned = (plan_str or "").strip()
    if not cleaned:
        return []
    # Split on the "<num>. " markers; the first piece is anything before "1." (likely empty).
    parts = PLAN_ITEM_RE.split(cleaned)
    skills: list[str] = []
    for part in parts:
        item = part.strip().rstrip(",;")
        if not item:
            continue
        match = SKILL_EXPR_RE.match(item)
        if not match:
            continue
        name = match.group(1).strip().upper()
        if name not in ALLOWED_SKILLS:
            continue
        # Re-emit canonical form (keeps spacing inside the parens as Gemini wrote it).
        skills.append(item)
    return skills


def skill_to_atomic_id(skill_text: str) -> int:
    return _trace_utils.skill_to_expert_id(skill_text)


def skill_name(skill_text: str) -> str:
    """Strip the (...) parameters off a skill expression, leaving the bare verb."""
    skill_text = (skill_text or "").strip()
    match = SKILL_EXPR_RE.match(skill_text)
    if not match:
        return skill_text.split("(", 1)[0].strip().upper()
    return match.group(1).strip().upper()


# ---------------------------------------------------------------------------
# OpenRouter (Gemini) querying
# ---------------------------------------------------------------------------

def _build_image_data_url(frame: np.ndarray, *, width: int | None = None, height: int | None = None) -> str:
    image = Image.fromarray(frame.astype(np.uint8))
    if width is not None and height is not None:
        image = image.resize((int(width), int(height)), resample=Image.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_request_payload(*, model: str, prompt: str, image_data_url: str, schema: dict[str, Any] | None,
                           temperature: float, max_tokens: int) -> dict[str, Any]:
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    if schema is not None:
        payload["response_format"] = {"type": "json_schema", "json_schema": schema}
        payload["provider"] = {"require_parameters": True}
        payload["plugins"] = [{"id": "response-healing"}]
    return payload


def _call_openrouter(api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openrouter.ai",
        "X-Title": "libero-trace-vla-inference",
    }
    request = Request(OPENROUTER_URL, data=json.dumps(payload).encode("utf-8"),
                      headers=headers, method="POST")
    with urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_message_content(response_json: dict[str, Any]) -> str:
    message = response_json["choices"][0]["message"]["content"]
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        text_parts = [str(item.get("text", "")) for item in message
                      if isinstance(item, dict) and item.get("type") == "text"]
        return "\n".join(text_parts)
    raise ValueError(f"Unexpected message content shape: {message!r}")


def query_openrouter_json(*, api_key: str, model: str, prompt: str, image_data_url: str,
                           schema: dict[str, Any] | None,
                           normalizer: Callable[[Any], dict[str, Any]],
                           temperature: float = 0.0, max_tokens: int = 16000,
                           max_retries: int = 4, retry_sleep_seconds: float = 8.0,
                           query_name: str = "openrouter_query") -> dict[str, Any]:
    """Mirror the annotation pipeline's retry / structured-output logic in a single helper."""
    last_error: Exception | None = None
    attempted_fallback = False
    for attempt in range(1, max_retries + 1):
        request_prompt = prompt
        if last_error is not None:
            request_prompt = (
                prompt
                + "\n\nThe previous output failed validation. Regenerate from scratch and fix this issue:\n"
                + f"- {last_error}\n"
                + "Return valid JSON only."
            )
        payload = _build_request_payload(
            model=model, prompt=request_prompt, image_data_url=image_data_url,
            schema=schema if not attempted_fallback else None,
            temperature=temperature, max_tokens=max_tokens,
        )
        try:
            t0 = time.time()
            response_json = _call_openrouter(api_key=api_key, payload=payload)
            print(f"    {query_name}: Gemini response in {time.time() - t0:.1f}s", flush=True)
            parsed = json.loads(_strip_json_fence(_extract_message_content(response_json)))
            normalized = normalizer(parsed)
            return {"parsed_output": parsed, "normalized": normalized}
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if (schema is not None and not attempted_fallback
                    and exc.code in (400, 422) and "response_format" in body):
                attempted_fallback = True
                print(f"    {query_name}: structured output rejected; retrying without json_schema", flush=True)
                last_error = ValueError(body)
                continue
            last_error = RuntimeError(f"HTTP {exc.code}: {body}")
        except URLError as exc:
            last_error = RuntimeError(f"Network error: {exc}")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            last_error = exc
        if attempt < max_retries:
            sleep_seconds = retry_sleep_seconds * attempt
            print(f"    retry {attempt}/{max_retries - 1} for {query_name}: {last_error}", flush=True)
            time.sleep(sleep_seconds)
    assert last_error is not None
    raise last_error


# ---------- Plan generation prompt (initial-image only, no video) ----------

PLAN_SCHEMA = {
    "name": "libero_initial_skill_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {"type": "string"},
            "plan": {"type": "string"},
        },
        "required": ["plan"],
    },
}


def build_plan_prompt(task_instruction: str) -> str:
    return f"""You are planning a robot manipulation task on the LIBERO benchmark.

You are given:
- the natural-language task instruction for this episode
- a single image of the INITIAL scene (the same agent-view image the robot sees)

Your task:
1. Decide the ordered sequence of atomic skills that, executed in order, will accomplish the instruction from the visible initial state.
2. Use ONLY the allowed skills below and follow the syntax exactly.

Task instruction:
{task_instruction}

{SKILL_DEFINITIONS_TEXT}

Planning rules:
- The first skill cannot be PLACE_ON or PLACE_IN — at the start the gripper holds nothing.
- Object descriptions must be short, specific, contain no commas, and reuse appearance + positional cues from the instruction (e.g. "left white mug").
- Use the visible initial scene to disambiguate (e.g. confirm whether a drawer is currently open or closed before deciding OPEN/CLOSE).
- Do not include skills that the instruction does not require.
- Keep the plan as short as possible while still completing the instruction.

Output format:
- Return JSON only.
- "plan" must be a single numbered string of the exact form:
  "1. PICKUP_FROM(white mug, table) 2. PLACE_ON(white mug, left plate)"

Example:
{{
  "reasoning": "the drawer in the scene is closed; the bowl sits on the table; the task is to put the bowl in the top drawer",
  "plan": "1. OPEN(top drawer of the cabinet) 2. PICKUP_FROM(black bowl, table) 3. PLACE_IN(black bowl, top drawer of the cabinet) 4. CLOSE(top drawer of the cabinet)"
}}
"""


def _normalize_plan_response(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Plan response must be a JSON object, got {type(raw).__name__}.")
    plan = str(raw.get("plan", "")).strip()
    if not plan:
        raise ValueError("Plan response is missing 'plan'.")
    skills = parse_plan(plan)
    if not skills:
        raise ValueError(f"Plan response could not be parsed into any allowed skills: {plan!r}")
    return {"plan": plan, "skills": skills}


# ---------- Semantic-target prompt (per-skill, single image, no hint video) ----------

SEMANTIC_POINT_SCHEMA = {
    "name": "libero_semantic_target_point",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"type": "string"},
            "reasoning": {"type": "string"},
            "object_id": {"type": "string"},
            "label": {"type": "string"},
            "point_2d": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "number"},
            },
        },
        "required": ["status", "object_id", "label", "point_2d"],
    },
}


def _semantic_target_policy_text(skill_text: str) -> str:
    name = skill_name(skill_text)
    if name == "PICKUP_FROM":
        return ("Choose the object that will be picked up. Put the point on the visible body of that object.")
    if name in {"PLACE_ON", "PLACE_IN"}:
        return ("Choose the destination where the carried object should be placed. Put the point on the receiving "
                "surface, inside the receiving container, or at the visually best placement location.")
    if name in {"OPEN", "CLOSE"}:
        return ("Choose the manipulation handle, lip, pull tab, or graspable edge used to open or close the object. "
                "Do not mark the center of the whole drawer or cabinet if a handle is visible.")
    if name in {"TURN_ON", "TURN_OFF"}:
        return "Choose the knob, switch, button, or control surface that the gripper should actuate."
    raise ValueError(f"Unsupported skill for semantic target policy: {skill_text}")


def build_semantic_point_prompt(*, task_instruction: str, plan_str: str, skill_text: str,
                                  image_width: int, image_height: int,
                                  coordinate_grid: int = GEMINI_COORDINATE_GRID) -> str:
    return f"""You are annotating a LIBERO robot manipulation skill from a single agent-view image (the CURRENT scene).

Note that any "left", "right", "front", and "back" descriptions should be with respect to the robot's perspective, which is opposite to the image's perspective.

Episode instruction:
{task_instruction}

Full skill plan:
{plan_str}

Current skill (the next one to execute):
- skill: {skill_text}
- sent image size: width={image_width}, height={image_height}

Your task is to choose one semantically clarifying target point for this skill on the CURRENT image.
{_semantic_target_policy_text(skill_text)}

Semantic hints:
1. The cookie box is the small, dark brown box
2. The ramekin is black-silver. 

Coordinate rules:
- Return coordinates on a fixed output grid width={coordinate_grid}, height={coordinate_grid}.
- The full image spans x=0..{coordinate_grid} left-to-right and y=0..{coordinate_grid} top-to-bottom.
- Return point_2d in row-column order: [y, x], not [x, y].
- The point must be on the visible object (or contact location), strictly inside the image bounds.
- "object_id" should be a brief description of the object you point at, derived from the skill arguments.
- "label" should be the literal string "semantic_target".
- Return JSON only — no markdown, prose, or code fences outside the JSON object.

Output shape:
{{
  "status": "OK",
  "reasoning": "brief visual justification",
  "object_id": "<short object description>",
  "label": "semantic_target",
  "point_2d": [y, x]
}}
"""


def _normalize_semantic_point_response(raw: Any, *, image_width: int, image_height: int,
                                         coordinate_grid: int = GEMINI_COORDINATE_GRID) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Semantic point response must be a JSON object, got {type(raw).__name__}.")
    status = str(raw.get("status", "")).strip().upper()
    if status != "OK":
        raise ValueError(f"Semantic point status must be OK, got {raw.get('status')!r}.")
    label = str(raw.get("label", "")).strip()
    if label != "semantic_target":
        raise ValueError(f"Semantic point label must be 'semantic_target', got {label!r}.")
    point = raw.get("point_2d")
    if not isinstance(point, list) or len(point) != 2:
        raise ValueError(f"point_2d must be a [y, x] list, got {point!r}.")
    row = float(point[0]); col = float(point[1])
    if not (0 <= row <= coordinate_grid and 0 <= col <= coordinate_grid):
        raise ValueError(f"point_2d {point!r} is outside [0, {coordinate_grid}].")
    x = int(round((col / float(coordinate_grid)) * image_width))
    y = int(round((row / float(coordinate_grid)) * image_height))
    x = max(0, min(int(image_width) - 1, x))
    y = max(0, min(int(image_height) - 1, y))
    return {
        "status": "OK",
        "object_id": str(raw.get("object_id", "")),
        "label": label,
        "point_pixel": (x, y),
    }


# ---------- Public Gemini wrappers used by the main loop ----------

def gemini_initial_plan(api_key: str, openrouter_model: str, *, task_instruction: str,
                          initial_scene_image: np.ndarray) -> dict[str, Any]:
    """Ask Gemini for a skill plan from the task instruction + initial scene image."""
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; cannot generate plan via Gemini.")
    image_data_url = _build_image_data_url(initial_scene_image)
    prompt = build_plan_prompt(task_instruction=task_instruction)
    result = query_openrouter_json(
        api_key=api_key, model=openrouter_model, prompt=prompt,
        image_data_url=image_data_url, schema=PLAN_SCHEMA,
        normalizer=_normalize_plan_response, query_name="initial_plan",
    )
    return result["normalized"]


def gemini_semantic_target(api_key: str, openrouter_model: str, *, task_instruction: str,
                             plan_str: str, skill_text: str,
                             current_scene_image: np.ndarray) -> tuple[int, int]:
    """Query Gemini for the semantic target point on the current scene for the given skill."""
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; cannot query semantic target via Gemini.")
    H, W, _ = current_scene_image.shape
    image_data_url = _build_image_data_url(current_scene_image)
    prompt = build_semantic_point_prompt(
        task_instruction=task_instruction, plan_str=plan_str, skill_text=skill_text,
        image_width=W, image_height=H,
    )
    result = query_openrouter_json(
        api_key=api_key, model=openrouter_model, prompt=prompt,
        image_data_url=image_data_url, schema=SEMANTIC_POINT_SCHEMA,
        normalizer=lambda raw: _normalize_semantic_point_response(
            raw, image_width=W, image_height=H,
        ),
        query_name=f"semantic_target[{skill_name(skill_text)}]",
    )
    return tuple(result["normalized"]["point_pixel"])


# ---------------------------------------------------------------------------
# Building obs dicts for the policy
# ---------------------------------------------------------------------------

def _make_base_obs(libero_obs: dict, calib: dict[str, Any]) -> dict[str, Any]:
    """Common obs fields shared across plan-mode and execution-mode invocations."""
    base_image = _to_lerobot_image(libero_obs["agentview_image"])
    wrist_image = _to_lerobot_wrist(libero_obs["robot0_eye_in_hand_image"])
    state_vec = _build_state_vec(libero_obs)
    return {
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "observation/state": state_vec,
    }


def make_planning_obs(libero_obs: dict, calib: dict[str, Any], skill_ctx: SkillContext,
                       task_description: str, ee_xy_norm: tuple[float, float]) -> dict[str, Any]:
    """Build the obs dict the trace expert (planning mode) needs."""
    base = _make_base_obs(libero_obs, calib)
    base.update({
        "atomic_token": float(skill_ctx.skill_id),
        "semantic_target_xy": np.asarray(skill_ctx.semantic_target_xy_norm, dtype=np.float32),
        "current_ee_xy": np.asarray(ee_xy_norm, dtype=np.float32),
        # Provide both forms so TraceTokenizePrompt picks the parameterized text first.
        "skill_text": skill_ctx.skill_text,
        "skill_name": skill_ctx.skill_name,
        "prompt": task_description,
        # Marker fields to keep TraceObservation typecheck happy when the policy
        # tries to read these (with `.get(...)` they default to None on missing).
        "has_trace": True,
        "has_overlay": False,
        "progress": 0.0,
    })
    return base


def make_execution_obs(libero_obs: dict, calib: dict[str, Any], skill_ctx: SkillContext,
                        task_description: str, ee_xy_norm: tuple[float, float],
                        overlay_image: np.ndarray) -> dict[str, Any]:
    """Build the obs dict the action expert + completion head need."""
    base = make_planning_obs(libero_obs, calib, skill_ctx, task_description, ee_xy_norm)
    base["observation/overlay_image"] = overlay_image
    base["has_overlay"] = True
    return base


# ---------------------------------------------------------------------------
# Trace overlay rendering + per-frame visualization
# ---------------------------------------------------------------------------

def _draw_dot(img_uint8_hwc: np.ndarray, x: int, y: int, color_rgb: tuple[int, int, int],
               *, radius: float = 4.0) -> None:
    """In-place antialiased filled disk (private helper from trace_utils)."""
    _trace_utils._filled_disk(img_uint8_hwc, float(x), float(y), float(radius), color_rgb)


def render_overlay_image(base_image: np.ndarray, trace_xy_norm: np.ndarray) -> np.ndarray:
    """Draw the cyan trace polyline onto a copy of the base image — same code path
    as the dataset's overlay rendering, so the action expert sees the same style."""
    return _trace_utils.draw_polyline_overlay(
        base_image,
        np.asarray(trace_xy_norm, dtype=np.float32),
        color=TRACE_OVERLAY_COLOR,
        line_thickness=TRACE_OVERLAY_THICKNESS,
        endpoint_radius=TRACE_OVERLAY_ENDPOINT_RADIUS,
    )


def render_video_frame(base_image: np.ndarray, *, trace_xy_norm: np.ndarray | None,
                        sem_pixel: tuple[int, int] | None, ee_pixel: tuple[int, int] | None,
                        skill_text: str, progress: float | None,
                        consecutive_high: int, completion_threshold: float) -> np.ndarray:
    """Compose the per-frame visualization (overlay + dots + text) for the video.

    The "what model sees" overlay (cyan polyline from `render_overlay_image`) is the
    background. We then add green/red dots for the EE/sem keypoints and a top-left
    text block annotating skill + progress, so a single video frame contains
    everything needed to debug the rollout.
    """
    img = np.array(base_image, dtype=np.uint8, copy=True)
    if trace_xy_norm is not None:
        img = render_overlay_image(img, trace_xy_norm)
    if ee_pixel is not None:
        _draw_dot(img, ee_pixel[0], ee_pixel[1], EE_DOT_COLOR, radius=4.0)
    if sem_pixel is not None:
        _draw_dot(img, sem_pixel[0], sem_pixel[1], SEM_DOT_COLOR, radius=4.0)

    # Text block.
    progress_str = "n/a" if progress is None else f"{progress:.3f}"
    lines = [
        f"Skill: {skill_text}",
        f"Progress: {progress_str}  (>= {completion_threshold:.2f} streak: {consecutive_high})",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.30
    thickness = 1
    margin = 6
    line_metrics = [cv2.getTextSize(line, font, font_scale, thickness) for line in lines]
    text_w = max(m[0][0] for m in line_metrics)
    text_h = max(m[0][1] for m in line_metrics)
    baseline = max(m[1] for m in line_metrics)
    line_gap = 3
    block_h = len(lines) * text_h + (len(lines) - 1) * line_gap

    x = margin
    y = margin + text_h
    cv2.rectangle(
        img,
        (max(0, x - 4), max(0, y - text_h - 4)),
        (min(img.shape[1] - 1, x + text_w + 4),
         min(img.shape[0] - 1, y - text_h + block_h + baseline + 4)),
        (0, 0, 0), -1,
    )
    for idx, line in enumerate(lines):
        line_y = y + idx * (text_h + line_gap)
        cv2.putText(img, line, (x, line_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TraceVLA LIBERO inference + benchmark.")
    p.add_argument("--model-name", default="trace_vla_lora",
                    help="TrainConfig name (trace_vla / trace_vla_lora).")
    p.add_argument("--checkpoint-dir", default="/work/hdd/bgtb/zhong2/checkpoints/trace_vla_lora/trace_vla_lora/30000",
                    help="Path to a TraceVLA checkpoint step dir (containing params/). If omitted, "
                         "we search standard locations under the repo and /work/hdd/bgtb/$USER/checkpoints.")
    p.add_argument("--assets-dir", default=None,
                    help="Optional override for the assets dir (where norm_stats live). "
                         "Defaults to <checkpoint_dir>/assets.")
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

    # Loop / replanning controls.
    p.add_argument("--replan-every-chunks", type=int, default=5,
                    help="Re-sample the trace every N action chunks within a skill (default 5).")
    p.add_argument("--actions-per-chunk", type=int, default=5,
                    help="Number of env steps to consume per action chunk before calling the model again. "
                         "Defaults to action_horizon // 2 = 5 for the standard trace_vla configs.")
    p.add_argument("--completion-check-interval", type=int, default=1,
                    help="Run a standalone completion-progress check every N action chunks (default 1 = "
                         "every chunk). When the check fires, the script runs predict_completion BEFORE "
                         "sample_actions_and_completion so it can abort the action denoising on a triggering frame.")
    p.add_argument("--completion-threshold", type=float, default=0.9,
                    help="Predicted progress >= this counts as 'high'.")
    p.add_argument("--consecutive-required", type=int, default=2,
                    help="Number of consecutive 'high' progress checks required before advancing to the next skill.")

    # OpenRouter / Gemini.
    p.add_argument("--openrouter-model", default=DEFAULT_OPENROUTER_MODEL)
    p.add_argument("--openrouter-api-key", default=None,
                    help="OpenRouter API key. Defaults to env OPENROUTER_API_KEY. Only used if --plan is empty "
                         "OR for per-skill semantic-target queries.")

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

def run_episode(*, env, calib: dict[str, Any], policy, task_description: str,
                  skill_list: list[str], plan_str: str, args: argparse.Namespace,
                  api_key: str) -> tuple[bool, list[np.ndarray]]:
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
    H = W = LIBERO_ENV_RESOLUTION
    frames: list[np.ndarray] = []
    env_done = False
    total_steps = 0

    # Settle the env (matches the AtomicVLA reference).
    obs = None
    for _ in range(20):
        obs, _r, _d, _i = env.step(LIBERO_DUMMY_ACTION)

    for skill_idx, skill_text in enumerate(skill_list):
        if env_done or total_steps >= args.episode_length:
            break

        # --- (1) Query Gemini for the semantic target on the current frame ---
        current_scene = _to_lerobot_image(obs["agentview_image"])
        try:
            sem_pixel = gemini_semantic_target(
                api_key, args.openrouter_model,
                task_instruction=task_description, plan_str=plan_str, skill_text=skill_text,
                current_scene_image=current_scene,
            )
        except Exception as exc:
            print(f"  [skill={skill_idx} {skill_text}] semantic-target query failed: {exc}", flush=True)
            print(f"  -> falling back to image center.", flush=True)
            sem_pixel = (W // 2, H // 2)
        sem_xy_norm = pixel_to_normalized_xy(sem_pixel[0], sem_pixel[1], W, H)
        skill_ctx = SkillContext(
            skill_text=skill_text,
            skill_name=skill_name(skill_text),
            skill_id=skill_to_atomic_id(skill_text),
            semantic_target_pixel=sem_pixel,
            semantic_target_xy_norm=sem_xy_norm,
        )
        print(f"  [skill={skill_idx} '{skill_ctx.skill_text}'] expert={skill_ctx.skill_id} "
              f"sem_pixel={sem_pixel}", flush=True)

        consecutive_high = 0
        chunks_since_replan = args.replan_every_chunks   # force replan on the first iteration
        chunks_since_completion_check = 0
        last_progress: float | None = None

        # Per-skill action loop. Bail out when (a) env done, (b) episode length cap,
        # or (c) the consecutive-high threshold fires.
        while total_steps < args.episode_length:
            # --- Project current EE to image pixel ---
            ee_world = np.asarray(obs["robot0_eef_pos"], dtype=float)
            ee_x, ee_y, _ee_depth = project_ee_world_to_pixel(ee_world, calib)
            ee_xy_norm = pixel_to_normalized_xy(ee_x, ee_y, W, H)

            # --- (2) Re-plan trace if needed ---
            if chunks_since_replan >= args.replan_every_chunks:
                planning_obs = make_planning_obs(obs, calib, skill_ctx, task_description, ee_xy_norm)
                trace = policy.sample_trace(planning_obs)        # (N, 2) in [0, 1]
                trace = np.asarray(trace, dtype=np.float32)
                # Defensive clamp (the model already does this, but it's cheap to repeat).
                trace = np.clip(trace, 0.0, 1.0)
                skill_ctx.cached_trace_xy_norm = trace
                chunks_since_replan = 0
                print(f"    chunk-replan: trace start={trace[0].tolist()}, end={trace[-1].tolist()}", flush=True)

            # --- Build overlay image for execution-mode forward ---
            base_image = _to_lerobot_image(obs["agentview_image"])
            overlay_image = render_overlay_image(base_image, skill_ctx.cached_trace_xy_norm)

            # --- (3a) Optional completion check BEFORE action gen ---
            should_check = (chunks_since_completion_check % args.completion_check_interval == 0)
            if should_check:
                exec_obs = make_execution_obs(obs, calib, skill_ctx, task_description, ee_xy_norm, overlay_image)
                progress_only = float(np.asarray(policy.predict_completion(exec_obs)))
                last_progress = progress_only
                consecutive_high = consecutive_high + 1 if progress_only >= args.completion_threshold else 0
                if consecutive_high >= args.consecutive_required:
                    print(f"    completion advance: progress={progress_only:.3f} streak={consecutive_high} "
                          f"-> next skill", flush=True)
                    # Render one final frame so the abort moment is visible in the video.
                    frames.append(render_video_frame(
                        base_image, trace_xy_norm=skill_ctx.cached_trace_xy_norm,
                        sem_pixel=skill_ctx.semantic_target_pixel, ee_pixel=(ee_x, ee_y),
                        skill_text=skill_ctx.skill_text, progress=last_progress,
                        consecutive_high=consecutive_high, completion_threshold=args.completion_threshold,
                    ))

                    # still have available skill, proceed to next skill
                    if skill_idx < len(skill_list) - 1:
                        break
                    
                    # no more available skill, keep acting until env is done
                    else:
                        ...

            chunks_since_completion_check += 1

            # --- (3b) Action generation (also returns a fresh progress estimate) ---
            exec_obs = make_execution_obs(obs, calib, skill_ctx, task_description, ee_xy_norm, overlay_image)
            infer_result = policy.infer(exec_obs)
            actions = np.asarray(infer_result["actions"])      # (action_horizon, 7)
            last_progress = float(np.asarray(infer_result["progress"]))
            chunks_since_replan += 1

            # --- (4) Execute the action chunk in env, saving a frame per env step ---
            chunk_steps = min(args.actions_per_chunk, actions.shape[0])
            for k in range(chunk_steps):
                if total_steps >= args.episode_length:
                    break
                act = np.asarray(actions[k], dtype=np.float32).copy()
                obs, _reward, env_done, _info = env.step(act)
                total_steps += 1

                # Re-project EE for the new obs so the dot tracks frame-to-frame.
                ee_world_after = np.asarray(obs["robot0_eef_pos"], dtype=float)
                ee_x, ee_y, _ = project_ee_world_to_pixel(ee_world_after, calib)
                base_after = _to_lerobot_image(obs["agentview_image"])
                frame = render_video_frame(
                    base_after,
                    trace_xy_norm=skill_ctx.cached_trace_xy_norm,
                    sem_pixel=skill_ctx.semantic_target_pixel,
                    ee_pixel=(ee_x, ee_y),
                    skill_text=skill_ctx.skill_text,
                    progress=last_progress,
                    consecutive_high=consecutive_high,
                    completion_threshold=args.completion_threshold,
                )
                frames.append(frame)

                if env_done:
                    print(f"    env reports done at step {total_steps}", flush=True)
                    break
            if env_done:
                break

        if env_done:
            break

    return env_done, frames


def main() -> None:
    args = _parse_args()
    if abs(args.jax_mem_fraction - 0.6) > 1e-6:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{args.jax_mem_fraction:.2f}"

    api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not args.plan and not api_key:
        raise SystemExit(
            "Either pass --plan to hardcode the skill plan, or set OPENROUTER_API_KEY "
            "(or --openrouter-api-key) so plan generation can use Gemini."
        )

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite not in benchmark_dict:
        raise SystemExit(f"Unknown task suite: {args.task_suite!r}. Available: {list(benchmark_dict)!r}")
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TraceVLA policy ({args.model_name})...", end="", flush=True)
    policy, checkpoint_dir, _ = create_trace_vla_policy(
        args.model_name, checkpoint_dir=args.checkpoint_dir, assets_dir=args.assets_dir,
    )
    print(" Done.", flush=True)
    print(f"Using checkpoint: {checkpoint_dir}", flush=True)

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
            initial_scene_lerobot = _to_lerobot_image(obs["agentview_image"])

            # --- Plan source: --plan flag or Gemini ---
            if args.plan:
                plan_str = args.plan.strip()
                skills = parse_plan(plan_str)
                if not skills:
                    raise SystemExit(f"Could not parse --plan {args.plan!r} into any allowed skills.")
                print(f"[Task {task_id} ep {episode_idx}] using --plan: {plan_str}", flush=True)
            else:
                plan_payload = gemini_initial_plan(
                    api_key, args.openrouter_model,
                    task_instruction=task_description, initial_scene_image=initial_scene_lerobot,
                )
                plan_str = plan_payload["plan"]
                skills = plan_payload["skills"]
                print(f"[Task {task_id} ep {episode_idx}] Gemini plan: {plan_str}", flush=True)

            done, frames = run_episode(
                env=env, calib=calib, policy=policy, task_description=task_description,
                skill_list=skills, plan_str=plan_str, args=args, api_key=api_key,
            )
            overall_success.append(1 if done else 0)
            if not done:
                failure_records.append({"task_id": task_id, "episode_idx": episode_idx,
                                          "task_prompt": task_description, "plan": plan_str})

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
