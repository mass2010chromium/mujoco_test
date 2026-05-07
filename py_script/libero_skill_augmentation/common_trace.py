from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import textwrap
from pathlib import Path
from typing import Any

from common import (  # noqa: F401
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_REPO_ID,
    EPISODE_FILE_RE,
    as_uint8_hwc,
    episode_shard_path,
    list_episode_shards,
    load_episode_frame,
    load_json,
    resolve_dataset_root,
    save_json_atomic,
    utc_now,
)


TARGET_TRACE_PROMPT_VERSION = "libero_target_trace_v1"
TARGET_TRACE_COORDINATE_RESOLUTION = 1024
DEFAULT_TRACE_FRAME_COUNT = 50

SEMANTIC_TARGET_LABEL = "semantic_target"
CONTACT_POINT_LABEL = "contact_point"
PREDICTION_TRACE_LABEL = "prediction_contact_trace"
EXTRACTION_TRACE_LABEL = "extraction_contact_trace"
END_EFFECTOR_TRACE_LABEL = "end_effector_trace"
END_EFFECTOR_TRACE_KIND = "end_effector_projection"
DEFAULT_EE_TRACE_MAX_STEP_DELTA_PIXELS = 80.0
DEFAULT_EE_TRACE_MAX_OUT_OF_BOUNDS_FRACTION = 1.00
DEFAULT_EE_REPROJECTION_MAX_ERROR_METERS = 1e-6
DEFAULT_EE_ROUNDED_REPROJECTION_MAX_ERROR_METERS = 0.02
LEROBOT_LIBERO_IMAGE_CONVENTION = "robosuite_agentview_horizontal_flip"
ROBOSUITE_AGENTVIEW_OPENGL_IMAGE_CONVENTION = "robosuite_agentview_opengl"
LEGACY_LEROBOT_LIBERO_IMAGE_CONVENTION = "sim_agentview_rotated_180_degrees"
NO_TRANSFORM_IMAGE_CONVENTIONS = {
    ROBOSUITE_AGENTVIEW_OPENGL_IMAGE_CONVENTION,
}
HORIZONTAL_FLIP_IMAGE_CONVENTIONS = {
    LEROBOT_LIBERO_IMAGE_CONVENTION,
}
ROTATED_180_IMAGE_CONVENTIONS = {
    LEGACY_LEROBOT_LIBERO_IMAGE_CONVENTION,
}

SKILL_EXPR_RE = re.compile(r"^([A-Z_]+)\((.*)\)$")


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"Expected a positive integer, got {value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"Expected a positive integer, got {parsed}.")
    return parsed


def parse_skill_name(skill: str) -> str:
    match = SKILL_EXPR_RE.match(" ".join(str(skill).strip().split()))
    if not match:
        raise ValueError(f"Invalid skill expression: {skill!r}")
    return match.group(1)


def normalize_skill_for_verifier(skill: str) -> str:
    """Match the left/right convention used by the original target generation script."""
    splits = re.split(r"([^a-zA-Z0-9]left[^a-zA-Z0-9]|[^a-zA-Z0-9]right[^a-zA-Z0-9])", skill)
    for idx in range(1, len(splits), 2):
        token = splits[idx]
        if len(token) == 6:
            splits[idx] = token[0] + "right" + token[-1]
        elif len(token) == 7:
            splits[idx] = token[0] + "left" + token[-1]
    return "".join(splits)


def load_skill_annotation_episodes(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Skill annotation JSON must be an object: {path}")

    if "episode_index" in data and "segments" in data:
        episode = dict(data)
        return data, {int(episode["episode_index"]): episode}

    episodes: dict[int, dict[str, Any]] = {}
    for key in sorted((item for item in data if isinstance(item, str) and item.isdigit()), key=lambda item: int(item)):
        value = data[key]
        if not isinstance(value, dict):
            continue
        episode = dict(value)
        episode.setdefault("episode_index", int(key))
        episodes[int(key)] = episode
    if not episodes:
        raise ValueError(f"No episode annotations found in {path}.")
    return data, episodes


def validate_skill_episode_shape(skill_episode: dict[str, Any]) -> None:
    segments = skill_episode.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"Episode {skill_episode.get('episode_index')} has no skill segments.")

    prev_end = 0
    for idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"Segment {idx} is not an object: {segment!r}.")
        start_step = int(segment["start_step"])
        end_step = int(segment["end_step"])
        if start_step != prev_end:
            raise ValueError(f"Segment {idx} starts at {start_step}, expected {prev_end}.")
        if end_step <= start_step:
            raise ValueError(f"Segment {idx} has non-positive length: {segment!r}.")
        prev_end = end_step

    if prev_end != int(skill_episode["num_steps"]):
        raise ValueError(
            f"Episode {skill_episode.get('episode_index')} segment steps end at {prev_end}, "
            f"but num_steps is {skill_episode['num_steps']}."
        )


def resolve_dataset_root_from_skill_data(args: Any, skill_data: dict[str, Any]) -> Path:
    if args.dataset_root is not None:
        return resolve_dataset_root(args.repo_id, args.dataset_root)

    recorded_root = skill_data.get("dataset_root")
    if recorded_root is not None:
        recorded_path = Path(str(recorded_root)).expanduser()
        if (recorded_path / "meta" / "info.json").exists():
            return resolve_dataset_root(args.repo_id, recorded_path)

    return resolve_dataset_root(args.repo_id, None)


def scene_object_to_dict(scene_object: Any) -> dict[str, Any]:
    if scene_object is None:
        raise ValueError("scene_object is None.")
    if hasattr(scene_object, "to_dict"):
        return dict(scene_object.to_dict(include_grounding=False))
    if isinstance(scene_object, dict):
        return dict(scene_object)
    raise TypeError(f"Unsupported scene object type: {type(scene_object).__name__}")


def grounded_action_to_dict(action: Any) -> dict[str, Any]:
    return {
        "name": str(action.name.value),
        "parameters": [str(item.value) for item in action.grounding],
    }


def resolve_skill_object_ids(action: Any) -> dict[str, str]:
    action_name = str(action.name.value)
    params = [str(item.value) for item in action.grounding]
    if action_name in {"pickup_from", "open", "close", "turn_on", "turn_off"}:
        return {
            "semantic_target_object_id": params[0],
            "contact_object_id": params[0],
        }
    if action_name in {"place_on", "place_in"}:
        return {
            "semantic_target_object_id": params[2],
            "contact_object_id": params[0],
        }
    raise ValueError(f"Unsupported grounded action for target trace extraction: {action_name}")


def parse_skill_object_descriptions(skill: str) -> dict[str, str]:
    """Parse a LIBERO skill expression and return target object descriptions.

    Returns a dict with the same shape as :func:`resolve_skill_object_ids` but
    populated from the skill text alone, so it does not depend on the symbolic
    scene-graph / PDDL verifier. The values are the natural-language object
    descriptions taken straight from the skill arguments (e.g. ``"white mug"``).
    """
    match = SKILL_EXPR_RE.match(" ".join(str(skill).strip().split()))
    if not match:
        raise ValueError(f"Invalid skill expression: {skill!r}")
    name = match.group(1).upper()
    raw_args = match.group(2)
    params = [piece.strip() for piece in raw_args.split(",")]
    if not params or any(not piece for piece in params):
        raise ValueError(f"Could not parse skill arguments from {skill!r}.")
    if name in {"PICKUP_FROM", "OPEN", "CLOSE", "TURN_ON", "TURN_OFF"}:
        target = params[0]
        return {
            "semantic_target_object_id": target,
            "contact_object_id": target,
        }
    if name in {"PLACE_ON", "PLACE_IN"}:
        if len(params) < 2:
            raise ValueError(f"{name} skill expects 2 arguments, got: {skill!r}")
        carried, destination = params[0], params[1]
        return {
            "semantic_target_object_id": destination,
            "contact_object_id": carried,
        }
    raise ValueError(f"Unsupported skill for target trace extraction: {name}")


def synthesize_skill_object(description: str) -> dict[str, Any]:
    """Build the minimal object record passed to the prompt builders when the
    full scene-graph object dict is not available.
    """
    return {"object_id": str(description), "description": str(description)}


def normalize_task_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def libero_language_from_bddl_stem(stem: str) -> str:
    if not stem:
        raise ValueError("Cannot extract LIBERO language from an empty BDDL stem.")
    if stem[0].isupper():
        scene_pos = stem.find("SCENE")
        if scene_pos < 0:
            raise ValueError(f"Expected LIBERO-100 scene prefix in BDDL stem: {stem!r}")
        language_start = scene_pos + (8 if "SCENE10" in stem else 7)
        language = " ".join(stem[language_start:].split("_"))
    else:
        language = " ".join(stem.split("_"))
    return language.strip()


def libero_language_aliases(language: str) -> list[str]:
    normalized = normalize_task_text(language)
    aliases: list[str] = []
    pickup_match = re.match(r"^pick up (.+) and (?:put|place) it (.+)$", normalized)
    if pickup_match:
        aliases.append(f"put {pickup_match.group(1)} {pickup_match.group(2)}")
    if normalized == "close the microwave":
        aliases.append("move away the yellow and white mug to close the microwave door")
    return aliases


def _candidate_bddl_roots(explicit_root: str | os.PathLike[str] | None = None) -> list[Path]:
    roots: list[Path] = []
    if explicit_root is not None:
        roots.append(Path(explicit_root).expanduser())

    try:
        from libero.libero import get_libero_path

        roots.append(Path(get_libero_path("bddl_files")).expanduser())
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    roots.extend(
        [
            repo_root / "lerobot-libero" / "libero" / "libero" / "bddl_files",
            repo_root / "pace" / "openpi" / "third_party" / "libero" / "libero" / "libero" / "bddl_files",
        ]
    )

    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        unique_roots.append(resolved)
    return unique_roots


def build_bddl_language_index(
    explicit_root: str | os.PathLike[str] | None = None,
) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    suite_priority = {
        "libero_10": 0,
        "libero_90": 1,
        "libero_goal": 2,
        "libero_object": 3,
        "libero_spatial": 4,
    }
    for root in _candidate_bddl_roots(explicit_root):
        for path in sorted(root.rglob("*.bddl")):
            try:
                language = libero_language_from_bddl_stem(path.stem)
            except ValueError:
                continue
            keys = [normalize_task_text(language), *libero_language_aliases(language)]
            for key in keys:
                index.setdefault(key, []).append(path)

    for paths in index.values():
        paths.sort(key=lambda path: (suite_priority.get(path.parent.name, 99), str(path)))
    return index


def resolve_bddl_path_for_instruction(
    instruction: str,
    *,
    bddl_index: dict[str, list[Path]] | None = None,
    bddl_root: str | os.PathLike[str] | None = None,
) -> Path:
    index = bddl_index if bddl_index is not None else build_bddl_language_index(bddl_root)
    key = normalize_task_text(instruction)
    matches = index.get(key, [])
    if not matches:
        available_sample = ", ".join(sorted(index)[:8])
        raise ValueError(
            f"Could not resolve a LIBERO BDDL file for instruction {instruction!r}. "
            f"Available language sample: {available_sample}"
        )
    return matches[0]


def camera_name_for_image_key(image_key: str) -> str:
    if image_key == "image":
        return "agentview"
    if image_key == "wrist_image":
        return "robot0_eye_in_hand"
    raise ValueError(f"Unsupported image_key for camera projection: {image_key!r}")


def compute_libero_camera_calibration(
    *,
    bddl_path: str | os.PathLike[str],
    camera_name: str,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    try:
        from libero.libero.envs.env_wrapper import ControlEnv
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix,
            get_camera_intrinsic_matrix,
            get_camera_transform_matrix,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Could not import LIBERO / robosuite camera utilities. Run inside the mujoco_playground/openpi environment."
        ) from exc

    env = None
    try:
        env = ControlEnv(
            bddl_file_name=Path(bddl_path).expanduser().resolve(),
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            camera_names=[camera_name],
            camera_heights=int(image_height),
            camera_widths=int(image_width),
        )
        env.reset()
        intrinsic = get_camera_intrinsic_matrix(env.sim, camera_name, int(image_height), int(image_width))
        extrinsic = get_camera_extrinsic_matrix(env.sim, camera_name)
        world_to_camera = get_camera_transform_matrix(env.sim, camera_name, int(image_height), int(image_width))
        cam_id = env.sim.model.camera_name2id(camera_name)
        fovy = float(env.sim.model.cam_fovy[cam_id])
    finally:
        if env is not None:
            env.close()

    return {
        "camera_name": camera_name,
        "bddl_path": str(Path(bddl_path).expanduser().resolve()),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "intrinsic_matrix": intrinsic.tolist(),
        "extrinsic_matrix": extrinsic.tolist(),
        "world_to_camera_transform": world_to_camera.tolist(),
        "fovy": fovy,
        "image_convention": LEROBOT_LIBERO_IMAGE_CONVENTION,
        "image_convention_notes": (
            "LeRobot LIBERO side-view image tensors are horizontally mirrored relative to robosuite's "
            "agentview OpenGL camera projection; the vertical axis is unchanged."
        ),
    }


def semantic_target_policy(skill: str) -> str:
    skill_name = parse_skill_name(skill)
    if skill_name == "PICKUP_FROM":
        return "Choose the object that will be picked up. Put the point on the visible body of that object."
    if skill_name in {"PLACE_ON", "PLACE_IN"}:
        return (
            "Choose the destination where the carried object should be placed. Put the point on the receiving "
            "surface, inside the receiving container, or at the visually best placement location."
        )
    if skill_name in {"OPEN", "CLOSE"}:
        return (
            "Choose the manipulation handle, lip, pull tab, or graspable edge used to open or close the object. "
            "Do not mark the center of the whole drawer or cabinet if a handle is visible."
        )
    if skill_name in {"TURN_ON", "TURN_OFF"}:
        return "Choose the knob, switch, button, or control surface that the gripper should actuate."
    raise ValueError(f"Unsupported skill for semantic target policy: {skill}")


def contact_point_policy(skill: str) -> str:
    skill_name = parse_skill_name(skill)
    if skill_name == "PICKUP_FROM":
        return (
            "Predict the intended gripper-object contact point on the object to be picked up. Prefer a visible "
            "rim, edge, handle, or graspable side where the gripper is likely to touch."
        )
    if skill_name in {"PLACE_ON", "PLACE_IN"}:
        return (
            "The robot should already be holding the object at the start of this skill. Identify the visible point "
            "on the held object that is initially in contact with the gripper, rather than predicting a new contact."
        )
    if skill_name in {"OPEN", "CLOSE"}:
        return (
            "Predict the gripper contact point on the handle, lip, pull tab, or graspable edge of the object that "
            "will be opened or closed."
        )
    if skill_name in {"TURN_ON", "TURN_OFF"}:
        return "Predict the gripper contact point on the knob, switch, button, or control surface."
    raise ValueError(f"Unsupported skill for contact point policy: {skill}")


def extraction_trace_policy(skill: str) -> str:
    skill_name = parse_skill_name(skill)
    if skill_name == "PICKUP_FROM":
        return "The contact point is usually on the rim, edge, handle, or side of the object being picked up."
    if skill_name in {"PLACE_ON", "PLACE_IN"}:
        return "The trace starts at the point on the carried object that is already touching the gripper."
    if skill_name in {"OPEN", "CLOSE"}:
        return "The contact point is likely on the drawer/cabinet handle, lip, tab, or graspable edge."
    if skill_name in {"TURN_ON", "TURN_OFF"}:
        return "The contact point is likely on the knob, switch, button, or control surface."
    raise ValueError(f"Unsupported skill for extraction trace policy: {skill}")


def build_point_schema(name: str) -> dict[str, Any]:
    return {
        "name": name,
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
            "required": ["status", "reasoning", "object_id", "label", "point_2d"],
        },
    }


def build_trace_schema(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string"},
                "reasoning": {"type": "string"},
                "contact_start_step": {"type": "integer"},
                "points": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "frame_index": {"type": "integer"},
                            "point_2d": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 2,
                                "items": {"type": "number"},
                            },
                            "label": {"type": "string"},
                        },
                        "required": ["frame_index", "point_2d", "label"],
                    },
                },
            },
            "required": ["status", "reasoning", "contact_start_step", "points"],
        },
    }


def coordinate_rules_text(coordinate_width: int, coordinate_height: int) -> str:
    return f"""Coordinate rules:
- Return coordinates on a fixed output grid width={coordinate_width}, height={coordinate_height}.
- The full image or video frame spans x=0..{coordinate_width} left to right and y=0..{coordinate_height} top to bottom.
- Return point_2d in row-column order: [y, x], not [x, y].
- Points must be on the visible object or contact location, inside the image bounds.
- Return JSON only, with no markdown, prose, or code fences outside the JSON object.
"""


def build_semantic_target_prompt(
    *,
    instruction: str,
    plan: str,
    skill: str,
    skill_index: int,
    start_step: int,
    image_width: int,
    image_height: int,
    target_object_id: str,
    coordinate_width: int,
    coordinate_height: int,
    video_hint_step: int | None = None,
) -> str:
    if video_hint_step is not None:
        skill_name = parse_skill_name(skill)
        if skill_name == "PICKUP_FROM":
            video_description = (
                "showing the robot perform the entire pickup including the grasp"
            )
            video_focus_clause = (
                " (especially the moment the robot end-effector closes around an object)"
            )
            object_role_phrase = "the pick-up target"
            destination_word = "object"
            extra_note = "Check which object is being picked up in the video to determine the target object. "
        elif skill_name in {"PLACE_ON", "PLACE_IN"}:
            video_description = (
                "showing the robot perform the entire placement including the release of the held object"
            )
            video_focus_clause = (
                " (especially the moment the robot end-effector releases the held object)"
            )
            object_role_phrase = "the placement destination"
            destination_word = "destination"
            extra_note = (
                "Check where in the video the robot releases the held object to determine the placement destination. In some cases where the destination is occluded (e.g.,  back compartment of the caddy), use the video end to determine the destination. For example, for back compartment of the caddy, the compartment is at the center back of the caddy, not toward the right or left side. "
            )
        else:
            raise ValueError(
                f"video_hint_step is set but skill {skill_name!r} is not a supported video-hint skill."
            )
        intro = (
            f"You are annotating a LIBERO {skill_name} skill. You will receive an IMAGE and a VIDEO for this query:\n"
            f"- Image 1: the FIRST frame of the skill segment (step {start_step}). This is the frame you must annotate.\n"
            f"- Video 2: the skill segment video covering steps {start_step}..{video_hint_step} from the same camera viewpoint, "
            f"{video_description}.\n"
            "\n"
            f"You MUST use Video 2{video_focus_clause} to help determine "
            f"which object in Image 1 is {object_role_phrase}. Predict the semantic target point on Image 1 ONLY. "
            "Do NOT report any coordinate from Video 2. Image 1 and Video 2 share the same resolution and camera viewpoint; "
            "only the scene contents differ over time. "
            f"{extra_note}"
        )
        usage_note = (
            "This point is semantic guidance for training and visual prompting, so choose the most indicative visible "
            "point for what the skill is trying to affect. Output the point on Image 1, using Video 2 only to "
            f"identify the correct {destination_word}."
        )
    else:
        intro = "You are annotating a LIBERO robot manipulation skill from only the first frame of the skill segment."
        usage_note = (
            "This point is semantic guidance for training and visual prompting, so choose the most indicative visible "
            "point for what the skill is trying to affect. Use only this first frame."
        )
    return f"""{intro}

Note that any "left", "right", "front", and "back" descriptions should be with respect to the robot's perspective, which is opposite to the image's perspective. 

Episode instruction:
{instruction}

Full skill plan:
{plan}

Current skill:
- skill_index: {skill_index}
- skill: {skill}
- first frame step: {start_step}
- sent image size: width={image_width}, height={image_height}

Your task is to choose one semantically clarifying target point for this skill.
{semantic_target_policy(skill)}

{usage_note}

{coordinate_rules_text(coordinate_width, coordinate_height)}

Output shape:
{{
  "status": "OK",
  "reasoning": "brief visual justification",
  "object_id": "{target_object_id}",
  "label": "{SEMANTIC_TARGET_LABEL}",
  "point_2d": [y, x]
}}
"""


def build_contact_point_prompt(
    *,
    instruction: str,
    plan: str,
    skill: str,
    skill_index: int,
    start_step: int,
    image_width: int,
    image_height: int,
    contact_object_id: str,
    coordinate_width: int,
    coordinate_height: int,
) -> str:
    return f"""You are annotating a LIBERO robot manipulation skill from only the first frame of the skill segment.

Episode instruction:
{instruction}

Full skill plan:
{plan}

Current skill:
- skill_index: {skill_index}
- skill: {skill}
- first frame step: {start_step}
- sent image size: width={image_width}, height={image_height}

Object of interest for gripper contact: "{contact_object_id}"

Your task is to locate the intended contact point between the robot gripper and the object of interest.
{contact_point_policy(skill)}

The point must be on the object of interest, not on the gripper. Use only this first frame.

{coordinate_rules_text(coordinate_width, coordinate_height)}

Output shape:
{{
  "status": "OK",
  "reasoning": "brief visual justification",
  "object_id": "{contact_object_id}",
  "label": "{CONTACT_POINT_LABEL}",
  "point_2d": [y, x]
}}
"""


def build_prediction_trace_prompt(
    *,
    instruction: str,
    plan: str,
    skill: str,
    skill_index: int,
    start_step: int,
    end_step: int,
    sampled_frame_indices: list[int],
    contact_object_id: str,
    contact_point: dict[str, Any],
    image_width: int,
    image_height: int,
    coordinate_width: int,
    coordinate_height: int,
) -> str:
    frame_list = ", ".join(str(idx) for idx in sampled_frame_indices)
    point = contact_point["point"]
    model_point = contact_point["model_point"]
    return f"""You are tracking a predicted gripper-object contact point through a LIBERO skill segment video.

Episode instruction:
{instruction}

Full skill plan:
{plan}

Current skill:
- skill_index: {skill_index}
- skill: {skill}
- half-open interval: [{start_step}, {end_step})
- sampled frame indices shown in the video, in order: [{frame_list}]
- sent video frame size: width={image_width}, height={image_height}

Object of interest: "{contact_object_id}"

The contact point was predicted from the first skill frame:
- object_id: {contact_object_id}
- original image pixel [x, y]: {point}
- model-grid point_2d [y, x]: {model_point}

Track this same physical point on the object through the sampled video after the gripper makes contact.
If the object is already being held at the first frame, start at {start_step}. If contact begins later, set
contact_start_step to the first sampled frame at or after visible contact. For TURN_ON/TURN_OFF or other stationary
controls, the trace may repeat nearly the same point while the gripper actuates it.

Return one point for each sampled frame from contact_start_step through the last sampled frame where contact is
maintained or the contacted object point remains trackable. Estimate through brief occlusion if the same physical
point is clear before and after.

{coordinate_rules_text(coordinate_width, coordinate_height)}

Output shape:
{{
  "status": "OK",
  "reasoning": "brief tracking justification",
  "contact_start_step": <one sampled frame index from the list above>,
  "points": [
    {{"frame_index": <sampled frame index>, "point_2d": [y, x], "label": "{PREDICTION_TRACE_LABEL}"}}
  ]
}}
"""


def build_extraction_trace_prompt(
    *,
    instruction: str,
    plan: str,
    skill: str,
    skill_index: int,
    start_step: int,
    end_step: int,
    sampled_frame_indices: list[int],
    contact_object_id: str,
    image_width: int,
    image_height: int,
    coordinate_width: int,
    coordinate_height: int,
) -> str:
    frame_list = ", ".join(str(idx) for idx in sampled_frame_indices)
    return f"""You are extracting a contact-point trace directly from a LIBERO skill segment video.

Episode instruction:
{instruction}

Full skill plan:
{plan}

Current skill:
- skill_index: {skill_index}
- skill: {skill}
- half-open interval: [{start_step}, {end_step})
- sampled frame indices shown in the video, in order: [{frame_list}]
- sent video frame size: width={image_width}, height={image_height}

Object of interest: "{contact_object_id}"

Directly find the first visible contact point between the robot gripper and the object of interest, then track the
same physical point on the object through the sampled video after contact begins.
{extraction_trace_policy(skill)}

The point must be on the object of interest, not on the gripper. Return one point for each sampled frame from
contact_start_step through the last sampled frame where contact is maintained or the contacted object point remains
trackable. Estimate through brief occlusion if the same physical point is clear before and after.

{coordinate_rules_text(coordinate_width, coordinate_height)}

Output shape:
{{
  "status": "OK",
  "reasoning": "brief extraction and tracking justification",
  "contact_start_step": <one sampled frame index from the list above>,
  "points": [
    {{"frame_index": <sampled frame index>, "point_2d": [y, x], "label": "{EXTRACTION_TRACE_LABEL}"}}
  ]
}}
"""


def build_image_data_url(frame: Any, *, width: int | None = None, height: int | None = None) -> str:
    from PIL import Image

    image = Image.fromarray(as_uint8_hwc(frame))
    if width is not None or height is not None:
        if width is None or height is None:
            raise ValueError("Both width and height must be provided when resizing an image for query.")
        if int(width) <= 0 or int(height) <= 0:
            raise ValueError(f"Resize width and height must be positive, got width={width}, height={height}.")
        resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
        image = image.resize((int(width), int(height)), resample=resampling)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_video_data_url(path: str | os.PathLike[str]) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def _draw_frame_label(frame: Any, *, local_step: int, sample_idx: int, sample_count: int) -> Any:
    from PIL import Image, ImageDraw

    image = Image.fromarray(as_uint8_hwc(frame)).convert("RGB")
    draw = ImageDraw.Draw(image)
    label = f"step {local_step} | sample {sample_idx + 1}/{sample_count}"
    draw.rectangle((0, 0, min(image.width, 230), 22), fill=(0, 0, 0))
    draw.text((5, 5), label, fill=(255, 255, 255))
    return image


def select_evenly_spaced_frame_indices(start_step: int, end_step: int, max_frame_count: int) -> list[int]:
    if end_step <= start_step:
        raise ValueError(f"Invalid segment interval [{start_step}, {end_step}).")
    if max_frame_count <= 0:
        raise ValueError(f"max_frame_count must be positive, got {max_frame_count}.")

    segment_len = end_step - start_step
    if segment_len <= max_frame_count:
        return list(range(start_step, end_step))
    if max_frame_count == 1:
        return [start_step]

    offsets = [round(idx * (segment_len - 1) / (max_frame_count - 1)) for idx in range(max_frame_count)]
    frame_indices = sorted({start_step + int(offset) for offset in offsets})
    if len(frame_indices) < max_frame_count:
        for frame_index in range(start_step, end_step):
            if frame_index not in frame_indices:
                frame_indices.append(frame_index)
                if len(frame_indices) == max_frame_count:
                    break
        frame_indices.sort()

    if frame_indices[0] != start_step or frame_indices[-1] != end_step - 1:
        raise ValueError(
            f"Frame selection failed to include segment endpoints for [{start_step}, {end_step}): {frame_indices}"
        )
    return frame_indices


def render_sampled_segment_video(
    dataset: Any,
    *,
    record: Any,
    episode_bounds: dict[int, tuple[int, int]],
    frame_indices: list[int],
    output_path: str | os.PathLike[str],
    image_key: str = "image",
    width: int | None = None,
    height: int | None = None,
    fps: int = 10,
    overlay_text: bool = True,
) -> Path:
    import imageio.v2 as imageio
    from PIL import Image

    if not frame_indices:
        raise ValueError("Cannot render a segment video with no frames.")
    if (width is None) != (height is None):
        raise ValueError("Use both width and height together, or omit both.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=7, macro_block_size=None) as writer:
        for sample_idx, local_step in enumerate(frame_indices):
            frame = load_episode_frame(
                dataset,
                record=record,
                episode_bounds=episode_bounds,
                local_step=local_step,
                image_key=image_key,
            )
            if width is not None and height is not None:
                image = Image.fromarray(as_uint8_hwc(frame))
                resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
                frame = image.resize((int(width), int(height)), resample=resampling)
            if overlay_text:
                frame = _draw_frame_label(
                    frame,
                    local_step=local_step,
                    sample_idx=sample_idx,
                    sample_count=len(frame_indices),
                )
            writer.append_data(as_uint8_hwc(frame))
    return output


def _coerce_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric, got boolean {value!r}.")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"{field_name} must be numeric, got {value!r}.") from exc
    else:
        raise ValueError(f"{field_name} must be numeric, got {type(value).__name__}.")
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}.")
    return number


def coerce_model_point(raw_point: Any, *, coordinate_width: int, coordinate_height: int) -> list[float]:
    if not isinstance(raw_point, list) or len(raw_point) != 2:
        raise ValueError(f"point_2d must be [y, x], got {raw_point!r}.")
    row = _coerce_number(raw_point[0], field_name="point_2d[0]")
    col = _coerce_number(raw_point[1], field_name="point_2d[1]")
    if not (0 <= row <= coordinate_height and 0 <= col <= coordinate_width):
        raise ValueError(
            f"point_2d {raw_point!r} is outside model coordinate grid "
            f"width={coordinate_width}, height={coordinate_height}."
        )
    return [row, col]


def model_point_to_image_point(
    raw_point: Any,
    *,
    coordinate_width: int,
    coordinate_height: int,
    image_width: int,
    image_height: int,
) -> list[int]:
    row, col = coerce_model_point(
        raw_point,
        coordinate_width=coordinate_width,
        coordinate_height=coordinate_height,
    )
    x = int(round((col / float(coordinate_width)) * image_width))
    y = int(round((row / float(coordinate_height)) * image_height))
    x = max(0, min(image_width - 1, x))
    y = max(0, min(image_height - 1, y))
    return [x, y]


def validate_image_point(point: Any, *, image_width: int, image_height: int, field_name: str = "point") -> list[int]:
    if not isinstance(point, list) or len(point) != 2:
        raise ValueError(f"{field_name} must be [x, y], got {point!r}.")
    x = _coerce_number(point[0], field_name=f"{field_name}[0]")
    y = _coerce_number(point[1], field_name=f"{field_name}[1]")
    if not (0 <= x < image_width and 0 <= y < image_height):
        raise ValueError(
            f"{field_name} {point!r} is outside image bounds width={image_width}, height={image_height}."
        )
    return [int(round(x)), int(round(y))]


def validate_dense_trace_deltas(
    trace: list[list[int]],
    *,
    max_step_delta_pixels: float,
    field_name: str = "trace",
) -> dict[str, float]:
    if max_step_delta_pixels <= 0:
        raise ValueError(f"max_step_delta_pixels must be positive, got {max_step_delta_pixels}.")
    if len(trace) < 2:
        return {
            "max_step_delta_pixels": 0.0,
            "mean_step_delta_pixels": 0.0,
            "p95_step_delta_pixels": 0.0,
        }

    import numpy as np

    points = np.asarray(trace, dtype=float)
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    max_delta = float(np.max(deltas))
    if max_delta > float(max_step_delta_pixels):
        raise ValueError(
            f"{field_name} has adjacent projected points with max delta {max_delta:.2f}px, "
            f"above threshold {max_step_delta_pixels:.2f}px."
        )
    return {
        "max_step_delta_pixels": max_delta,
        "mean_step_delta_pixels": float(np.mean(deltas)),
        "p95_step_delta_pixels": float(np.percentile(deltas, 95)),
    }


def load_episode_states(
    dataset: Any,
    *,
    record: Any,
    episode_bounds: dict[int, tuple[int, int]],
    start_step: int,
    end_step: int,
) -> Any:
    import numpy as np

    if not (0 <= int(start_step) < int(end_step) <= int(record.length)):
        raise ValueError(
            f"Requested state interval [{start_step}, {end_step}) is out of bounds for "
            f"episode {record.episode_index} with length {record.length}."
        )

    episode_start, _ = episode_bounds[record.episode_index]
    states: list[Any] = []
    for local_step in range(int(start_step), int(end_step)):
        item = dataset.hf_dataset[episode_start + local_step]
        state = item["state"]
        if hasattr(state, "detach"):
            state = state.detach().cpu().numpy()
        elif hasattr(state, "numpy"):
            state = state.numpy()
        states.append(np.asarray(state, dtype=float))
    if not states:
        raise ValueError(f"No states loaded for interval [{start_step}, {end_step}).")
    state_array = np.stack(states, axis=0)
    if state_array.ndim != 2 or state_array.shape[1] < 3:
        raise ValueError(f"Expected state array with shape [N, >=3], got {state_array.shape}.")
    return state_array


def transform_camera_xy_to_image_xy(
    xy: Any,
    *,
    image_width: int,
    image_height: int,
    image_convention: str | None,
) -> Any:
    import numpy as np

    points = np.asarray(xy, dtype=float)
    convention = str(image_convention or LEROBOT_LIBERO_IMAGE_CONVENTION)
    if convention in NO_TRANSFORM_IMAGE_CONVENTIONS:
        return points.copy()
    if convention in HORIZONTAL_FLIP_IMAGE_CONVENTIONS:
        return np.column_stack(
            [
                (int(image_width) - 1) - points[:, 0],
                points[:, 1],
            ]
        )
    if convention in ROTATED_180_IMAGE_CONVENTIONS:
        return np.column_stack(
            [
                (int(image_width) - 1) - points[:, 0],
                (int(image_height) - 1) - points[:, 1],
            ]
        )
    raise ValueError(f"Unsupported LIBERO image convention for projection: {convention!r}")


def transform_image_xy_to_camera_xy(
    xy: Any,
    *,
    image_width: int,
    image_height: int,
    image_convention: str | None,
) -> Any:
    # The currently supported transforms are self-inverse.
    return transform_camera_xy_to_image_xy(
        xy,
        image_width=image_width,
        image_height=image_height,
        image_convention=image_convention,
    )


def project_world_points_to_lerobot_image(
    points_xyz: Any,
    *,
    world_to_camera_transform: Any,
    image_width: int,
    image_height: int,
    image_convention: str | None = None,
    rotate_180: bool | None = None,
) -> tuple[Any, Any, Any]:
    import numpy as np

    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape [N, 3], got {points.shape}.")
    transform = np.asarray(world_to_camera_transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"world_to_camera_transform must have shape [4, 4], got {transform.shape}.")

    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=float)], axis=1)
    projected = (transform @ homogeneous.T).T
    depth = projected[:, 2].copy()
    if np.any(~np.isfinite(projected)):
        raise ValueError("Projection produced non-finite homogeneous coordinates.")

    xy = projected[:, :2] / projected[:, 2:3]
    if rotate_180 is not None:
        image_convention = LEGACY_LEROBOT_LIBERO_IMAGE_CONVENTION if rotate_180 else LEROBOT_LIBERO_IMAGE_CONVENTION
    image_xy = transform_camera_xy_to_image_xy(
        xy,
        image_width=image_width,
        image_height=image_height,
        image_convention=image_convention,
    )
    return image_xy, depth, projected


def reproject_lerobot_image_points_to_world(
    points_xy: Any,
    *,
    depth: Any,
    world_to_camera_transform: Any,
    image_width: int,
    image_height: int,
    image_convention: str | None = None,
    rotate_180: bool | None = None,
) -> Any:
    import numpy as np

    image_xy = np.asarray(points_xy, dtype=float)
    if image_xy.ndim != 2 or image_xy.shape[1] != 2:
        raise ValueError(f"points_xy must have shape [N, 2], got {image_xy.shape}.")
    depth_array = np.asarray(depth, dtype=float).reshape(-1)
    if depth_array.shape[0] != image_xy.shape[0]:
        raise ValueError(f"depth must have one value per point, got {depth_array.shape[0]} for {image_xy.shape[0]}.")
    transform = np.asarray(world_to_camera_transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"world_to_camera_transform must have shape [4, 4], got {transform.shape}.")
    if np.any(~np.isfinite(image_xy)) or np.any(~np.isfinite(depth_array)):
        raise ValueError("Cannot reproject non-finite image coordinates or depths.")
    if np.any(depth_array <= 0):
        raise ValueError("Cannot reproject points with non-positive depth.")

    if rotate_180 is not None:
        image_convention = LEGACY_LEROBOT_LIBERO_IMAGE_CONVENTION if rotate_180 else LEROBOT_LIBERO_IMAGE_CONVENTION
    camera_xy = transform_image_xy_to_camera_xy(
        image_xy,
        image_width=image_width,
        image_height=image_height,
        image_convention=image_convention,
    )
    camera_homogeneous = np.column_stack(
        [
            camera_xy[:, 0] * depth_array,
            camera_xy[:, 1] * depth_array,
            depth_array,
            np.ones_like(depth_array),
        ]
    )
    world_homogeneous = (np.linalg.inv(transform) @ camera_homogeneous.T).T
    if np.any(~np.isfinite(world_homogeneous)):
        raise ValueError("Reprojection produced non-finite world coordinates.")
    scale = world_homogeneous[:, 3:4]
    if np.any(np.abs(scale) <= 1e-12):
        raise ValueError("Reprojection produced invalid homogeneous scale.")
    return world_homogeneous[:, :3] / scale


def build_projected_end_effector_trace(
    dataset: Any,
    *,
    record: Any,
    episode_bounds: dict[int, tuple[int, int]],
    start_step: int,
    end_step: int,
    camera_calibration: dict[str, Any],
    image_width: int,
    image_height: int,
    max_step_delta_pixels: float = DEFAULT_EE_TRACE_MAX_STEP_DELTA_PIXELS,
    max_out_of_bounds_fraction: float = DEFAULT_EE_TRACE_MAX_OUT_OF_BOUNDS_FRACTION,
    max_reprojection_error_meters: float = DEFAULT_EE_REPROJECTION_MAX_ERROR_METERS,
    max_rounded_reprojection_error_meters: float = DEFAULT_EE_ROUNDED_REPROJECTION_MAX_ERROR_METERS,
) -> dict[str, Any]:
    import numpy as np

    states = load_episode_states(
        dataset,
        record=record,
        episode_bounds=episode_bounds,
        start_step=start_step,
        end_step=end_step,
    )
    ee_world_positions = states[:, :3]
    raw_xy, depth, _ = project_world_points_to_lerobot_image(
        ee_world_positions,
        world_to_camera_transform=camera_calibration["world_to_camera_transform"],
        image_width=image_width,
        image_height=image_height,
        image_convention=camera_calibration.get("image_convention"),
    )

    finite_xy = np.all(np.isfinite(raw_xy), axis=1)
    in_front = depth > 1e-6
    in_bounds = (
        finite_xy
        & in_front
        & (raw_xy[:, 0] >= 0)
        & (raw_xy[:, 0] < image_width)
        & (raw_xy[:, 1] >= 0)
        & (raw_xy[:, 1] < image_height)
    )
    out_of_bounds_count = int(len(in_bounds) - int(np.count_nonzero(in_bounds)))
    out_of_bounds_fraction = out_of_bounds_count / float(len(in_bounds))
    if out_of_bounds_fraction > float(max_out_of_bounds_fraction):
        raise ValueError(
            f"Projected end-effector trace has {out_of_bounds_count}/{len(in_bounds)} points outside image bounds "
            f"({out_of_bounds_fraction:.1%}), above threshold {max_out_of_bounds_fraction:.1%}."
        )

    clipped_xy = np.column_stack(
        [
            np.clip(raw_xy[:, 0], 0, image_width - 1),
            np.clip(raw_xy[:, 1], 0, image_height - 1),
        ]
    )
    trace = [[int(round(x)), int(round(y))] for x, y in clipped_xy]
    for point_idx, point in enumerate(trace):
        validate_image_point(
            point,
            image_width=image_width,
            image_height=image_height,
            field_name=f"end_effector_trace.trace[{point_idx}]",
        )
    delta_stats = validate_dense_trace_deltas(
        trace,
        max_step_delta_pixels=max_step_delta_pixels,
        field_name="end_effector_trace.trace",
    )

    exact_reprojected = reproject_lerobot_image_points_to_world(
        raw_xy,
        depth=depth,
        world_to_camera_transform=camera_calibration["world_to_camera_transform"],
        image_width=image_width,
        image_height=image_height,
        image_convention=camera_calibration.get("image_convention"),
    )
    exact_errors = np.linalg.norm(exact_reprojected - ee_world_positions, axis=1)
    exact_max_error = float(np.max(exact_errors)) if len(exact_errors) else 0.0
    if exact_max_error > float(max_reprojection_error_meters):
        raise ValueError(
            f"Exact EE project-and-reproject max error {exact_max_error:.3e}m exceeds "
            f"{max_reprojection_error_meters:.3e}m."
        )

    rounded_trace_xy = np.asarray(trace, dtype=float)
    rounded_mask = in_bounds & finite_xy & in_front
    rounded_errors = np.asarray([], dtype=float)
    if np.any(rounded_mask):
        rounded_reprojected = reproject_lerobot_image_points_to_world(
            rounded_trace_xy[rounded_mask],
            depth=depth[rounded_mask],
            world_to_camera_transform=camera_calibration["world_to_camera_transform"],
            image_width=image_width,
            image_height=image_height,
            image_convention=camera_calibration.get("image_convention"),
        )
        rounded_errors = np.linalg.norm(rounded_reprojected - ee_world_positions[rounded_mask], axis=1)
        rounded_max_error = float(np.max(rounded_errors))
        if rounded_max_error > float(max_rounded_reprojection_error_meters):
            raise ValueError(
                f"Rounded in-bounds EE project-and-reproject max error {rounded_max_error:.3e}m exceeds "
                f"{max_rounded_reprojection_error_meters:.3e}m."
            )
    else:
        rounded_max_error = 0.0

    frame_indices = list(range(int(start_step), int(end_step)))
    return {
        "status": "OK",
        "trace_kind": END_EFFECTOR_TRACE_KIND,
        "label": END_EFFECTOR_TRACE_LABEL,
        "camera_name": str(camera_calibration["camera_name"]),
        "state_source": "state[0:3] / robot0_eef_pos",
        "frame_indices": frame_indices,
        "trace": trace,
        "raw_trace": [[round(float(x), 6), round(float(y), 6)] for x, y in raw_xy],
        "projection_depth": [round(float(value), 10) for value in depth],
        "source_world_positions": [
            [round(float(coord), 10) for coord in point]
            for point in ee_world_positions
        ],
        "in_bounds": [bool(value) for value in in_bounds],
        "out_of_bounds_count": out_of_bounds_count,
        "out_of_bounds_fraction": out_of_bounds_fraction,
        "max_out_of_bounds_fraction": float(max_out_of_bounds_fraction),
        "max_allowed_step_delta_pixels": float(max_step_delta_pixels),
        "reprojection_checks": {
            "exact_max_error_meters": exact_max_error,
            "exact_mean_error_meters": float(np.mean(exact_errors)) if len(exact_errors) else 0.0,
            "exact_max_allowed_error_meters": float(max_reprojection_error_meters),
            "rounded_in_bounds_max_error_meters": rounded_max_error,
            "rounded_in_bounds_mean_error_meters": float(np.mean(rounded_errors)) if len(rounded_errors) else 0.0,
            "rounded_in_bounds_count": int(np.count_nonzero(rounded_mask)),
            "rounded_max_allowed_error_meters": float(max_rounded_reprojection_error_meters),
        },
        **delta_stats,
    }


def normalize_point_response(
    raw: Any,
    *,
    expected_object_id: str,
    expected_label: str,
    coordinate_width: int,
    coordinate_height: int,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Point response must be an object, got {type(raw).__name__}.")
    status = str(raw.get("status", "")).strip().upper()
    if status != "OK":
        raise ValueError(f"Point response status must be OK, got {raw.get('status')!r}: {raw!r}")
    object_id = str(raw.get("object_id", "")).strip()
    if object_id != expected_object_id:
        raise ValueError(f"Point response object_id {object_id!r} does not match expected {expected_object_id!r}.")
    label = str(raw.get("label", "")).strip()
    if label != expected_label:
        raise ValueError(f"Point response label {label!r} does not match expected {expected_label!r}.")

    model_point_float = coerce_model_point(
        raw.get("point_2d"),
        coordinate_width=coordinate_width,
        coordinate_height=coordinate_height,
    )
    model_point = [int(round(model_point_float[0])), int(round(model_point_float[1]))]
    image_point = model_point_to_image_point(
        model_point,
        coordinate_width=coordinate_width,
        coordinate_height=coordinate_height,
        image_width=image_width,
        image_height=image_height,
    )
    validate_image_point(image_point, image_width=image_width, image_height=image_height)

    return {
        "status": "OK",
        "reasoning": str(raw.get("reasoning", "")),
        "object_id": object_id,
        "label": label,
        "point": image_point,
        "model_point": model_point,
        "model_coordinate_width": int(coordinate_width),
        "model_coordinate_height": int(coordinate_height),
    }


def normalize_trace_response(
    raw: Any,
    *,
    expected_label: str,
    sampled_frame_indices: list[int],
    start_step: int,
    end_step: int,
    coordinate_width: int,
    coordinate_height: int,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Trace response must be an object, got {type(raw).__name__}.")
    status = str(raw.get("status", "")).strip().upper()
    if status != "OK":
        raise ValueError(f"Trace response status must be OK, got {raw.get('status')!r}: {raw!r}")
    sampled_set = {int(idx) for idx in sampled_frame_indices}
    if not sampled_set:
        raise ValueError("sampled_frame_indices must be non-empty.")

    contact_start_step = int(raw.get("contact_start_step"))
    if contact_start_step not in sampled_set:
        raise ValueError(
            f"contact_start_step {contact_start_step} must be one of sampled frames {sampled_frame_indices}."
        )
    if not (start_step <= contact_start_step < end_step):
        raise ValueError(
            f"contact_start_step {contact_start_step} is outside segment [{start_step}, {end_step})."
        )

    points_raw = raw.get("points")
    if not isinstance(points_raw, list) or not points_raw:
        raise ValueError("Trace response must contain a non-empty points list.")

    frame_indices: list[int] = []
    trace: list[list[int]] = []
    model_trace: list[list[int]] = []
    for point_idx, point_obj in enumerate(points_raw):
        if not isinstance(point_obj, dict):
            raise ValueError(f"Trace point {point_idx} is not an object: {point_obj!r}.")
        label = str(point_obj.get("label", "")).strip()
        if label != expected_label:
            raise ValueError(f"Trace point {point_idx} label {label!r} does not match expected {expected_label!r}.")
        frame_index = int(point_obj.get("frame_index"))
        if frame_index not in sampled_set:
            raise ValueError(f"Trace point {point_idx} frame_index {frame_index} is not in sampled frames.")
        if not (contact_start_step <= frame_index < end_step):
            raise ValueError(
                f"Trace point {point_idx} frame_index {frame_index} is outside "
                f"[contact_start_step={contact_start_step}, end_step={end_step})."
            )
        model_point_float = coerce_model_point(
            point_obj.get("point_2d"),
            coordinate_width=coordinate_width,
            coordinate_height=coordinate_height,
        )
        model_point = [int(round(model_point_float[0])), int(round(model_point_float[1]))]
        image_point = model_point_to_image_point(
            model_point,
            coordinate_width=coordinate_width,
            coordinate_height=coordinate_height,
            image_width=image_width,
            image_height=image_height,
        )
        validate_image_point(image_point, image_width=image_width, image_height=image_height)
        frame_indices.append(frame_index)
        trace.append(image_point)
        model_trace.append(model_point)

    if frame_indices != sorted(frame_indices):
        raise ValueError(f"Trace frame indices must be sorted, got {frame_indices}.")
    if len(set(frame_indices)) != len(frame_indices):
        raise ValueError(f"Trace frame indices contain duplicates: {frame_indices}.")
    if frame_indices[0] != contact_start_step:
        raise ValueError(
            f"First trace frame {frame_indices[0]} must equal contact_start_step {contact_start_step}."
        )

    return {
        "status": "OK",
        "reasoning": str(raw.get("reasoning", "")),
        "contact_start_step": contact_start_step,
        "frame_indices": frame_indices,
        "trace": trace,
        "model_trace": model_trace,
        "sampled_frame_indices": [int(idx) for idx in sampled_frame_indices],
        "model_coordinate_width": int(coordinate_width),
        "model_coordinate_height": int(coordinate_height),
    }


def build_episode_target_trace_annotation(
    *,
    skill_episode: dict[str, Any],
    target_trace_entries: list[dict[str, Any]],
    image_key: str,
    image_width: int,
    image_height: int,
    query_image_width: int | None,
    query_image_height: int | None,
    model_coordinate_width: int,
    model_coordinate_height: int,
    trace_frame_count: int,
    model: str,
    source_repo_id: str,
    dataset_root: Path,
    skill_annotation_source: str | os.PathLike[str],
    semantic_target_enabled: bool,
    contact_prediction_enabled: bool,
    contact_extraction_enabled: bool,
    contact_prediction_only_enabled: bool = False,
    end_effector_trace_enabled: bool,
    ee_projection_camera: dict[str, Any] | None = None,
    prompt_version: str = TARGET_TRACE_PROMPT_VERSION,
) -> dict[str, Any]:
    return {
        "episode_index": int(skill_episode["episode_index"]),
        "task_index": int(skill_episode["task_index"]),
        "instruction": str(skill_episode["instruction"]),
        "num_steps": int(skill_episode["num_steps"]),
        "fps": int(skill_episode.get("fps", 10)),
        "plan": str(skill_episode["plan"]),
        "segments": skill_episode["segments"],
        "target_traces": target_trace_entries,
        "image_key": image_key,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "query_image_width": int(query_image_width) if query_image_width is not None else None,
        "query_image_height": int(query_image_height) if query_image_height is not None else None,
        "model_coordinate_width": int(model_coordinate_width),
        "model_coordinate_height": int(model_coordinate_height),
        "trace_frame_count": int(trace_frame_count),
        "semantic_target_enabled": bool(semantic_target_enabled),
        "contact_prediction_enabled": bool(contact_prediction_enabled),
        "contact_extraction_enabled": bool(contact_extraction_enabled),
        "contact_prediction_only_enabled": bool(contact_prediction_only_enabled),
        "end_effector_trace_enabled": bool(end_effector_trace_enabled),
        "ee_projection_camera": ee_projection_camera,
        "model": model,
        "prompt_version": prompt_version,
        "source_repo_id": source_repo_id,
        "dataset_root": str(dataset_root),
        "skill_annotation_source": str(Path(skill_annotation_source).expanduser().resolve()),
        "created_at": utc_now(),
    }


def _strip_combined_episode(episode: dict[str, Any]) -> dict[str, Any]:
    """Return a slimmed deep-copy of a target-trace episode for the combined JSON.

    Drops fields that are either recoverable from segment metadata or only
    meaningful in shard files:

    - ``end_effector_trace.frame_indices`` — always equals
      ``list(range(start_step, end_step))``; the combined-JSON validator now
      enforces ``len(trace) == end_step - start_step`` instead.
    - ``end_effector_trace.in_bounds`` — replaced by the summary
      ``out_of_bounds_count`` (and ``out_of_bounds_fraction``); the combined-JSON
      validator now enforces that every saved ``trace`` point is inside the
      image rectangle.
    - ``end_effector_trace.projection_depth`` — only used by the shard-time
      reprojection diagnostics; the saved ``reprojection_checks`` summary already
      records the max errors that resulted, so the per-frame depths are dropped.
    - ``sampled_frame_indices`` — only meaningful when at least one
      contact-point trace was generated for the segment, otherwise dropped.

    Shards are not modified; this transformation only runs inside
    ``aggregate_episode_target_traces``.
    """
    import copy

    slim = copy.deepcopy(episode)
    for entry in slim.get("target_traces") or []:
        if not isinstance(entry, dict):
            continue
        has_contact_trace = (
            isinstance(entry.get("prediction_trace"), dict)
            or isinstance(entry.get("extraction_trace"), dict)
        )
        if not has_contact_trace:
            entry.pop("sampled_frame_indices", None)
        ee = entry.get("end_effector_trace")
        if isinstance(ee, dict):
            ee.pop("frame_indices", None)
            ee.pop("in_bounds", None)
            ee.pop("projection_depth", None)
    return slim


def compute_trace_length_stats(
    episodes: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Per-skill-name segment length statistics across the combined episodes.

    Each segment's length is ``end_step - start_step`` (also the dense EE-trace
    length). Segments are grouped by skill name (e.g. ``"PICKUP_FROM"``) and
    summarised with ``count``, ``min``, ``max``, ``mean``, ``std`` (population).
    """
    grouped: dict[str, list[int]] = {}
    for episode in episodes:
        for entry in episode.get("target_traces") or []:
            skill = entry.get("skill")
            if not isinstance(skill, str):
                continue
            try:
                skill_name = parse_skill_name(skill)
            except ValueError:
                continue
            try:
                length = int(entry["end_step"]) - int(entry["start_step"])
            except (KeyError, TypeError, ValueError):
                continue
            if length <= 0:
                continue
            grouped.setdefault(skill_name, []).append(length)

    stats: dict[str, dict[str, float | int]] = {}
    for skill_name in sorted(grouped):
        values = grouped[skill_name]
        n = len(values)
        mean = sum(values) / n
        variance = sum((value - mean) ** 2 for value in values) / n
        std = math.sqrt(variance)
        stats[skill_name] = {
            "count": int(n),
            "min": int(min(values)),
            "max": int(max(values)),
            "mean": round(float(mean), 1),
            "std": round(float(std), 1),
        }
    return stats


def aggregate_episode_target_traces(
    episodes: list[dict[str, Any]],
    *,
    skill_annotations_path: str | os.PathLike[str],
    source_repo_id: str,
    dataset_root: Path | str,
    prompt_version: str = TARGET_TRACE_PROMPT_VERSION,
) -> dict[str, Any]:
    top_level: dict[str, Any] = {
        "schema_version": "libero_skill_target_trace_v1",
        "prompt_version": prompt_version,
        "source_repo_id": source_repo_id,
        "dataset_root": str(dataset_root),
        "fps": 10,
        "skill_annotation_file": str(Path(skill_annotations_path).expanduser().resolve()),
        "vision_language_episode_idx": [],
    }
    if episodes:
        first = episodes[0]
        for key in (
            "image_key",
            "image_width",
            "image_height",
            "query_image_width",
            "query_image_height",
            "model_coordinate_width",
            "model_coordinate_height",
            "trace_frame_count",
            "semantic_target_enabled",
            "contact_prediction_enabled",
            "contact_extraction_enabled",
            "contact_prediction_only_enabled",
            "end_effector_trace_enabled",
            "ee_projection_camera",
        ):
            top_level[key] = first.get(key)

    top_level["trace_length_stats"] = compute_trace_length_stats(episodes)

    for episode in sorted(episodes, key=lambda item: int(item["episode_index"])):
        top_level[str(int(episode["episode_index"]))] = _strip_combined_episode(episode)
    return top_level


def target_trace_scene_episode_dir(root: str | os.PathLike[str], episode_index: int) -> Path:
    return Path(root) / f"episode_{episode_index:06d}"


def target_trace_scene_paths(
    root: str | os.PathLike[str],
    *,
    episode_index: int,
    target_trace_entries: list[dict[str, Any]],
) -> list[Path]:
    episode_dir = target_trace_scene_episode_dir(root, episode_index)
    return [
        episode_dir / f"skill_{int(entry['skill_index']):03d}_start_step_{int(entry['start_step']):06d}.png"
        for entry in target_trace_entries
    ]


def _wrap_text_to_image_width(text: str, *, font: Any, max_width: int, draw: Any) -> list[str]:
    words = str(text).split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _prepend_skill_header(
    image: Any,
    *,
    skill_text: str | None,
    verifier_skill_text: str | None,
) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    if not skill_text and not verifier_skill_text:
        return image

    base = image.convert("RGB")
    width, height = base.size
    size = max(11, height // 20)
    font = None
    for candidate in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            font = ImageFont.truetype(candidate, size=size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    pad = 4
    measure = ImageDraw.Draw(base)
    raw_lines = [
        f"skill: {skill_text}" if skill_text else None,
        f"verifier: {verifier_skill_text}" if verifier_skill_text else None,
    ]
    rendered_lines: list[str] = []
    line_metrics: list[tuple[int, int, int, int]] = []
    for raw in raw_lines:
        if not raw:
            continue
        for piece in _wrap_text_to_image_width(raw, font=font, max_width=width - 2 * pad, draw=measure):
            rendered_lines.append(piece)
            line_metrics.append(measure.textbbox((0, 0), piece, font=font))

    if not rendered_lines:
        return image

    line_height = max(bbox[3] - bbox[1] for bbox in line_metrics)
    line_spacing = max(2, line_height // 4)
    bar_h = 2 * pad + len(rendered_lines) * line_height + max(0, len(rendered_lines) - 1) * line_spacing

    canvas = Image.new("RGB", (width, height + bar_h), (0, 0, 0))
    canvas.paste(base, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    cursor_y = pad
    for piece, bbox in zip(rendered_lines, line_metrics):
        draw.text((pad, cursor_y - bbox[1]), piece, fill=(255, 255, 255), font=font)
        cursor_y += line_height + line_spacing
    return canvas


def _draw_point(draw: Any, point: list[int], *, fill: tuple[int, int, int], radius: int = 5) -> None:
    x, y = int(point[0]), int(point[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=(0, 0, 0), width=2)


def _draw_trace(
    draw: Any,
    trace: list[list[int]],
    *,
    fill: tuple[int, int, int],
    width: int = 3,
    max_markers: int = 24,
) -> None:
    if not trace:
        return
    points = [(int(point[0]), int(point[1])) for point in trace]
    if len(points) >= 2:
        draw.line(points, fill=fill, width=width, joint="curve")
    stride = max(1, math.ceil(len(points) / max_markers))
    for x, y in points[::stride]:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=fill, outline=(0, 0, 0))
    start_x, start_y = points[0]
    draw.rectangle((start_x - 4, start_y - 4, start_x + 4, start_y + 4), fill=fill, outline=(0, 0, 0))


def overlay_target_traces_on_frame(frame: Any, entry: dict[str, Any]) -> Any:
    from PIL import Image, ImageDraw

    image = Image.fromarray(as_uint8_hwc(frame)).convert("RGB")
    draw = ImageDraw.Draw(image)
    legend_lines: list[tuple[str, tuple[int, int, int]]] = []

    semantic_target = entry.get("semantic_target")
    if isinstance(semantic_target, dict):
        color = (0, 220, 255)
        _draw_point(draw, semantic_target["point"], fill=color, radius=6)
        legend_lines.append(("semantic", color))

    prediction_trace = entry.get("prediction_trace")
    if isinstance(prediction_trace, dict):
        color = (255, 190, 0)
        _draw_trace(draw, prediction_trace.get("trace", []), fill=color, width=3)
        legend_lines.append(("pred trace", color))

    contact_prediction = entry.get("contact_prediction")
    if isinstance(contact_prediction, dict):
        contact_color = (255, 130, 0)
        _draw_point(draw, contact_prediction["point"], fill=contact_color, radius=5)
        legend_lines.append(("contact pt", contact_color))

    extraction_trace = entry.get("extraction_trace")
    if isinstance(extraction_trace, dict):
        color = (255, 60, 210)
        _draw_trace(draw, extraction_trace.get("trace", []), fill=color, width=3)
        legend_lines.append(("extract trace", color))

    end_effector_trace = entry.get("end_effector_trace")
    if isinstance(end_effector_trace, dict):
        color = (80, 255, 120)
        _draw_trace(draw, end_effector_trace.get("trace", []), fill=color, width=2)
        legend_lines.append(("ee trace", color))

    if legend_lines:
        x0, y0 = 4, max(4, image.height - (len(legend_lines) * 17 + 7))
        draw.rectangle((0, y0 - 4, 122, image.height), fill=(0, 0, 0))
        for idx, (label, color) in enumerate(legend_lines):
            y = y0 + idx * 17
            draw.rectangle((x0, y + 3, x0 + 10, y + 13), fill=color)
            draw.text((x0 + 15, y), label, fill=(255, 255, 255))

    skill_text = entry.get("skill")
    return _prepend_skill_header(
        image,
        skill_text=str(skill_text) if skill_text else None,
        verifier_skill_text=None,
    )


def save_target_trace_scene_images(
    dataset: Any,
    *,
    record: Any,
    episode_bounds: dict[int, tuple[int, int]],
    target_trace_entries: list[dict[str, Any]],
    output_dir: str | os.PathLike[str],
    image_key: str = "image",
) -> list[Path]:
    paths = target_trace_scene_paths(
        output_dir,
        episode_index=record.episode_index,
        target_trace_entries=target_trace_entries,
    )
    for path, entry in zip(paths, target_trace_entries, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = load_episode_frame(
            dataset,
            record=record,
            episode_bounds=episode_bounds,
            local_step=int(entry["start_step"]),
            image_key=image_key,
        )
        overlay_target_traces_on_frame(frame, entry).save(path)
    return paths


def format_object_json_for_prompt(obj: dict[str, Any]) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def compact_frame_list(frame_indices: list[int], *, max_chars: int = 900) -> str:
    text = ", ".join(str(idx) for idx in frame_indices)
    if len(text) <= max_chars:
        return text
    head = ", ".join(str(idx) for idx in frame_indices[:20])
    tail = ", ".join(str(idx) for idx in frame_indices[-20:])
    return f"{head}, ... , {tail}"


def dedent_prompt(prompt: str) -> str:
    return textwrap.dedent(prompt).strip()
