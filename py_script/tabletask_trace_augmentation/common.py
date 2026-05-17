from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


PROMPT_VERSION = "tabletask_skill_video_v1"
DEFAULT_OPENROUTER_MODEL = "google/gemini-3.1-pro-preview"
DEFAULT_REPO_ID = "n5zhong/table_tasks"
DEFAULT_ANNOTATION_FPS = 30     # match the tabletask fps

SKILL_ARG_COUNTS: dict[str, int] = {
    "PLACE_ON": 2,
    "PLACE_IN": 2,
    "PICKUP_FROM": 2,
}

SKILL_NAMES: tuple[str, ...] = tuple(SKILL_ARG_COUNTS.keys())

FORBIDDEN_FIRST_SKILLS: frozenset[str] = frozenset({"PLACE_ON", "PLACE_IN"})

SKILL_DEFINITIONS = """Allowed skill set and exact syntax:
1) PLACE_ON(object1, object2)
- description: place object1 onto object2
- object1: the object being placed
- object2: object that will support object1
- both object1 and object2 must be a single object with no commas in the description
- Example: PLACE_ON(red bell pepper, blue plate), PLACE_ON(carrot, white plate), PLACE_ON(blue plate, green plate)

2) PLACE_IN(object1, object2)
- description: place object1 into object2
- object1: the object being placed
- object2: object that will contain object1
- both object1 and object2 must be a single object with no commas in the description
- Example: PLACE_IN(black bowl, basket), PLACE_IN(butter, basket)

3) PICKUP_FROM(object1, object2)
- description: pick up object1 from object2
- object1: the object being picked up. This must be a movable object that can be picked up.
- object2: object that supports object1 originally
- both object1 and object2 must be a single object with no commas in the description
- Example: PICKUP_FROM(red bell pepper, table), PICKUP_FROM(eggplant, table)

Notes:
- PICKUP_FROM is for movable objects such as plates, baskets, peppers, vegetables, and similar that can be picked up and placed.
- Generally, prefer PLACE_ON for support surfaces (e.g. plates) and PLACE_IN for containment (e.g. basket), except for tasks that put the plate onto the basket, use PLACE_ON.
"""

SKILL_EXPR_RE = re.compile(r"^([A-Z_]+)\((.*)\)$")
EPISODE_FILE_RE = re.compile(r"^episode_(\d{6})\.json$")
EPISODE_VIDEO_RE = re.compile(r"^episode_(\d{6})\.mp4$")
PLAN_ITEM_RE = re.compile(r"(\d+)\.\s*")


@dataclasses.dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    task_index: int
    instruction: str
    length: int


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def normalize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def resolve_dataset_root(repo_id: str = DEFAULT_REPO_ID, root: str | os.PathLike[str] | None = None) -> Path:
    if root is not None:
        resolved = Path(root).expanduser().resolve()
        if not (resolved / "meta" / "info.json").exists():
            raise FileNotFoundError(f"Dataset root does not look like a LeRobot dataset: {resolved}")
        return resolved

    cache_root = Path("~/.cache/huggingface/hub").expanduser()
    repo_dir = cache_root / f"datasets--{normalize_repo_id(repo_id)}"
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"Could not find cached dataset repo for {repo_id} under {repo_dir}. "
            "Pass --dataset-root explicitly after downloading the dataset."
        )

    ref_main = repo_dir / "refs" / "main"
    if ref_main.exists():
        commit_hash = ref_main.read_text(encoding="utf-8").strip()
        snapshot = repo_dir / "snapshots" / commit_hash
        if snapshot.exists():
            return snapshot.resolve()

    snapshots = sorted((repo_dir / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for snapshot in snapshots:
        if (snapshot / "meta" / "info.json").exists():
            return snapshot.resolve()

    raise FileNotFoundError(f"No valid snapshot found for {repo_id} under {repo_dir}.")


def load_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json_atomic(path: str | os.PathLike[str], data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(target)


def load_dataset_info(dataset_root: str | os.PathLike[str]) -> dict[str, Any]:
    return load_json(Path(dataset_root) / "meta" / "info.json")


def load_task_rows(dataset_root: str | os.PathLike[str]) -> list[dict[str, Any]]:
    return load_jsonl(Path(dataset_root) / "meta" / "tasks.jsonl")


def load_episode_records(dataset_root: str | os.PathLike[str]) -> list[EpisodeRecord]:
    root = Path(dataset_root)
    task_rows = load_task_rows(root)
    instruction_to_task_index: dict[str, int] = {}
    for row in task_rows:
        instruction = str(row["task"]).strip()
        instruction_to_task_index.setdefault(instruction, int(row["task_index"]))

    episodes_rows = load_jsonl(root / "meta" / "episodes.jsonl")
    records: list[EpisodeRecord] = []
    for row in episodes_rows:
        tasks = row.get("tasks", [])
        if not tasks:
            task_value = row.get("task")
            if task_value is None:
                raise ValueError(f"Episode {row.get('episode_index')} has no task text.")
            tasks = [task_value]
        instruction = str(tasks[0]).strip()
        if instruction not in instruction_to_task_index:
            raise ValueError(
                f"Episode {row.get('episode_index')} uses task text not found in tasks.jsonl: {instruction!r}"
            )
        records.append(
            EpisodeRecord(
                episode_index=int(row["episode_index"]),
                task_index=int(row.get("task_index", instruction_to_task_index[instruction])),
                instruction=instruction,
                length=int(row["length"]),
            )
        )
    return records


def episode_video_path(
    dataset_root: str | os.PathLike[str],
    info: dict[str, Any],
    episode_index: int,
    image_key: str = "image",
) -> Path:
    template = info["video_path"]
    chunk_size = int(info["chunks_size"])
    episode_chunk = episode_index // chunk_size
    rel = template.format(episode_chunk=episode_chunk, video_key=image_key, episode_index=episode_index)
    return Path(dataset_root) / rel


def format_plan_string(skills: list[str]) -> str:
    validated = [validate_skill_expr(skill) for skill in skills]
    return " ".join(f"{idx + 1}. {skill}" for idx, skill in enumerate(validated))


def parse_plan_string(plan: str) -> list[str]:
    text = " ".join(plan.strip().split())
    if text.startswith("Plan:"):
        text = text[len("Plan:") :].strip()
    if not text:
        raise ValueError("Plan string is empty.")

    matches = list(PLAN_ITEM_RE.finditer(text))
    if not matches:
        raise ValueError(f"Plan string does not contain numbered items: {plan!r}")

    skills: list[str] = []
    for idx, match in enumerate(matches):
        number = int(match.group(1))
        if number != idx + 1:
            raise ValueError(f"Plan numbering must start at 1 and increase by 1: {plan!r}")
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        skill = text[start:end].strip()
        if not skill:
            raise ValueError(f"Plan item {number} is empty: {plan!r}")
        skills.append(validate_skill_expr(skill))
    return skills


def parse_plan_skills(raw_plan: Any) -> list[str]:
    if isinstance(raw_plan, list):
        if not raw_plan:
            raise ValueError("Plan list is empty.")
        return [validate_skill_expr(str(skill)) for skill in raw_plan]
    if isinstance(raw_plan, str):
        return parse_plan_string(raw_plan)
    raise ValueError(f"Unsupported plan type: {type(raw_plan).__name__}")


def format_instruction_field(instruction: str) -> str:
    return f"Instruction: {instruction}"


def format_instruction_text(instruction: str) -> str:
    return f"{format_instruction_field(instruction)}\n"


def format_plan_text(plan: str | list[str]) -> str:
    if isinstance(plan, list):
        plan_string = format_plan_string(plan)
    else:
        plan_string = format_plan_string(parse_plan_string(plan))
    return f"Plan: {plan_string}\n"


def format_skill_context(
    instruction: str,
    plan: str | list[str],
    current_skill: str | None = None,
    *,
    include_instruction: bool,
) -> str:
    parts: list[str] = []
    if include_instruction:
        parts.append(format_instruction_text(instruction))
    parts.append(format_plan_text(plan))
    if current_skill is not None:
        parts.append(f"Current skill: {current_skill}\n")
    return "".join(parts)


def validate_skill_expr(skill: str) -> str:
    raw = " ".join(skill.strip().split())
    match = SKILL_EXPR_RE.match(raw)
    if not match:
        raise ValueError(f"Invalid skill expression: {skill!r}")
    name, args_blob = match.groups()
    if name not in SKILL_ARG_COUNTS:
        raise ValueError(f"Unsupported skill name: {name}")
    args = [part.strip() for part in args_blob.split(",")]
    expected = SKILL_ARG_COUNTS[name]
    if len(args) != expected:
        raise ValueError(f"{name} expects {expected} argument(s), got {len(args)}: {skill!r}")
    for arg in args:
        if not arg:
            raise ValueError(f"Empty argument in skill expression: {skill!r}")
        if "," in arg:
            raise ValueError(f"Object names must not contain commas: {skill!r}")
    return f"{name}(" + ", ".join(args) + ")"


def skill_list_from_segments(segments: list[dict[str, Any]]) -> list[str]:
    return [str(segment["skill"]) for segment in segments]


def transition_boundary_steps_from_segments(segments: list[dict[str, Any]]) -> list[int]:
    if not isinstance(segments, list) or not segments:
        raise ValueError("Segments must be a non-empty list.")

    boundary_steps: list[int] = []
    prev_end: int | None = None
    for idx, segment in enumerate(segments):
        try:
            start_step = int(segment["start_step"])
            end_step = int(segment["end_step"])
        except Exception as exc:
            raise ValueError(f"Segment {idx} is missing integer start/end steps: {segment!r}") from exc

        if idx == 0 and start_step != 0:
            raise ValueError(f"First segment must start at step 0, got {start_step}.")
        if prev_end is not None and start_step != prev_end:
            raise ValueError(
                f"Segments must be contiguous. Segment {idx} starts at {start_step}, expected {prev_end}."
            )
        if end_step <= start_step:
            raise ValueError(f"Segment {idx} has non-positive length: {segment!r}")

        boundary_steps.append(start_step)
        prev_end = end_step

    return boundary_steps


def build_annotation_schema() -> dict[str, Any]:
    return {
        "name": "tabletask_skill_annotation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "plan": {"type": "string"},
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "skill": {"type": "string"},
                            "end_step": {"type": "integer", "minimum": 1},
                        },
                        "required": ["skill", "end_step"],
                    },
                },
            },
            "required": ["plan", "steps"],
        },
    }


def build_multimodal_prompt(instruction: str, num_steps: int, fps: int) -> str:
    return f"""You are annotating a robot demonstration from a tabletop manipulation dataset.

You are given:
- a single successful demonstration video rendered at {fps} FPS
- frame overlays that show the current step number
- the natural-language task instruction for this episode

Task instruction:
{instruction}

Your task:
1. Decompose the full episode into a sequence of atomic skill executions.
2. Use only the allowed skills below and follow the syntax exactly.
3. Return the ordered plan and the exclusive end_step for each skill segment.

Allowed skills:
{SKILL_DEFINITIONS}

Segmentation rules:
- The first skill segment always starts at step 0.
- The last end_step must equal exactly {num_steps}.
- end_step is EXCLUSIVE, so a segment [start_step, end_step) includes start_step and excludes end_step.
- The returned steps must be contiguous and cover the entire episode with no gaps and no overlaps.
- Use the step overlays in the video to estimate boundaries.
- The plan must match the exact ordered skill sequence in the returned steps.
- Object descriptions must be short, specific, and contain no commas.
- Object descriptions should contain both appearance and positional or prepositional descriptors from the task instruction, if there is any.
    For instance, "put the white mug on the left..." should correspond to an object "left white mug".
- The first skill cannot be PLACE_ON or PLACE_IN, as they are symbolically infeasible.
- Assume that the given task instruction is always feasible, and the mentioned objects are always present in the scene.
- Assume that the given demonstration is successful.
- DO NOT assume that the given demonstration completes the task instruction in the implied order in task instructions:
    - Assume all subtasks mentioned in the instruction are completed in the demonstrations.
    - But DO NOT assume the subtasks are completed in the order implied in the instruction
    - For example, for task "place the eggplant, asparagus, and carrot onto the white plate", you should assume that eggplant, asparagus, and carrot are all placed onto the white plate by the demonstration, but not necessarily in that order. 
    - You should examine the video carefully for the order of subtask completions. DO NOT ASSUME ORDER. 

Output format:
- Return JSON only.
- "plan" must be a single numbered string like:
  "1. PICKUP_FROM(white mug, table) 2. PLACE_ON(white mug, left plate)"
- "steps" must be a list where each item has:
  - "skill": one skill string
  - "end_step": the exclusive ending step for that skill

Example shape:
{{
  "plan": "1. PICKUP_FROM(white mug, table) 2. PLACE_ON(white mug, left plate)",
  "steps": [
    {{"skill": "PICKUP_FROM(white mug, table)", "end_step": 73}},
    {{"skill": "PLACE_ON(white mug, left plate)", "end_step": {num_steps}}}
  ]
}}
"""


def normalize_model_steps(raw: dict[str, Any], num_steps: int) -> dict[str, Any]:
    if "steps" not in raw:
        raise ValueError("Model response is missing 'steps'.")
    if "plan" not in raw:
        raise ValueError("Model response is missing 'plan'.")

    raw_steps = raw["steps"]
    raw_plan = raw["plan"]
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("'steps' must be a non-empty list.")
    if not isinstance(raw_plan, (list, str)):
        raise ValueError("'plan' must be a numbered string or a non-empty list.")

    steps: list[dict[str, Any]] = []
    prev_end = 0
    for idx, item in enumerate(raw_steps):
        if not isinstance(item, dict):
            raise ValueError(f"Step {idx} is not an object.")
        if "skill" not in item or "end_step" not in item:
            raise ValueError(f"Step {idx} must contain 'skill' and 'end_step'.")
        skill = validate_skill_expr(str(item["skill"]))
        try:
            end_step = int(item["end_step"])
        except Exception as exc:
            raise ValueError(f"Step {idx} has a non-integer end_step: {item['end_step']!r}") from exc
        if end_step <= prev_end:
            raise ValueError(
                f"Step {idx} end_step must be greater than the previous end ({prev_end}), got {end_step}."
            )
        steps.append({"start_step": prev_end, "end_step": end_step, "skill": skill})
        prev_end = end_step

    plan_skills = parse_plan_skills(raw_plan)
    step_skills = skill_list_from_segments(steps)
    if plan_skills != step_skills:
        raise ValueError(
            "The returned plan does not exactly match the skill sequence in steps.\n"
            f"plan={plan_skills}\nsteps={step_skills}"
        )

    last_end = steps[-1]["end_step"]
    if last_end != num_steps:
        if abs(last_end - num_steps) <= 2:
            steps[-1]["end_step"] = num_steps
        else:
            raise ValueError(f"Last end_step must equal {num_steps}, got {last_end}.")

    for idx, step in enumerate(steps):
        if step["start_step"] < 0 or step["end_step"] > num_steps:
            raise ValueError(f"Step {idx} is out of bounds: {step}.")
        if step["end_step"] <= step["start_step"]:
            raise ValueError(f"Step {idx} has non-positive length: {step}.")
        if idx > 0 and step["start_step"] != steps[idx - 1]["end_step"]:
            raise ValueError(f"Gap or overlap before step {idx}: {step}.")

    if plan_skills[0].split("(", 1)[0] in FORBIDDEN_FIRST_SKILLS:
        raise ValueError(
            f"The first planned skill cannot be one of {sorted(FORBIDDEN_FIRST_SKILLS)}; got {plan_skills[0]!r}."
        )

    return {"plan": format_plan_string(plan_skills), "plan_skills": plan_skills, "segments": steps}


def build_episode_annotation(
    *,
    record: EpisodeRecord,
    normalized: dict[str, Any],
    model: str,
    source_repo_id: str,
    dataset_root: Path,
    annotation_fps: int,
    prompt_version: str = PROMPT_VERSION,
    raw_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "episode_index": record.episode_index,
        "task_index": record.task_index,
        "instruction": record.instruction,
        "num_steps": record.length,
        "fps": annotation_fps,
        "plan": normalized["plan"],
        "segments": normalized["segments"],
        "model": model,
        "prompt_version": prompt_version,
        "source_repo_id": source_repo_id,
        "dataset_root": str(dataset_root),
        "created_at": utc_now(),
        "raw_model_response": raw_response,
    }


def episode_to_training_record(
    episode: dict[str, Any],
    *,
    boundary_window: int,
) -> dict[str, Any]:
    if boundary_window <= 0:
        raise ValueError(f"boundary_window must be positive, got {boundary_window}")
    instruction = str(episode["instruction"])
    plan_skills = parse_plan_skills(episode["plan"])
    plan_string = format_plan_string(plan_skills)
    canonical_segments = episode["segments"]

    if not canonical_segments:
        raise ValueError(f"Episode {episode['episode_index']} has no canonical segments.")

    training_segments: list[dict[str, Any]] = []
    for idx, segment in enumerate(canonical_segments):
        start_step = int(segment["start_step"])
        end_step = int(segment["end_step"])
        skill = validate_skill_expr(str(segment["skill"]))
        think_end = min(end_step, start_step + boundary_window)

        boundary_segment = {
            "start_step": start_step,
            "end_step": think_end,
            "instruction": format_instruction_field(instruction),
            "plan": plan_string,
            "skill": skill,
            "updated_skill": skill,
            "content": format_skill_context(instruction, plan_string, None, include_instruction=True),
            "updated_content": format_skill_context(instruction, plan_string, skill, include_instruction=False),
            "updated_content_w_instruction": format_skill_context(
                instruction, plan_string, skill, include_instruction=True
            ),
        }
        training_segments.append(boundary_segment)

        if think_end < end_step:
            action_segment = {
                "start_step": think_end,
                "end_step": end_step,
                "instruction": format_instruction_field(instruction),
                "plan": plan_string,
                "skill": skill,
                "content": format_skill_context(instruction, plan_string, skill, include_instruction=True),
                "updated_content": None,
            }
            training_segments.append(action_segment)

        if idx == len(canonical_segments) - 1 and think_end == end_step:
            boundary_segment["updated_content_w_instruction"] = format_skill_context(
                instruction,
                plan_string,
                skill,
                include_instruction=True,
            )

    episode_start_end = int(training_segments[0]["end_step"])
    return {
        "episode_start_interval": [0, episode_start_end],
        "segments": training_segments,
    }


def aggregate_episode_annotations(
    episodes: list[dict[str, Any]],
    *,
    source_repo_id: str,
    dataset_root: Path,
    annotation_fps: int = DEFAULT_ANNOTATION_FPS,
    prompt_version: str = PROMPT_VERSION,
) -> dict[str, Any]:
    top_level: dict[str, Any] = {
        "schema_version": "tabletask_skill_annotation_v1",
        "prompt_version": prompt_version,
        "source_repo_id": source_repo_id,
        "dataset_root": str(dataset_root),
        "fps": annotation_fps,
        "vision_language_episode_idx": [],
    }
    for episode in sorted(episodes, key=lambda item: int(item["episode_index"])):
        top_level[str(int(episode["episode_index"]))] = episode
    return top_level


def aggregate_training_annotations(
    episodes: list[dict[str, Any]],
    *,
    boundary_window: int,
    source_repo_id: str,
    dataset_root: Path,
    annotation_fps: int = DEFAULT_ANNOTATION_FPS,
    prompt_version: str = PROMPT_VERSION,
) -> dict[str, Any]:
    top_level: dict[str, Any] = {
        "schema_version": "tabletask_skill_training_v1",
        "prompt_version": prompt_version,
        "source_repo_id": source_repo_id,
        "dataset_root": str(dataset_root),
        "fps": annotation_fps,
        "vision_language_episode_idx": [],
        "boundary_window": boundary_window,
    }
    for episode in sorted(episodes, key=lambda item: int(item["episode_index"])):
        top_level[str(int(episode["episode_index"]))] = episode_to_training_record(
            episode,
            boundary_window=boundary_window,
        )
    return top_level


def list_episode_shards(shard_dir: str | os.PathLike[str]) -> list[Path]:
    root = Path(shard_dir)
    if not root.exists():
        return []
    shards: list[Path] = []
    for child in root.iterdir():
        if child.is_file() and EPISODE_FILE_RE.match(child.name):
            shards.append(child)
    return sorted(shards)


def parse_episode_index_from_filename(path: Path) -> int:
    match = EPISODE_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Not an episode shard filename: {path}")
    return int(match.group(1))


def episode_shard_path(shard_dir: str | os.PathLike[str], episode_index: int) -> Path:
    return Path(shard_dir) / f"episode_{episode_index:06d}.json"


def build_video_data_url(path: str | os.PathLike[str]) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def as_uint8_hwc(image: Any) -> Any:
    import numpy as np

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    elif hasattr(image, "numpy") and not isinstance(image, np.ndarray):
        image = image.numpy()

    if hasattr(image, "convert"):
        image = np.array(image)

    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D image array, got shape {array.shape}")

    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))

    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def overlay_step_text_for_annotation(frame: Any, *, step_idx: int, total_steps: int, instruction: str) -> Any:
    """Annotation-time overlay: top bar = step number, bottom bar = instruction.

    This mirrors the libero pipeline's overlay style so the model sees a familiar layout.
    """
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)

    caption = f"step {step_idx:03d} / {total_steps - 1:03d}"
    subtitle = instruction

    draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0))
    draw.rectangle((0, image.height - 26, image.width, image.height), fill=(0, 0, 0))
    draw.text((8, 8), caption, fill=(255, 255, 255))
    draw.text((8, image.height - 21), subtitle, fill=(255, 255, 255))
    return image


def overlay_step_text_for_visualization(
    frame: Any,
    *,
    step_idx: int,
    total_steps: int,
    skill: str,
    episode_index: int,
    total_episodes: int,
    task_index: int,
    instruction: str,
) -> Any:
    """Visualization-time overlay.

    Top line is identical in position and content to ``overlay_step_text_for_annotation`` so the
    frame index appears where the annotator saw it. A second top line carries the skill name.
    The bottom bar carries (line 1) episode + task indices and (line 2) the task instruction.
    """
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)

    step_caption = f"step {step_idx:03d} / {total_steps - 1:03d}"
    skill_caption = f"skill: {skill}"
    episode_caption = (
        f"episode {episode_index:06d} / {total_episodes - 1:06d}    task {task_index:02d}"
    )
    instruction_caption = f"task: {instruction}"

    top_bar_height = 60
    bottom_bar_height = 52
    draw.rectangle((0, 0, image.width, top_bar_height), fill=(0, 0, 0))
    draw.rectangle((0, image.height - bottom_bar_height, image.width, image.height), fill=(0, 0, 0))
    draw.text((8, 8), step_caption, fill=(255, 255, 255))
    draw.text((8, 34), skill_caption, fill=(255, 230, 120))
    draw.text((8, image.height - 47), episode_caption, fill=(255, 255, 255))
    draw.text((8, image.height - 21), instruction_caption, fill=(255, 255, 255))
    return image


def iter_episode_frames(video_path: str | os.PathLike[str]):
    """Iterate decoded RGB uint8 frames from an mp4 video, sequentially."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(video_path))
    try:
        for frame in reader:
            yield as_uint8_hwc(frame)
    finally:
        reader.close()


def get_episode_frames_at_indices(
    video_path: str | os.PathLike[str],
    frame_indices: Iterable[int],
) -> list[Any]:
    targets = sorted({int(idx) for idx in frame_indices})
    if not targets:
        return []
    needed = set(targets)
    collected: dict[int, Any] = {}
    for idx, frame in enumerate(iter_episode_frames(video_path)):
        if idx in needed:
            collected[idx] = frame
            needed.remove(idx)
            if not needed:
                break
    missing = [t for t in targets if t not in collected]
    if missing:
        raise ValueError(f"Could not read frames {missing} from {video_path}")
    return [collected[idx] for idx in frame_indices]


def render_annotation_video(
    *,
    record: EpisodeRecord,
    source_video_path: Path,
    output_path: str | os.PathLike[str],
    fps: int = DEFAULT_ANNOTATION_FPS,
    overlay_text: bool = True,
) -> Path:
    """Render a step-overlaid mp4 of the episode to send to the annotator model.

    The source frames are read from the dataset's pre-encoded mp4 file. Each source frame becomes
    one output frame; output FPS controls playback speed only.
    """
    import imageio.v2 as imageio

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    with imageio.get_writer(str(output), fps=fps, codec="libx264", quality=7, macro_block_size=1) as writer:
        for local_step, frame in enumerate(iter_episode_frames(source_video_path)):
            if overlay_text:
                composed = overlay_step_text_for_annotation(
                    frame,
                    step_idx=local_step,
                    total_steps=record.length,
                    instruction=record.instruction,
                )
            else:
                composed = frame
            writer.append_data(as_uint8_hwc(composed))
            frame_count += 1

    if frame_count != record.length:
        raise ValueError(
            f"Source video {source_video_path} produced {frame_count} frames but episode "
            f"{record.episode_index} reports length {record.length}."
        )
    return output


def segment_skill_at_step(segments: list[dict[str, Any]], step_idx: int) -> str:
    for segment in segments:
        start = int(segment["start_step"])
        end = int(segment["end_step"])
        if start <= step_idx < end:
            return str(segment["skill"])
    if segments and step_idx == int(segments[-1]["end_step"]):
        return str(segments[-1]["skill"])
    raise ValueError(f"Step {step_idx} is not covered by any segment: {segments!r}")


def render_visualization_video(
    *,
    record: EpisodeRecord,
    segments: list[dict[str, Any]],
    source_video_path: Path,
    output_path: str | os.PathLike[str],
    total_episodes: int,
    fps: int = DEFAULT_ANNOTATION_FPS,
) -> Path:
    """Render an annotated visualization mp4: every frame is overlaid with the inferred skill
    label for its step, plus the step index in the same position/size as the annotator saw, plus
    the episode index in the bottom bar.
    """
    import imageio.v2 as imageio

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    with imageio.get_writer(str(output), fps=fps, codec="libx264", quality=7, macro_block_size=1) as writer:
        for local_step, frame in enumerate(iter_episode_frames(source_video_path)):
            skill = segment_skill_at_step(segments, local_step)
            composed = overlay_step_text_for_visualization(
                frame,
                step_idx=local_step,
                total_steps=record.length,
                skill=skill,
                episode_index=record.episode_index,
                total_episodes=total_episodes,
                task_index=record.task_index,
                instruction=record.instruction,
            )
            writer.append_data(as_uint8_hwc(composed))
            frame_count += 1

    if frame_count != record.length:
        raise ValueError(
            f"Source video {source_video_path} produced {frame_count} frames but episode "
            f"{record.episode_index} reports length {record.length}."
        )
    return output


def transition_scene_episode_dir(root: str | os.PathLike[str], episode_index: int) -> Path:
    return Path(root) / f"episode_{episode_index:06d}"


def transition_scene_paths(
    root: str | os.PathLike[str],
    *,
    episode_index: int,
    segments: list[dict[str, Any]],
) -> list[Path]:
    episode_dir = transition_scene_episode_dir(root, episode_index)
    return [
        episode_dir / f"transition_{boundary_idx:03d}_step_{step:06d}.png"
        for boundary_idx, step in enumerate(transition_boundary_steps_from_segments(segments))
    ]


def save_transition_scene_images(
    *,
    record: EpisodeRecord,
    source_video_path: Path,
    segments: list[dict[str, Any]],
    output_dir: str | os.PathLike[str],
) -> list[Path]:
    from PIL import Image

    boundary_steps = transition_boundary_steps_from_segments(segments)
    paths = transition_scene_paths(
        output_dir,
        episode_index=record.episode_index,
        segments=segments,
    )
    frames = get_episode_frames_at_indices(source_video_path, boundary_steps)
    for path, frame in zip(paths, frames):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(path)
    return paths


def list_visualization_videos(videos_dir: str | os.PathLike[str]) -> list[Path]:
    root = Path(videos_dir)
    if not root.exists():
        return []
    out: list[Path] = []
    for child in root.iterdir():
        if child.is_file() and EPISODE_VIDEO_RE.match(child.name):
            out.append(child)
    return sorted(out)


def visualization_video_path(videos_dir: str | os.PathLike[str], episode_index: int) -> Path:
    return Path(videos_dir) / f"episode_{episode_index:06d}.mp4"


def rebuild_combined_video(
    videos_dir: str | os.PathLike[str],
    *,
    combined_filename: str = "combined.mp4",
) -> Path | None:
    """Concatenate all visualization videos in ``videos_dir`` (sorted by episode index) into
    a single combined mp4. Uses ffmpeg's concat demuxer (no re-encoding) when all inputs share
    the same codec/format. The combined video is written atomically.

    Returns the combined video path, or None if there are no individual videos.
    """
    videos_root = Path(videos_dir)
    individual = list_visualization_videos(videos_root)
    combined_path = videos_root / combined_filename

    if not individual:
        if combined_path.exists():
            combined_path.unlink()
        return None

    videos_root.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, dir=str(videos_root)) as f:
        for video_path in individual:
            abs_path = video_path.resolve()
            f.write(f"file '{abs_path}'\n")
        list_path = Path(f.name)

    tmp_output = combined_path.with_name(combined_path.stem + ".tmp" + combined_path.suffix)
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(tmp_output),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            shutil.copyfile(list_path, videos_root / "combined.failed.txt")
            raise RuntimeError(
                f"ffmpeg concat failed (rc={result.returncode}). stderr:\n{result.stderr}"
            )
    finally:
        try:
            list_path.unlink()
        except FileNotFoundError:
            pass

    tmp_output.replace(combined_path)
    return combined_path
