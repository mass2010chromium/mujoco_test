from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import io
import json
import os
import re
from pathlib import Path
from typing import Any


PROMPT_VERSION = "calvin_skill_video_v1"
DEFAULT_OPENROUTER_MODEL = "google/gemini-3.1-pro-preview"
DEFAULT_REPO_ID = "fywang/calvin-task-ABC-D-lerobot"
DEFAULT_FPS = 10

IMAGE_KEY_ALIASES: dict[str, str] = {
    "top": "observation.images.top",
    "wrist": "observation.images.wrist",
    "observation.images.top": "observation.images.top",
    "observation.images.wrist": "observation.images.wrist",
}

SKILL_ARG_COUNTS: dict[str, int] = {
    "PLACE_ON": 1,
    "PLACE_IN": 1,
    "PICKUP_FROM": 2,
    "OPEN": 1,
    "CLOSE": 1,
    "TURN_ON": 1,
    "TURN_OFF": 1,
    "MOVE_SLIDER": 1,
    "PUSH": 3,
    "PUSH_INTO": 2,
    "TURN_OBJECT": 2,
}

DIRECTION_ONLY_SKILLS = {"MOVE_SLIDER"}
DIRECTION_ARG_INDICES = {
    "PUSH": {1},
    "TURN_OBJECT": {1},
}
DIRECTION_VALUES = {"left", "right"}
PUSH_MODE_VALUES = {"slide", "sweep_off"}

OBJECT_NAME_ALIASES: dict[str, str] = {
    "table surface": "table",
    "cabinet drawer": "drawer",
    "green light": "led light",
    "led": "led light",
    "green lamp": "led light",
    "led lamp": "led light",
    "yellow lamp": "light bulb",
    "yellow light": "light bulb",
}

SKILL_DEFINITIONS = """Allowed skill set and exact syntax:
1) PLACE_ON(target_location)
- description: place the object being grasped onto the target_location
- target_location: object that will support object being grasped
- target_location must be a single object with no commas in the description
- Example: PLACE_ON(table) for placing the grasped object onto the table surface
- Example: PLACE_ON(block) for placing the grasped object onto another block, i.e. stacking

2) PLACE_IN(target_container)
- description: place the object being grasped into target_container
- target_container: the container object, such as a slider cabinet or a drawer, that will contain object being grasped
- target_container must be a single object with no commas in the description
- Example: PLACE_IN(drawer) for placing the grasped object into the drawer
- Example: PLACE_IN(sliding cabinet) for placing the grasped object into a sliding cabinet

3) PICKUP_FROM(object1, object2)
- description: pick up or lift up object1 from object2
- object1: the object being picked up. This must be a movable object that can be picked up.
- object2: object that supports object1 originally
- both object1 and object2 must be a single object with no commas in the description
- Example: PICKUP_FROM(red block, drawer) for picking up or lifting up a red block from the drawer
- Example: PICKUP_FROM(pink block, table) for picking up or lifting up a pink block from the table

4) OPEN(object)
- description: this is particularly for opening a drawer
- object: the articulated object being opened
- should be a single drawer object with no commas in the description
- Example: OPEN(drawer) opens the drawer

5) CLOSE(object)
- description: this is particularly for closing a drawer
- object: the articulated object being closed
- should be a single drawer object with no commas in the description
- Example: CLOSE(drawer) closes the drawer

6) TURN_ON(object)
- object: the object being turned on, such as a light
- should be a single object with no commas in the description
- Example: TURN_ON(led light) turns on the LED light
- Example: TURN_ON(light bulb) turns on the light bulb

7) TURN_OFF(object)
- object: the object being turned off, such as a light
- should be a single object with no commas in the description
- Example: TURN_OFF(led light) turns off the LED light
- Example: TURN_OFF(light bulb) turns off the light bulb

8) MOVE_SLIDER(direction)
- description: move the slider door toward a direction
- direction: should be either "left" or "right"
- Example: MOVE_SLIDER(left) moves the sliding door to the left
- Example: MOVE_SLIDER(right) slides the door to the right side

9) PUSH(object, direction, mode)
- description: push or sweep a non-attached, liftable object in direction without grasping it.
- object: the object being pushed or swept, such as a block.
- direction: should be either "left" or "right".
- mode: should be either "slide" or "sweep_off".
  - "slide" means the object should move in the given direction while remaining on its original support surface.
  - "sweep_off" means the object should be pushed in the given direction so that it is removed from its original support, such as sweeping the top block off another block during unstacking.
- Use PUSH(object, direction, slide) for ordinary push-left / push-right tasks where the object stays on the table or same support.
- Use PUSH(object, direction, sweep_off) for unstacking by pushing, knocking, or sweeping the object off the object that originally supports it.
- Do not use PUSH for the sliding door, drawer, button, or switch.
- Example: PUSH(red block, left, slide) pushes the red block left while it remains on the table.
- Example: PUSH(blue block, right, slide) slides the blue block right on the same support.
- Example: PUSH(red block, left, sweep_off) unstack the red block by sweeping the red block left off the block it was originally on.
- Example: PUSH(pink block, right, sweep_off) remove the pink block from the stack by knocking the pink block right off its supporting block.

10) PUSH_INTO(object, target_container)
- description: push or sweep the object into the target_container
- object: the object being pushed or swept
- target_container: the container object, such as a drawer, that will contain object being pushed or swept
- Example: PUSH_INTO(object, drawer) pushes or sweeps the object into the drawer
- Example: PUSH_INTO(block, drawer) pushes or sweeps the block into the drawer

11) TURN_OBJECT(object, direction)
- description: grasp an object, turn or rotate the object toward a direction and then place it onto the table in that way.
- direction: should be either "left" or "right"
- Example: TURN_OBJECT(blue block, left) rotates the blue block to the left
- Example: TURN_OBJECT(red block, right) grasps the red block and turns it right

Important Notes:
- PICKUP_FROM is only for liftable and non-attached objects such as blocks that can be picked up and placed.
- Use OPEN/CLOSE for drawers.
- Use MOVE_SLIDER for the sliding door.
- Differentiate PUSH(object, direction, mode) against OPEN, CLOSE, and MOVE_SLIDER. PUSH(object, direction, mode) is for pushing a block or another non-attached object, whereas MOVE_SLIDER is for the sliding door, and OPEN/CLOSE is for the drawer.
- Use PUSH(object, direction, slide) when the object should remain on its original support after being pushed.
- Use PUSH(object, direction, sweep_off) when the object should be removed from its original support, such as sweeping the top block off another block during unstacking.
- For unstack-related tasks, based on visual demonstration, use PICKUP_FROM(object, support) if the robot grasps and lifts the object from its support; use PUSH(object, direction, sweep_off) if the robot does not grasp the object and instead sweeps/knocks it off its original support.
- Prefer PLACE_ON for support surfaces and PLACE_IN for containment.
- Only use OPEN if the object is currently closed.
- Only use CLOSE if the object is currently open.
- PLACE_ON and PLACE_IN can only be used when the robot is already grasping the object, either because the episode starts with a grasped object or because a previous PICKUP_FROM has happened.
- Do not translate instructions such as "push down button" or "push button" to PUSH. The button controls the led light, so always treat pushing the button as either TURN_ON(led light) or TURN_OFF(led light).
- Do not translate instructions such as "move down the switch" or "push the switch downwards" to PUSH or MOVE_SLIDER. The switch controls the light bulb, so always treat manipulating the switch as either TURN_ON(light bulb) or TURN_OFF(light bulb).
- Typically, manipulating the switch downwards means TURN_OFF(light bulb), while manipulating the switch upwards means TURN_ON(light bulb).
- TURN_OBJECT(object, direction) includes grasping the object (unless the object is grasped at episode start), turning the grasped object and potentially placing it down onto the table. So do not have a PLACE_ON skill after TURN_OBJECT, and do not have a PICKUP_FROM before TURN_OBJECT.

Given the variance in prompt language, for wording consistency, maintain the following object naming conventions:
- use the term "table" consistently to represent table or table surface
- use the term "drawer" consistently to represent "drawer" or "cabinet drawer"
- use the term "led light" consistently to represent the following objects: "led light", "green light", "led", "green lamp", "led lamp"
- use the term "light bulb" consistently to represent the following objects: "light bulb", "yellow lamp", "yellow light"
"""

SKILL_EXPR_RE = re.compile(r"^([A-Z_]+)\((.*)\)$")
EPISODE_FILE_RE = re.compile(r"^episode_(\d{6})\.json$")
PLAN_ITEM_RE = re.compile(r"(\d+)\.\s*")
LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


@dataclasses.dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    task_index: int
    task_name: str | None
    raw_instruction: str
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


def split_task_text(task_text: str) -> tuple[str | None, str]:
    normalized = " ".join(str(task_text).strip().split())
    if ":" not in normalized:
        return None, normalized
    task_name, instruction = normalized.split(":", 1)
    return task_name.strip() or None, instruction.strip()


def load_episode_records(dataset_root: str | os.PathLike[str]) -> list[EpisodeRecord]:
    root = Path(dataset_root)
    task_rows = load_task_rows(root)
    instruction_to_task_index: dict[str, int] = {}
    for row in task_rows:
        instruction = str(row["task"]).strip()
        instruction_to_task_index.setdefault(instruction, int(row["task_index"]))

    episode_rows = load_jsonl(root / "meta" / "episodes.jsonl")
    records: list[EpisodeRecord] = []
    for row in episode_rows:
        tasks = row.get("tasks", [])
        if not tasks:
            task_value = row.get("task")
            if task_value is None:
                raise ValueError(f"Episode {row.get('episode_index')} has no task text.")
            tasks = [task_value]
        raw_instruction = str(tasks[0]).strip()
        if raw_instruction not in instruction_to_task_index:
            raise ValueError(
                f"Episode {row.get('episode_index')} uses task text not found in tasks.jsonl: {raw_instruction!r}"
            )
        task_name, instruction = split_task_text(raw_instruction)
        records.append(
            EpisodeRecord(
                episode_index=int(row["episode_index"]),
                task_index=int(row.get("task_index", instruction_to_task_index[raw_instruction])),
                task_name=task_name,
                raw_instruction=raw_instruction,
                instruction=instruction,
                length=int(row["length"]),
            )
        )
    return records


def load_task_rows(dataset_root: str | os.PathLike[str]) -> list[dict[str, Any]]:
    return load_jsonl(Path(dataset_root) / "meta" / "tasks.jsonl")


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


def format_instruction_text(instruction: str) -> str:
    return f"{format_instruction_field(instruction)}\n"


def format_instruction_field(instruction: str) -> str:
    return f"Instruction: {instruction}"


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


def normalize_object_name(arg: str) -> str:
    normalized = " ".join(arg.strip().split())
    if not normalized:
        return normalized
    normalized = LEADING_ARTICLE_RE.sub("", normalized).strip().lower()
    return OBJECT_NAME_ALIASES.get(normalized, normalized)


def normalize_direction_arg(arg: str) -> str:
    normalized = " ".join(arg.strip().split()).lower()
    if normalized not in DIRECTION_VALUES:
        raise ValueError(f"Direction arguments must be one of {sorted(DIRECTION_VALUES)}, got {arg!r}")
    return normalized


def normalize_push_mode_arg(arg: str) -> str:
    normalized = " ".join(arg.strip().split()).lower()
    if normalized not in PUSH_MODE_VALUES:
        raise ValueError(f"PUSH mode must be one of {sorted(PUSH_MODE_VALUES)}, got {arg!r}")
    return normalized


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

    normalized_args: list[str] = []
    for idx, arg in enumerate(args):
        if not arg:
            raise ValueError(f"Empty argument in skill expression: {skill!r}")
        if "," in arg:
            raise ValueError(f"Object names must not contain commas: {skill!r}")
        if name in DIRECTION_ONLY_SKILLS:
            normalized_args.append(normalize_direction_arg(arg))
            continue
        if idx in DIRECTION_ARG_INDICES.get(name, set()):
            normalized_args.append(normalize_direction_arg(arg))
            continue
        if name == "PUSH" and idx == 2:
            normalized_args.append(normalize_push_mode_arg(arg))
            continue

        normalized_obj = normalize_object_name(arg)
        if not normalized_obj:
            raise ValueError(f"Empty object argument in skill expression: {skill!r}")
        normalized_args.append(normalized_obj)

    return f"{name}(" + ", ".join(normalized_args) + ")"


def parse_skill_expr(skill: str) -> tuple[str, list[str]]:
    normalized = validate_skill_expr(skill)
    match = SKILL_EXPR_RE.match(normalized)
    if match is None:  # pragma: no cover - defensive
        raise ValueError(f"Invalid normalized skill expression: {normalized!r}")
    name, args_blob = match.groups()
    args = [part.strip() for part in args_blob.split(",")] if args_blob else []
    return name, args


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
        "name": "calvin_skill_annotation",
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


def build_multimodal_prompt(record: EpisodeRecord, fps: int = DEFAULT_FPS) -> str:
    task_name_block = f"CALVIN task name: {record.task_name}\n" if record.task_name else ""
    return f"""You are annotating a robot demonstration from the CALVIN benchmark.

You are given:
- a single successful demonstration video rendered at {fps} FPS
- frame overlays that show the current step number
- the CALVIN task name and natural-language instruction for this episode

{task_name_block}Natural-language instruction:
{record.instruction}

Original task text:
{record.raw_instruction}

Your task:
1. Decompose the full episode into a sequence of atomic skill executions.
2. Use only the allowed skills below and follow the syntax exactly.
3. Return the ordered plan and the exclusive end_step for each skill segment.

Allowed skills:
{SKILL_DEFINITIONS}

Segmentation rules:
- The first skill segment always starts at step 0.
- The last end_step must equal exactly {record.length}.
- end_step is EXCLUSIVE, so a segment [start_step, end_step) includes start_step and excludes end_step.
- The returned steps must be contiguous and cover the entire episode with no gaps and no overlaps.
- Use the step overlays in the video to estimate boundaries.
- The plan must match the exact ordered skill sequence in the returned steps.
- The demonstration may start with the robot already holding an object. In that case the first skill may legitimately be PLACE_ON(...), PLACE_IN(...), or TURN_OBJECT(object, direction).
- If a skill is PLACE_ON or PLACE_IN, the object being manipulated is implicit from the object currently grasped by the robot and should not be repeated in the skill arguments.
- TURN_OBJECT(object, direction) is atomic. It already includes any needed grasping and any needed placement after the rotation, so do not put PICKUP_FROM immediately before it or PLACE_ON / PLACE_IN immediately after it.
- Annotate every executed skill needed to complete the episode, including prerequisite steps such as OPEN(drawer) before PLACE_IN(drawer), not just the nominal target primitive in the task name.
- Object descriptions must be short, specific, and contain no commas.
- When blocks are involved, include the block color whenever needed for disambiguation.
- Use the naming conventions from the allowed-skill definition block for table, drawer, led light, and light bulb.
- Assume that the given task instruction is always feasible, the demonstration is successful, and the objects mentioned in the instruction are present in the scene.
- Use the gripper-state overlay in the top-right corner to help distinguish grasping, holding, releasing, and non-grasped pushing or sweeping. "gripper: open" means the gripper is open; "gripper: closed" means the gripper is closed.

Special notes regarding MOVE_SLIDER:
- In a MOVE_SLIDER skill, the gripper must be closed during holding and sliding portion. MOVE_SLIDER must have a period where the gripper is closed. Use the gripper overlay to verify this.
- Pay extra attention to motions that look like MOVE_SLIDER. If the gripper is never closed during the motion, the motion should not be classified as MOVE_SLIDER. Use the gripper overlay to verify this.

Special notes regarding PICKUP_FROM:
- By the end of a PICKUP_FROM skill, the gripper must be closed. Use the gripper overlay to verify this.

Output format:
- Return JSON only.
- "plan" must be a single numbered string like:
  "1. OPEN(drawer) 2. PICKUP_FROM(red block, drawer) 3. PLACE_ON(table)"
- "steps" must be a list where each item has:
  - "skill": one skill string
  - "end_step": the exclusive ending step for that skill

Example shape:
{{
  "plan": "1. OPEN(drawer) 2. PICKUP_FROM(red block, drawer) 3. PLACE_ON(table)",
  "steps": [
    {{"skill": "OPEN(drawer)", "end_step": 14}},
    {{"skill": "PICKUP_FROM(red block, drawer)", "end_step": 32}},
    {{"skill": "PLACE_ON(table)", "end_step": {record.length}}}
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

    return {"plan": format_plan_string(plan_skills), "plan_skills": plan_skills, "segments": steps}


def resolve_image_key(image_key: str) -> str:
    if image_key not in IMAGE_KEY_ALIASES:
        choices = ", ".join(sorted(IMAGE_KEY_ALIASES))
        raise ValueError(f"Unsupported image key {image_key!r}. Expected one of: {choices}")
    return IMAGE_KEY_ALIASES[image_key]


def load_dataset_info(dataset_root: str | os.PathLike[str]) -> dict[str, Any]:
    return load_json(Path(dataset_root) / "meta" / "info.json")


def episode_parquet_path(
    dataset_root: str | os.PathLike[str],
    episode_index: int,
    dataset_info: dict[str, Any] | None = None,
) -> Path:
    root = Path(dataset_root)
    info = dataset_info if dataset_info is not None else load_dataset_info(root)
    chunk_size = int(info.get("chunks_size", 1000))
    template = str(info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"))
    relative = template.format(
        episode_chunk=int(episode_index) // chunk_size,
        episode_index=int(episode_index),
    )
    return root / relative


def load_episode_rows(
    dataset_root: str | os.PathLike[str],
    episode_index: int,
    *,
    dataset_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet_path = episode_parquet_path(dataset_root, episode_index, dataset_info=dataset_info)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode parquet does not exist: {parquet_path}")
    return pq.read_table(parquet_path).to_pylist()


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


def decode_image_value(value: Any) -> Any:
    from PIL import Image

    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
        if image_bytes is not None:
            with Image.open(io.BytesIO(bytes(image_bytes))) as image:
                return as_uint8_hwc(image.convert("RGB"))
        if image_path is not None:
            with Image.open(image_path) as image:
                return as_uint8_hwc(image.convert("RGB"))
        raise ValueError(f"Image dict contains neither bytes nor path: {value!r}")
    return as_uint8_hwc(value)


def gripper_state_label(state: Any) -> str:
    import numpy as np

    array = np.asarray(state, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError("Cannot infer gripper state from an empty state vector.")
    return "open" if float(array[-1]) > 0 else "closed"


def overlay_step_text(
    frame: Any,
    *,
    step_idx: int,
    total_steps: int,
    instruction: str,
    gripper_state,
) -> Any:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)

    caption = f"step {step_idx:03d} / {total_steps - 1:03d}"
    subtitle = instruction

    draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0))
    draw.rectangle((0, image.height - 26, image.width, image.height), fill=(0, 0, 0))
    draw.text((8, 8), caption, fill=(255, 255, 255))

    # write gripper text
    gripper_text = f"gripper: {gripper_state_label(gripper_state)}"
    try:
        left, top, right, bottom = draw.textbbox((0, 0), gripper_text)
    except AttributeError:  # pragma: no cover - older Pillow fallback
        left, top, right, bottom = (0, 0, 8 * len(gripper_text), 11)
    text_width = right - left
    draw.text((max(8, image.width - text_width - 8), 8), gripper_text, fill=(255, 255, 255))

    draw.text((8, image.height - 21), subtitle, fill=(255, 255, 255))
    return image


def render_episode_video(
    episode_rows: list[dict[str, Any]],
    *,
    record: EpisodeRecord,
    output_path: str | os.PathLike[str],
    image_key: str = "top",
    overlay_text: bool = True,
    fps: int = DEFAULT_FPS,
) -> Path:
    import imageio.v2 as imageio

    resolved_image_key = resolve_image_key(image_key)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(output, fps=fps, codec="libx264", quality=7) as writer:
        for local_step, item in enumerate(episode_rows):
            frame = decode_image_value(item[resolved_image_key])
            if overlay_text:
                frame = overlay_step_text(
                    frame,
                    step_idx=local_step,
                    total_steps=record.length,
                    instruction=record.instruction,
                    gripper_state=item["observation.state"],
                )
            writer.append_data(as_uint8_hwc(frame))

    return output


def load_episode_frame(
    episode_rows: list[dict[str, Any]],
    *,
    record: EpisodeRecord,
    local_step: int,
    image_key: str = "top",
) -> Any:
    if not (0 <= int(local_step) < record.length):
        raise ValueError(
            f"Requested local step {local_step} is out of bounds for episode {record.episode_index} "
            f"with length {record.length}."
        )
    resolved_image_key = resolve_image_key(image_key)
    return decode_image_value(episode_rows[int(local_step)][resolved_image_key])


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
    episode_rows: list[dict[str, Any]],
    *,
    record: EpisodeRecord,
    segments: list[dict[str, Any]],
    output_dir: str | os.PathLike[str],
    image_key: str = "top",
) -> list[Path]:
    from PIL import Image

    paths = transition_scene_paths(
        output_dir,
        episode_index=record.episode_index,
        segments=segments,
    )
    for path, local_step in zip(paths, transition_boundary_steps_from_segments(segments)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = load_episode_frame(
            episode_rows,
            record=record,
            local_step=local_step,
            image_key=image_key,
        )
        Image.fromarray(frame).save(path)
    return paths


def jsonable_vector(value: Any) -> list[float]:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return [float(x) for x in array.tolist()]


def build_episode_annotation(
    *,
    record: EpisodeRecord,
    normalized: dict[str, Any],
    model: str,
    source_repo_id: str,
    dataset_root: Path,
    episode_rows: list[dict[str, Any]],
    prompt_version: str = PROMPT_VERSION,
    raw_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    for segment in normalized["segments"]:
        end_step = int(segment["end_step"])
        if end_step <= 0:
            raise ValueError(f"Segment must end after the first frame, got {segment}.")
        row = episode_rows[end_step - 1]
        segments.append(
            {
                "start_step": int(segment["start_step"]),
                "end_step": end_step,
                "skill": validate_skill_expr(str(segment["skill"])),
                "state": jsonable_vector(row["observation.state"]),
                "action": jsonable_vector(row["action"]),
            }
        )

    return {
        "episode_index": record.episode_index,
        "task_index": record.task_index,
        "task_name": record.task_name,
        "raw_instruction": record.raw_instruction,
        "instruction": record.instruction,
        "num_steps": record.length,
        "fps": DEFAULT_FPS,
        "plan": normalized["plan"],
        "segments": segments,
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
                instruction,
                plan_string,
                skill,
                include_instruction=True,
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
    prompt_version: str = PROMPT_VERSION,
) -> dict[str, Any]:
    top_level: dict[str, Any] = {
        "schema_version": "calvin_skill_annotation_v1",
        "prompt_version": prompt_version,
        "source_repo_id": source_repo_id,
        "dataset_root": str(dataset_root),
        "fps": DEFAULT_FPS,
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
    prompt_version: str = PROMPT_VERSION,
) -> dict[str, Any]:
    top_level: dict[str, Any] = {
        "schema_version": "calvin_skill_training_v1",
        "prompt_version": prompt_version,
        "source_repo_id": source_repo_id,
        "dataset_root": str(dataset_root),
        "fps": DEFAULT_FPS,
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


def episode_shard_path(shard_dir: str | os.PathLike[str], episode_index: int) -> Path:
    return Path(shard_dir) / f"episode_{episode_index:06d}.json"


def build_video_data_url(path: str | os.PathLike[str]) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"
