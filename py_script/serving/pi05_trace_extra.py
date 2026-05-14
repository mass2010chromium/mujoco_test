from typing import Any

from skill_processing import (
    get_skill_name,
    parse_plan
)

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


GEMINI_COORDINATE_GRID = 1024
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
    name = get_skill_name(skill_text)
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

def build_semantic_point_prompt(task_instruction: str, plan_str: str, skill_text: str,
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
3. Tomato sauce is the red and green can.
4. Alphabet soup is the blue and yellow can.
5. The back compartment of the caddy is at the center back of the caddy, not toward the right or left side.

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
