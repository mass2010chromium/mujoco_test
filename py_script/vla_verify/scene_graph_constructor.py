"""
Construct a scene graph from an image using a Vision-Language Model (VLM).

The VLM analyzes the initial camera observation and produces a structured
scene graph following our schema. This provides the initial symbolic state
before any actions are applied.

The prompt design is informed by:
  - SayPlan's scene graph representation (nodes with type, state, affordances,
    attributes; edges with spatial relations).
  - ConceptGraphs' object-centric approach (detailed captions, spatial reasoning).
"""

import json
import re
from pathlib import Path
from typing import Optional

from .openrouter_client import query_vlm, DEFAULT_VLM_MODEL
from .scene_graph import SceneGraph, SceneNode, SceneEdge


SCENE_GRAPH_PROMPT = """\
You are a scene analysis system for a robotic manipulation workspace. \
Given an image of a tabletop scene with a robot arm, construct a detailed \
scene graph in JSON format.

The scene graph must capture ALL visible entities and their spatial \
relationships. This will be used to verify robot action plans, so accuracy \
and completeness are critical.

For each node, provide these fields:
- "id": unique snake_case identifier (e.g., "black_bowl_1", "wooden_table"). \
Use numbered suffixes when there are multiple similar objects.
- "name": human-readable name (e.g., "black bowl", "wooden table")
- "node_type": one of:
  * "object" - movable items the robot can pick up (bowls, cups, blocks, etc.)
  * "asset" - immovable fixtures (cabinets, drawers, shelves - can be \
opened/closed but not picked up)
  * "surface" - flat support surfaces (table, counter)
  * "agent" - the robot arm/gripper
- "attributes": dict of visual properties, e.g. \
{"color": "black", "material": "ceramic", "size": "small", "shape": "round"}
- "state": dict of current state:
  * For containers/drawers: {"open": true/false}
  * For the agent gripper: {"gripper_empty": true/false}
  * For objects: {"held": false} (nothing is held initially)
- "affordances": list from: \
["pick_up", "place_on", "place_in", "open", "close", "push"]
  * Movable objects: ["pick_up"]
  * Flat surfaces / plates: ["place_on"]
  * Containers with openable lids/drawers: ["place_in", "open", "close"]
  * Objects that also serve as surfaces (e.g., plate): ["pick_up", "place_on"]
- "location_description": brief text describing spatial position in the scene

For each edge (relationship), provide:
- "source": node id of the entity being described
- "target": node id of the reference entity
- "relation": one of "on", "in", "next_to", "part_of", "holding", "near"
  * "on" - source rests on top of target
  * "in" - source is inside target
  * "next_to" - source is adjacent to target
  * "part_of" - source is a structural component of target
  * "near" - source is in the vicinity of target

IMPORTANT GUIDELINES:
1. List EVERY visible object, even small or partially occluded ones.
2. Distinguish similar objects with numbered IDs \
(e.g., "black_bowl_1", "black_bowl_2").
3. Include the robot arm as an agent node with id "robot_gripper" and \
state {"gripper_empty": true}.
4. ONLY include objects you can actually see - do NOT hallucinate objects.
5. For drawers/cabinets, note whether they are open or closed.
6. Be precise about colors and object types.

Output ONLY valid JSON (no markdown fences, no explanation) with this structure:
{
  "nodes": [
    {
      "id": "example_object_1",
      "name": "example object",
      "node_type": "object",
      "attributes": {"color": "red", "material": "plastic"},
      "state": {"held": false},
      "affordances": ["pick_up"],
      "location_description": "on the left side of the table"
    }
  ],
  "edges": [
    {"source": "example_object_1", "target": "table", "relation": "on"}
  ]
}"""


def _extract_json_from_response(text: str) -> dict:
    """Extract JSON from a VLM response that may contain markdown fences."""
    # Strip markdown code fences if present
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if code_block:
        json_str = code_block.group(1).strip()
    else:
        # Find the outermost { ... }
        brace_start = text.find("{")
        if brace_start < 0:
            raise json.JSONDecodeError("No JSON object found", text, 0)
        depth = 0
        json_end = brace_start
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break
        json_str = text[brace_start:json_end]

    return json.loads(json_str)


def construct_scene_graph(
    image_path: Path,
    vlm_model: str = DEFAULT_VLM_MODEL,
    api_key: Optional[str] = None,
) -> SceneGraph:
    """
    Construct a scene graph from an image using a VLM.

    Args:
        image_path: Path to the scene image.
        vlm_model: OpenRouter model ID for the VLM.
        api_key: OpenRouter API key (or from env).

    Returns:
        SceneGraph representing the initial scene state.
    """
    print(f"  [VLM] Querying {vlm_model} for scene graph construction...")
    raw_response = query_vlm(
        image_path=image_path,
        prompt=SCENE_GRAPH_PROMPT,
        model=vlm_model,
        api_key=api_key,
        temperature=0.1,
    )
    print(f"  [VLM] Response received ({len(raw_response)} chars)")

    try:
        graph_dict = _extract_json_from_response(raw_response)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  [VLM] WARNING: Failed to parse JSON: {e}")
        print(f"  [VLM] Raw response (first 2000 chars):\n{raw_response[:2000]}")
        raise ValueError(f"VLM returned invalid JSON: {e}") from e

    sg = SceneGraph.from_dict(graph_dict)
    _post_process(sg)

    print(
        f"  [VLM] Scene graph constructed: "
        f"{len(sg.nodes)} nodes, {len(sg.edges)} edges"
    )
    return sg


def _post_process(sg: SceneGraph):
    """Fix common VLM output issues to ensure graph consistency."""
    # Ensure an agent node exists
    agent = sg.get_agent_node()
    if not agent:
        sg.add_node(SceneNode(
            id="robot_gripper",
            name="robot gripper",
            node_type="agent",
            attributes={},
            state={"gripper_empty": True},
            affordances=[],
            location_description="above the workspace",
        ))
    else:
        if "gripper_empty" not in agent.state:
            agent.state["gripper_empty"] = True

    for node in sg.nodes.values():
        # Ensure movable objects have the held state
        if node.node_type == "object":
            if "held" not in node.state:
                node.state["held"] = False
            if "pick_up" not in node.affordances:
                node.affordances.append("pick_up")

        # Ensure surfaces support placement
        if node.node_type == "surface":
            if "place_on" not in node.affordances:
                node.affordances.append("place_on")

        # Ensure assets that look like containers have container affordances
        if node.node_type == "asset":
            has_open_state = "open" in node.state
            if has_open_state:
                if "open" not in node.affordances:
                    node.affordances.append("open")
                if "close" not in node.affordances:
                    node.affordances.append("close")
                if "place_in" not in node.affordances:
                    node.affordances.append("place_in")
