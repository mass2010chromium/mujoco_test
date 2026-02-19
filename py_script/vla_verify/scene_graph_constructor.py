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
import time

import numpy as np

from llm_apis.llm_tool import extract_json_from_response
from llm_apis import transformers_api

from .scene_graph import SceneGraph, SceneNode


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


def scene_graph_from_image(llm_response, image_rgb: np.ndarray):
    """
    Interact with an LLM to construct a scene graph from an image observation.
    """
    t0 = time.monotonic()
    print(f"  [VLM] Querying VLM for scene graph construction...")
    yield [ transformers_api.make_message(images=[image_rgb]) ]

    raw_response = llm_response['content']
    try:
        scene_graph_json = extract_json_from_response(raw_response)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  [VLM] WARNING: Failed to parse JSON: {e}")
        print(f"  [VLM] Raw response (first 2000 chars):\n{raw_response[:2000]}")
        raise ValueError(f"VLM returned invalid JSON: {e}") from e
    t1 = time.monotonic()
    print(f"  [VLM] Time elapsed: {t1 - t0}")
    sg = SceneGraph.from_dict(scene_graph_json)

    # Postprocessing to fix common VLM issues
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
    yield sg
scene_graph_from_image.system_prompt = SCENE_GRAPH_PROMPT
