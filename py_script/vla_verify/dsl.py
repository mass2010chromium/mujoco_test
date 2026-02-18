"""
Domain-Specific Language (DSL) for scene-graph-executable actions.

Inspired by SayPlan's scene graph simulator, each DSL action is a state
transition on the scene graph with:
  - Preconditions: predicates that must hold on the current graph state.
  - Effects: modifications applied to produce the successor state.

Supported primitive actions for tabletop manipulation:
  PICK_UP, PLACE_ON, PLACE_IN, OPEN, CLOSE, RELEASE

The verify_and_apply() function is the core verification entry point.
"""

from dataclasses import dataclass, field
from typing import Optional

from .scene_graph import SceneGraph, SceneNode, SceneEdge


@dataclass
class VerificationResult:
    """Outcome of verifying a DSL action against a scene graph state."""
    success: bool
    message: str
    failed_preconditions: list = field(default_factory=list)
    resulting_graph: Optional[SceneGraph] = None


@dataclass
class DSLAction:
    """A parsed DSL action produced by the subtask translator."""
    action_type: str
    params: dict = field(default_factory=dict)
    reasoning: str = ""
    object_resolution: str = ""


def verify_and_apply(action: DSLAction, graph: SceneGraph) -> VerificationResult:
    """Verify preconditions and apply effects if satisfied."""
    dispatch = {
        "PICK_UP":  _verify_pick_up,
        "PLACE_ON": _verify_place_on,
        "PLACE_IN": _verify_place_in,
        "OPEN":     _verify_open,
        "CLOSE":    _verify_close,
        "RELEASE":  _verify_release,
    }
    handler = dispatch.get(action.action_type)
    if handler:
        return handler(action, graph)
    if action.action_type == "INVALID":
        return VerificationResult(
            success=False,
            message=f"Subtask marked INVALID by translator: {action.reasoning}",
            failed_preconditions=["subtask_invalid"],
        )
    return VerificationResult(
        success=False,
        message=f"Unknown action type: {action.action_type}",
        failed_preconditions=["unknown_action_type"],
    )


def _verify_pick_up(action, graph):
    """PICK_UP(object_id): grasp a movable object."""
    object_id = action.params.get("object_id")
    if not object_id:
        return VerificationResult(False, "PICK_UP requires object_id", ["missing_object_id"])

    obj = graph.get_node(object_id)
    if not obj:
        return VerificationResult(
            False, f"Object '{object_id}' does not exist in the scene graph", ["object_not_found"])

    failed = []
    if obj.node_type not in ("object",):
        failed.append(f"'{object_id}' is type '{obj.node_type}', not movable")
    if not graph.is_gripper_empty():
        held = graph.get_held_objects()
        held_names = [h[0].name for h in held]
        failed.append(f"Gripper not empty; holding: {', '.join(held_names)}")
    if "pick_up" not in obj.affordances:
        failed.append(f"Object '{obj.name}' lacks pick_up affordance")
    if obj.state.get("held", False):
        failed.append(f"Object '{obj.name}' already held")

    # Check accessibility (not inside closed container)
    in_edges = graph.find_edges(source=object_id, relation="in")
    for edge in in_edges:
        container = graph.get_node(edge.target)
        if container and container.state.get("open") is False:
            failed.append(f"Object '{obj.name}' inside closed '{container.name}'")

    if failed:
        return VerificationResult(False, "; ".join(failed), failed)

    # Apply effects
    new_graph = graph.deep_copy()
    agent = new_graph.get_agent_node()
    for rel in ("on", "in"):
        new_graph.remove_edges(source=object_id, relation=rel)
    new_graph.add_edge(SceneEdge(source=agent.id, target=object_id, relation="holding"))
    agent.state["gripper_empty"] = False
    new_graph.get_node(object_id).state["held"] = True
    return VerificationResult(True, f"PICK_UP({object_id}) verified and applied", resulting_graph=new_graph)


def _verify_place_on(action, graph):
    """PLACE_ON(object_id, target_id): place held object on a surface."""
    object_id = action.params.get("object_id")
    target_id = action.params.get("target_id")
    if not object_id or not target_id:
        return VerificationResult(False, "PLACE_ON requires object_id and target_id", ["missing_params"])

    obj = graph.get_node(object_id)
    if not obj:
        return VerificationResult(False, f"Object '{object_id}' not found", ["object_not_found"])
    target = graph.get_node(target_id)
    if not target:
        return VerificationResult(False, f"Target '{target_id}' not found", ["target_not_found"])

    failed = []
    if not obj.state.get("held", False):
        failed.append(f"Object '{obj.name}' is not currently held")
    if "place_on" not in target.affordances and target.node_type != "surface":
        failed.append(f"Target '{target.name}' does not support placement")
    if failed:
        return VerificationResult(False, "; ".join(failed), failed)

    new_graph = graph.deep_copy()
    agent = new_graph.get_agent_node()
    new_graph.remove_edges(source=agent.id, target=object_id, relation="holding")
    new_graph.add_edge(SceneEdge(source=object_id, target=target_id, relation="on"))
    agent.state["gripper_empty"] = True
    new_graph.get_node(object_id).state["held"] = False
    return VerificationResult(True, f"PLACE_ON({object_id}, {target_id}) verified", resulting_graph=new_graph)


def _verify_place_in(action, graph):
    """PLACE_IN(object_id, target_id): place held object inside container."""
    object_id = action.params.get("object_id")
    target_id = action.params.get("target_id")
    if not object_id or not target_id:
        return VerificationResult(False, "PLACE_IN requires object_id and target_id", ["missing_params"])

    obj = graph.get_node(object_id)
    if not obj:
        return VerificationResult(False, f"Object '{object_id}' not found", ["object_not_found"])
    target = graph.get_node(target_id)
    if not target:
        return VerificationResult(False, f"Target '{target_id}' not found", ["target_not_found"])

    failed = []
    if not obj.state.get("held", False):
        failed.append(f"Object '{obj.name}' not held")
    if "place_in" not in target.affordances:
        failed.append(f"Target '{target.name}' does not support place_in")
    if target.state.get("open") is False:
        failed.append(f"Container '{target.name}' is closed")
    if failed:
        return VerificationResult(False, "; ".join(failed), failed)

    new_graph = graph.deep_copy()
    agent = new_graph.get_agent_node()
    new_graph.remove_edges(source=agent.id, target=object_id, relation="holding")
    new_graph.add_edge(SceneEdge(source=object_id, target=target_id, relation="in"))
    agent.state["gripper_empty"] = True
    new_graph.get_node(object_id).state["held"] = False
    return VerificationResult(True, f"PLACE_IN({object_id}, {target_id}) verified", resulting_graph=new_graph)


def _verify_open(action, graph):
    """OPEN(target_id): open a container or drawer."""
    target_id = action.params.get("target_id")
    if not target_id:
        return VerificationResult(False, "OPEN requires target_id", ["missing_target_id"])
    target = graph.get_node(target_id)
    if not target:
        return VerificationResult(False, f"Target '{target_id}' not found", ["target_not_found"])

    failed = []
    if "open" not in target.affordances:
        failed.append(f"'{target.name}' cannot be opened")
    if target.state.get("open") is True:
        failed.append(f"'{target.name}' is already open")
    if failed:
        return VerificationResult(False, "; ".join(failed), failed)

    new_graph = graph.deep_copy()
    new_graph.get_node(target_id).state["open"] = True
    return VerificationResult(True, f"OPEN({target_id}) verified", resulting_graph=new_graph)


def _verify_close(action, graph):
    """CLOSE(target_id): close a container or drawer."""
    target_id = action.params.get("target_id")
    if not target_id:
        return VerificationResult(False, "CLOSE requires target_id", ["missing_target_id"])
    target = graph.get_node(target_id)
    if not target:
        return VerificationResult(False, f"Target '{target_id}' not found", ["target_not_found"])

    failed = []
    if "close" not in target.affordances:
        failed.append(f"'{target.name}' cannot be closed")
    if target.state.get("open") is not True:
        failed.append(f"'{target.name}' is not open")
    if failed:
        return VerificationResult(False, "; ".join(failed), failed)

    new_graph = graph.deep_copy()
    new_graph.get_node(target_id).state["open"] = False
    return VerificationResult(True, f"CLOSE({target_id}) verified", resulting_graph=new_graph)


def _verify_release(action, graph):
    """RELEASE(object_id): release the held object."""
    object_id = action.params.get("object_id")
    if not object_id:
        return VerificationResult(False, "RELEASE requires object_id", ["missing_object_id"])
    obj = graph.get_node(object_id)
    if not obj:
        return VerificationResult(False, f"Object '{object_id}' not found", ["object_not_found"])
    if not obj.state.get("held", False):
        return VerificationResult(False, f"Object '{obj.name}' not held", ["object_not_held"])

    new_graph = graph.deep_copy()
    agent = new_graph.get_agent_node()
    new_graph.remove_edges(source=agent.id, target=object_id, relation="holding")
    agent.state["gripper_empty"] = True
    new_graph.get_node(object_id).state["held"] = False
    return VerificationResult(True, f"RELEASE({object_id}) verified", resulting_graph=new_graph)
