"""
Scene graph data structures and operations for robotic manipulation verification.

Inspired by:
  - SayPlan (Rana et al., CoRL 2023): Hierarchical 3D scene graphs with
    state transitions, node affordances, and predicates for robot task planning.
  - ConceptGraphs (Gu et al., 2023): Open-vocabulary object-centric 3D scene
    graphs with node captions and LLM-inferred edge relationships.

The scene graph represents the state of a manipulation workspace at a discrete
symbolic timestep. Actions on the scene graph are state transitions with
preconditions (checked before) and effects (applied after).

Hierarchy for tabletop manipulation:
  workspace (root)
    ├── surfaces (table, counter)
    │   ├── objects (bowls, plates, blocks -- movable)
    │   └── assets (cabinets, drawers -- immovable fixtures)
    │       └── objects inside containers
    └── agent (robot gripper)
"""

import copy
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class SceneNode:
    """
    A node in the scene graph representing a physical entity.

    Attributes:
        id: Unique snake_case identifier (e.g., "black_bowl_1").
        name: Human-readable name (e.g., "black bowl").
        node_type: One of "object" (movable), "asset" (immovable fixture),
                   "surface" (support surface), "agent" (robot).
        attributes: Visual properties -- color, material, size, shape, etc.
        state: Current state -- e.g., {"held": False}, {"open": True},
               {"gripper_empty": True}.
        affordances: Possible actions -- e.g., ["pick_up"], ["place_on"],
                     ["open", "close", "place_in"].
        location_description: Brief spatial description.
    """
    id: str
    name: str
    node_type: str
    attributes: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    affordances: list = field(default_factory=list)
    location_description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SceneNode":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SceneEdge:
    """
    An edge in the scene graph encoding a spatial or functional relationship.

    Relations:
        on      -- source rests on top of target
        in      -- source is inside target
        next_to -- source is adjacent to target
        near    -- source is in the vicinity of target
        part_of -- source is a structural part of target (e.g., drawer part_of cabinet)
        holding -- agent is holding the object (source=agent, target=object)
    """
    source: str
    target: str
    relation: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SceneEdge":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SceneGraph:
    """
    A scene graph representing the symbolic state of a robotic manipulation
    workspace. Supports querying, modification, and deep copying for
    state-transition verification.
    """

    def __init__(
        self,
        nodes: Optional[dict] = None,
        edges: Optional[list] = None,
    ):
        self.nodes: dict[str, SceneNode] = nodes or {}
        self.edges: list[SceneEdge] = edges or []

    # ── Node operations ─────────────────────────────────────────────────

    def add_node(self, node: SceneNode):
        self.nodes[node.id] = node

    def remove_node(self, node_id: str):
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.edges = [
                e for e in self.edges
                if e.source != node_id and e.target != node_id
            ]

    def get_node(self, node_id: str) -> Optional[SceneNode]:
        return self.nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    # ── Edge operations ─────────────────────────────────────────────────

    def add_edge(self, edge: SceneEdge):
        self.edges.append(edge)

    def remove_edges(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        relation: Optional[str] = None,
    ):
        """Remove all edges matching the given filter(s)."""
        def matches(e: SceneEdge) -> bool:
            if source is not None and e.source != source:
                return False
            if target is not None and e.target != target:
                return False
            if relation is not None and e.relation != relation:
                return False
            return True
        self.edges = [e for e in self.edges if not matches(e)]

    def find_edges(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        relation: Optional[str] = None,
    ) -> list[SceneEdge]:
        """Find all edges matching the given filter(s)."""
        results = []
        for e in self.edges:
            if source is not None and e.source != source:
                continue
            if target is not None and e.target != target:
                continue
            if relation is not None and e.relation != relation:
                continue
            results.append(e)
        return results

    # ── Query helpers ───────────────────────────────────────────────────

    def find_nodes_by_type(self, node_type: str) -> list[SceneNode]:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def find_nodes_by_name(self, name: str) -> list[SceneNode]:
        """Find nodes whose name contains the search string (case-insensitive)."""
        name_lower = name.lower()
        return [n for n in self.nodes.values() if name_lower in n.name.lower()]

    def find_nodes_by_attribute(self, key: str, value: str) -> list[SceneNode]:
        return [
            n for n in self.nodes.values()
            if str(n.attributes.get(key, "")).lower() == value.lower()
        ]

    def get_agent_node(self) -> Optional[SceneNode]:
        agents = self.find_nodes_by_type("agent")
        return agents[0] if agents else None

    def get_held_objects(self) -> list[tuple[SceneNode, SceneEdge]]:
        """Return (node, holding_edge) for all objects the agent is holding."""
        agent = self.get_agent_node()
        if not agent:
            return []
        holding_edges = self.find_edges(source=agent.id, relation="holding")
        results = []
        for edge in holding_edges:
            node = self.get_node(edge.target)
            if node:
                results.append((node, edge))
        return results

    def is_gripper_empty(self) -> bool:
        return len(self.get_held_objects()) == 0

    # ── Copy & serialization ────────────────────────────────────────────

    def deep_copy(self) -> "SceneGraph":
        """Create a deep copy for state-transition simulation."""
        new_nodes = {}
        for nid, n in self.nodes.items():
            new_nodes[nid] = SceneNode(
                id=n.id,
                name=n.name,
                node_type=n.node_type,
                attributes=copy.deepcopy(n.attributes),
                state=copy.deepcopy(n.state),
                affordances=list(n.affordances),
                location_description=n.location_description,
            )
        new_edges = [SceneEdge(e.source, e.target, e.relation) for e in self.edges]
        return SceneGraph(new_nodes, new_edges)

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "SceneGraph":
        nodes = {}
        for nd in d.get("nodes", []):
            node = SceneNode.from_dict(nd)
            nodes[node.id] = node
        edges = [SceneEdge.from_dict(ed) for ed in d.get("edges", [])]
        return cls(nodes, edges)

    @classmethod
    def from_json(cls, json_str: str) -> "SceneGraph":
        return cls.from_dict(json.loads(json_str))

    # ── Pretty printing ─────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of the scene graph state."""
        lines = ["=== Scene Graph State ==="]
        lines.append(f"Nodes ({len(self.nodes)}):")
        for n in self.nodes.values():
            state_str = (
                ", ".join(f"{k}={v}" for k, v in n.state.items())
                if n.state else "none"
            )
            attr_str = (
                ", ".join(f"{k}={v}" for k, v in n.attributes.items())
                if n.attributes else "none"
            )
            lines.append(
                f"  [{n.node_type:7s}] {n.id}: \"{n.name}\" "
                f"| attrs: {attr_str} | state: {state_str} "
                f"| affordances: {n.affordances}"
            )
        lines.append(f"\nEdges ({len(self.edges)}):")
        for e in self.edges:
            lines.append(f"  {e.source} --[{e.relation}]--> {e.target}")
        return "\n".join(lines)
