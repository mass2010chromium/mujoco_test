from dataclasses import dataclass
import textwrap
import time

import numpy as np
from PIL import Image

from llm_apis import transformers_api
from llm_apis.response_parsing import extract_in_backticks

from .pddl_parsing import setup_pddl_simulation

@dataclass
class SceneObject:
    object_type: str    # Corresponding to object type in PDDL
    object_id: str      # Corresponding to object id in PDDL
    appearance: str     # LLM text description of object
                        # TODO: 3d location
    grounding = None

    def to_dict(self, include_grounding=True):
        res = {
            'type': self.object_type,
            'id': self.object_id,
            'appearance': self.appearance,
        }
        if include_grounding and self.grounding is not None:
            res['grounding'] = self.grounding.to_dict()
        return res

class TaskSceneGraph:
    """
    Scene graph grounded to an image and a PDDL representation.
    """
    DUMMY_PDDL_GOAL = "(:goal (free {robot}))"
    def __init__(self, pddl_domain_desc: str, vlm_interface, gpus_to_use=[0]):
        self.pddl_domain_desc = pddl_domain_desc
        self.domain = None
        self.init_state = None
        self.simulator = None
        self.object_data = None

        self.read_image = vlm_interface(
            self._read_image,
            system_prompt=TaskSceneGraph.READ_IMAGE_PROMPT.format(pddl_domain=pddl_domain_desc)
        )

        self._grounder = None
        self._grounder_args = [gpus_to_use]

    @property
    def grounder(self):
        try:
            from .object_grounder import ObjectGrounder
            if self._grounder is None:
                self._grounder = ObjectGrounder(*self._grounder_args)
        except ImportError as e:
            import traceback
            traceback.print_exc()
            print("WARNING: SAM 3 not installed. Grounding is disabled")
        return self._grounder

    def _read_image(self, llm_response, image_rgb, ground=True):
        """
        Read an RGB image with a VLM, output a PDDL domain file and parse that into a scene graph.
        """
        t0 = time.monotonic()
        print(f"  [VLM] Querying VLM for scene graph construction...")
        yield [ transformers_api.make_message(images=[image_rgb]) ]
        t1 = time.monotonic()
        print(f"  [VLM] Time elapsed: {t1 - t0}")

        # Strip triple backticks
        raw_response = llm_response['content']
        response = extract_in_backticks(raw_response, 'pddl')
        if response is None:
            print("Warning: No triple backticks given")
            response = raw_response

        yield self.construct_from_pddl(image_rgb, response, ground=ground)
    READ_IMAGE_PROMPT = textwrap.dedent("""\
    The following is a PDDL domain description for a generic pick and place task:

    ```pddl
    {pddl_domain}
    ```

    Given an image, detect the relevant objects in the image, and output a
    corresponding PDDL problem file for this image. Omit the `goal` clause.
    Define each distinct object on its own line, using comments to add descriptions
    in the following format, separated by a vertical bar:

    <object appearance> | <object location>

    The robot should be called `robot_0` and have no description comment.
    Remember not to use logical expressions in the initial state -- only use predicates.

    """)

    def construct_from_pddl(self, image_rgb, raw_pddl_state, ground=True):
        try:
            self.init_state, self.domain, self.simulator = setup_pddl_simulation(
                raw_pddl_state, self.pddl_domain_desc
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("PDDL Parse Error!")
            print(raw_pddl_state)
            return
        lines = raw_pddl_state.split('\n')

        # These two are used for grounding.
        full_descriptions = []
        label_descriptions = []
        object_ids = []
        self.object_data = {}
        for v in self.init_state.objects_section:
            #print(v)
            #print(v.type, v.value, type(v), type(v.type))
            pddl_line = lines[v.type.location.line-1]
            if ';' in pddl_line:
                comment = pddl_line.split(';', 1)[1]
                appearance, location = [x.strip() for x in comment.split('|')]
            else:
                appearance, location = (None, None)
            object_type, object_id = v.type.value, v.value.value
            obj = SceneObject(
                object_type=object_type,
                object_id=object_id,
                appearance=appearance,
                #location=location,
            )
            self.object_data[object_id] = obj

            if object_type != "robot":
                full_descriptions.append(f"a {appearance} {location}")
                label_descriptions.append(f"a {appearance}")
                object_ids.append(object_id)

        if ground and self.grounder is not None:
            video_data = [Image.fromarray(image_rgb)]
            # Much faster and seems more accurate than gemini.
            mask_results = self.grounder.predict_masks_video(video_data, full_descriptions, label_descriptions, use_clip=True)
            for object_id, result in zip(object_ids, mask_results):
                self.object_data[object_id].grounding = result


    def get_available_actions(self):
        """
        Get actions that are available from PDDL.

        Maybe there will be actions that cannot be described in pddl... how will that be handled?
        Need to manually set predicates. But this can break the simulator.
        """
        return [
            {
                'name': x.name.value,
                'parameters': [g.value for g in x.grounding]
            }
            for x in self.simulator.get_grounded_actions()
        ]

    def to_dict(self, include_grounding=False):
        """Machine readable summary of scene graph objects"""
        all_predicates = list(self.simulator.state)
        edges = []
        attrs = {obj_id: [] for obj_id in self.object_data.keys()}
        for pred in all_predicates:
            name = pred.name.value
            targets = [x.value for x in pred.assignment]
            if len(targets) == 1:   # Assume attribute
                attrs[targets[0]].append(name)
            elif len(targets) == 2: # Assume directed edge
                edges.append((name, *targets))
        node_data = []
        for obj in self.object_data.values():
            data = obj.to_dict()
            data['attributes'] = attrs[obj.object_id]
            node_data.append(data)
        return {
            'nodes': node_data,
            'edges': [
                {
                    'from': x,
                    'to': y,
                    'relation': r
                }
                for r, x, y in edges
            ]
        }

    def summary(self) -> str:
        """Human-readable summary of the scene graph state."""
        info = self.to_dict()

        lines = ["=== Scene Graph State ==="]
        lines.append(f"Nodes ({len(info['nodes'])}):")
        for data in info['nodes']:
            attr_str = (
                ", ".join(data['attributes'])
            )
            lines.append(
                f"  [{data['type']:14s}] {data['id']}: \n"
                f"| description: '{data['appearance']}'\n"
                f"| attrs: {attr_str}"
            )

        lines.append(f"\nEdges ({len(info['edges'])})")
        for e in info['edges']:
            lines.append(f"  {e['from']} --[{e['relation']}]--> {e['to']}")
        return '\n'.join(lines)
