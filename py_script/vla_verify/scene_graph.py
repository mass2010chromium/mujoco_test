from dataclasses import dataclass
import textwrap
import time

import numpy as np
from PIL import Image

from llm_apis import transformers_api
from llm_apis.response_parsing import extract_in_backticks

from .pddl_parsing import setup_pddl_simulation
from pddlsim.simulation import Simulation

@dataclass
class SceneObject:
    object_type: str    # Corresponding to object type in PDDL
    object_id: str      # Corresponding to object id in PDDL
    appearance: str     # LLM text description of object
    location: str       # LLM text description of object location
                        # TODO: 3d location
    grounding = None

    def to_dict(self, include_grounding=True):
        res = {
            'type': self.object_type,
            'id': self.object_id,
            'appearance': self.appearance,
            'location': self.location,
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
        self.raw_pddl_state = None

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

    def _read_image(self, llm_response, images, hint=[], ground=True):
        """
        Read an RGB image with a VLM, output a PDDL domain file and parse that into a scene graph.
        """
        if type(images) == list:
            image_rgb = images[0]
        else:
            image_rgb = images
            images = [image_rgb]
        t0 = time.monotonic()
        print(f"  [VLM] Querying VLM for scene graph construction...")
        yield [ transformers_api.make_message(texts=hint, images=[image_rgb]) ]
        t1 = time.monotonic()
        print(f"  [VLM] Time elapsed: {t1 - t0}")

        # Strip triple backticks
        raw_response = llm_response['content']
        response = extract_in_backticks(raw_response, 'pddl')
        if response is None:
            print("Warning: No triple backticks given")
            response = raw_response

        yield self.construct_from_pddl(images, response, ground=ground)
    READ_IMAGE_PROMPT = textwrap.dedent("""\
    The following is a PDDL domain description for a generic pick and place task:

    ```pddl
    {pddl_domain}
    ```

    Given an image, detect the relevant objects in the image, and output a
    corresponding PDDL problem file for this image. Omit the `goal` clause.
    Define each distinct object on its own line, using comments to add descriptions
    strictly using the following format, separated by a vertical bar:

    <object appearance> | <object location>

    As part of the location, specify if the object is in the foreground or background.

    If multiple objects are of the same type are present, include positional descriptions such as "left", "right", "front", and "back". 
    Note that any "left", "right", "front", and "back" descriptions should be with respect to the robot's perspective, which is opposite to the image's perspective. 

    The robot should be called `robot_0` and have no description comment.

    If an object involves different articulated components, each component should be defined as a separate object.
    For example, a cabinet can have different drawers, a caddy can be divided into different compartments, and a wine rack can have different shelves.

    If a "Task Instruction" is given, assume that the objects mentioned in the instruction are present in the scene, and you should attempt to match the objects to the objects in the scene.
    
    Remember not to use logical expressions in the initial state -- only use predicates.

    """)

    def construct_from_pddl(self, images, raw_pddl_state, ground=True):
        print("raw_pddl_state", raw_pddl_state)
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
                location=location,
            )
            self.object_data[object_id] = obj

            if object_type != "robot":
                full_descriptions.append(f"a {appearance} {location}")
                label_descriptions.append(f"a {appearance}")
                object_ids.append(object_id)

        if ground and self.grounder is not None:
            video_data = [Image.fromarray(image_rgb) for image_rgb in images]
            # Much faster and seems more accurate than gemini.
            mask_results = self.grounder.predict_masks_video(video_data, full_descriptions, label_descriptions, use_clip=True)
            for object_id, result in zip(object_ids, mask_results):
                self.object_data[object_id].grounding = result


    def ground_video(self, additional_points_labels=[]):
        """
        Spread the grounding to the entire video.
        """
        def mask_to_points(mask, n_sample=4):
            indices = np.argwhere(mask)[:, ::-1]
            sample = np.random.choice(len(indices), n_sample, replace=False)
            return indices[sample]

        point_requests = []
        obj_order = []
        for obj_id, obj in self.object_data.items():
            if obj.grounding is None:
                continue
            points_abs = mask_to_points(obj.grounding.mask)
            obj_order.append(obj_id)
            point_requests.append((len(point_requests), points_abs))

        for name, points in additional_points_labels:
            obj_order.append(name)
            point_requests.append((len(point_requests), points))

        results = self.grounder.propagate_all_detections(point_requests)
        return [{obj_order[i]: v for i, v in result.items() } for result in results]



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
    

    def apply_action(self, action):
        """
        Apply an action to the simulator. Return True if successful, False otherwise.
        """
        return self.simulator.apply_grounded_action(action)


    def match_grounded_action(self, action_name: str, action_params: list[str]):
        """
        Match a grounded action to an action name and parameters.
        """
        for action in self.simulator.get_grounded_actions():
            if action.name.value != action_name:
                    continue
            params = [obj.value for obj in action.grounding]
            if params == action_params:
                return action
        return None

    def reset_simulator(self):
        """
        Reset the simulator to the initial state.
        """
        self.simulator = Simulation.from_domain_and_problem(
            self.domain,
            self.init_state,
        )


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
