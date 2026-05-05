from dataclasses import dataclass
import json
import textwrap
import time

import numpy as np
from PIL import Image

from llm_apis import transformers_api
from llm_apis.response_parsing import extract_in_backticks, extract_json_from_response

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

    def to_dict(self, include_location=True, include_grounding=True):
        res = {
            'type': self.object_type,
            'id': self.object_id,
            'appearance': self.appearance,
        }
        if include_location:
            res['location'] = self.location,
        if include_grounding and self.grounding is not None:
            res['grounding'] = self.grounding.to_dict()
        return res

    def as_pddl_string(self):
        s = f"{self.object_id} - {self.object_type}"
        if self.appearance is not None:
            s += f" ; {self.appearance} | {self.location}"
        return s

class TaskSceneGraph:
    """
    Scene graph grounded to an image and a PDDL representation.
    """
    DUMMY_PDDL_GOAL = "(:goal (free {robot}))"
    def __init__(self, pddl_domain_desc: str, vlm_interface, gpus_to_use=[0], debug=True):
        self.pddl_domain_desc = pddl_domain_desc
        self.domain = None
        self.init_state = None
        self.simulator = None
        self.object_data = None

        self.read_image = vlm_interface(
            self._read_image,
            system_prompt=TaskSceneGraph.READ_IMAGE_PROMPT.format(pddl_domain=pddl_domain_desc)
        )
        self.ground_openrouter = vlm_interface(
            self._ground_openrouter
        )
        self.update_scene_graph = vlm_interface(
            self._update_scene_graph,
            system_prompt=TaskSceneGraph.UPDATE_GRAPH_SYSTEM_PROMPT.format(pddl_domain=pddl_domain_desc)
        )

        self.debug = debug
        self._grounder = None
        # NOTE: debug is functionally immutable
        self._grounder_args = [gpus_to_use, self.debug]
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

    def _ground_openrouter(self, llm_response, image_rgb, target_object: SceneObject, task_hint=None):
        """
        Get the (relative) pixel position of an object in the image.

        Returns a dict with (up to) two fields:
            status:     OK | NOT_FOUND
            position:   [x, y] normalized image coords (top left is 0, 0; bottom right is 1, 1)
        """
        image_height, image_width = image_rgb.shape[:2]
        t0 = time.monotonic()
        print(f"  [VLM] Querying VLM for grounding...")
        user_prompt = TaskSceneGraph.GROUND_USER_TEMPLATE.format(
            image_width=image_width,
            image_height=image_height,
            scene_graph_pddl=self.pddl_summary(),
            object_json=json.dumps(target_object.to_dict(include_grounding=False), indent=2),
            task_hint=task_hint
        )
        yield [ transformers_api.make_message(texts=user_prompt, images=[image_rgb]) ]
        t1 = time.monotonic()
        print(f"  [VLM] Time elapsed: {t1 - t0}")

        raw_response = llm_response['content']
        try:
            result = extract_json_from_response(raw_response)
            assert (result['status'] == "OK" or result['status'] == "NOT_FOUND")
            result['position'] = np.array(result['position'])[::-1] / [image_width, image_height]
            yield result
        except Exception as e:
            print(f"Warning: VLM response extraction raised exception: {e}")
            yield { "status": "NOT_FOUND" }
    _ground_openrouter.system_prompt = textwrap.dedent("""\
    Given an image, give the pixel position of the best match to a given target.

    The target is given as an object, and you might also get a task hint as to where specifically on the target \
    you should be selecting. For example, you might be told to find the position of the table, but the task says \
    PLACE(box, to the right of the can). In this case you should find a point on the table that is to the right \
    of a can on the table, and select that point.

    The possible task hint formats are:

    1) PLACE_ON(object1, object2)
    - description: place object1 onto object2
    - object1: the object being placed
    - object2: object that will support object1

    2) PLACE_IN(object1, object2)
    - description: place object1 into object2
    - object1: the object being placed
    - object2: object that will contain object1

    3) PICKUP_FROM(object1, object2)
    - description: pick up object1 from object2
    - object1: the object being picked up. This must be a movable object that can be picked up.
    - object2: object that supports object1 originally

    4) OPEN(object)
    - object: the object being opened

    5) CLOSE(object)
    - object: the object being closed

    6) TURN_ON(object)
    - object: the object being turned on

    7) TURN_OFF(object)
    - object: the object being turned off
    
    As help, a scene graph has been constructed out containing the known objects in the image.
    The given object corresponds to one of these.

    Return your response in the following format:

    ```json
    {
      "status": "OK" | "NOT_FOUND",
      "reasoning": <natural language explanation>,
      "position": [pixel_row, pixel_column] (integers, leave out if not found)
    }
    ```
    """)
    GROUND_USER_TEMPLATE = textwrap.dedent("""\
    The image is {image_width} by {image_height} (width x height).
    
    Current scene state:
    {scene_graph_pddl}

    Object:
    {object_json}

    Hint:
    {task_hint}
    """)


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
        # One time for some bizzare reason I saw gemini output ```lisp instead of ```pddl.
        # So let's be maximally permissive and accept ```([^\s]*)
        response = extract_in_backticks(raw_response, r"[^\s]*")
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

    As part of the appearance, you must ALWAYS describe the color of the object.
    Make sure to be accurate with your descriptions and disambiguate objects as much as possible.

    If multiple objects are of the same type are present, include positional descriptions such as "left", "right", "front", and "back". 
    "front" should refer to objects closer to the viewer.

    The description should form an english phrase, for example, "a white cup | on the left side of the table".

    The robot should be called `robot_0` and have no description comment.

    If an object involves different articulated components, each component should be defined as a separate object.
    For example, a cabinet can have different drawers, a caddy can be divided into different compartments, and a wine rack can have different shelves.
    Additionally define the entire object as a separate pddl object, using the `part` predicate to relate them instead of the `on` or `in` predicates.

    If a "Task Instruction" is given, assume that the objects mentioned in the instruction are present in the scene, and you should attempt to match the objects to the objects in the scene.

    Additional rules:
     - Never use logical expressions (and, or, not) when describing the initial state -- only use predicates themselves. It is not necessary to specify that something is both "open" and "not closed".
     - Never use shorthand when specifying relationships between objects -- references to objects match the object IDs exactly. For example, never shorten "compartment" to "compart" when specifying affordances, part relationships, or state relationships (on/in).

    Always output a PDDL problem file to describe the scene, omitting the goal clause.
    """)
    #Note that any "left", "right", "front", and "back" descriptions should be with respect to the robot's perspective, which is opposite to the image's perspective. 

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
            return False
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
        return True


    def ground_video(self, additional_points_labels=[]):
        """
        Spread the grounding to the entire video.
        """
        def mask_to_points(grounding, n_sample=4):
            indices = np.argwhere(grounding.mask)[:, ::-1]
            if len(indices) > 0:
                sample = np.random.choice(len(indices), n_sample, replace=False)
                return indices[sample]
            else:
                middle = (grounding.box[:2] + grounding.box[2:]) / 2
                return np.array([middle])

        point_requests = []
        obj_order = []
        for obj_id, obj in self.object_data.items():
            if obj.grounding is None:
                continue
            points_abs = mask_to_points(obj.grounding)
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
    

    def apply_action(self, action, new_image=None):
        """
        Apply an action to the simulator. Return True if successful, False otherwise.
        """
        if new_image is not None:
            self.update_scene_graph(str(action), new_image)
        return self.simulator.apply_grounded_action(action)

    def _update_scene_graph(self, llm_response, action_str, new_image):
        user_text = TaskSceneGraph.UPDATE_GRAPH_USER_TEMPLATE.format(
            pddl_scene=self.pddl_summary(),
            pddl_action=action_str
        )
        t0 = time.monotonic()
        print(f"  [VLM] Querying VLM for scene graph update...")
        yield [ transformers_api.make_message(texts=user_text, images=[new_image]) ]
        t1 = time.monotonic()
        print(f"  [VLM] Time elapsed: {t1 - t0}")
        raw_response = llm_response['content']
        try:
            results = extract_json_from_response(raw_response)
            for result in results:
                obj = self.object_data[result['object_id']]
                obj.appearance = result['appearance']
                obj.location = result['location']
            
        except Exception as e:
            print(f"Warning: VLM response extraction raised exception: {e}")
        yield None

    UPDATE_GRAPH_SYSTEM_PROMPT = textwrap.dedent("""\
    The following is a PDDL domain description for a generic pick and place task:

    ```pddl
    {pddl_domain}
    ```

    Given an image of the world, a description of the prior state, and the previous \
    action taken by the robot, update the appearance and location descriptions for \
    objects in the scene.

    Object descriptions in the scene description are formatted as PDDL comments:
    <object id> - <object PDDL type> ; <object appearance> | <object location>

    For example, if the action is `(pickup_from bowl0 robot table)`, you know that \
    bowl0 is the object of interest, and that it should be gripped by the robot. \

    You should cross reference this against the given image -- for example, if the \
    action is `(place_on bowl0 robot table)`, the PDDL action does not specify where \
    the bowl was placed, so you should describe the location by looking at the image.

    Format your descriptions as two parts of a sentence. For example:

    ```json
    {{
      "appearance": "a white bowl",
      "location": "on the left side of the table"
    }}
    ```

    Return your response as a list of changed descriptions. Return an empty list if \
    nothing changed.
    ```json
    [
      {{
        "object_id": <object id of the target object>
        "appearance": <description of the object's appearance>
        "location": <description of the object's new location.>
      }}
      ...
    ]
    ```

    """)
    UPDATE_GRAPH_USER_TEMPLATE = textwrap.dedent("""\
    Previous scene state:
    {pddl_scene}

    Action taken:
    {pddl_action}
    """)


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


    def to_dict(self, include_location=True, include_grounding=False):
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
            data = obj.to_dict(include_location=include_location, include_grounding=include_grounding)
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

    def pddl_summary(self) -> str:
        lines = []
        lines.append("(:objects")
        lines.append(textwrap.indent('\n'.join(o.as_pddl_string() for o in self.object_data.values()), "  "))
        lines.append(")\n(:state")
        lines.append(textwrap.indent('\n'.join(repr(s) for s in self.simulator.state), "  "))
        lines.append(")")
        return '\n'.join(lines)

    def summary(self, *args, header=True, **kwargs) -> str:
        """Human-readable summary of the scene graph state."""
        info = self.to_dict(*args, **kwargs)

        lines = []
        if header:
            lines.append("=== Scene Graph State ===")
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
