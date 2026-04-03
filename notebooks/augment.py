    import gzip
import json
import os
from pathlib import Path
import re
import sys
SCRIPT_DIR = Path(__file__).resolve().parent

import cv2
import einops
import matplotlib.pyplot as plt
import mediapy
import numpy as np
import scipy

from openpi.policies.libero_reason_dataset import LiberoSkillReasonDataset
from openpi.training import config as _config

sys.path.insert(0, str(SCRIPT_DIR/ ".." / "py_script"))
from vlm_interfaces import *
from vla_verify.scene_graph import TaskSceneGraph
from vla_verify.verifier import VLAVerifier
PDDL_PATH = SCRIPT_DIR / ".." / "py_script" / "pddl" / "libero_domain.pddl"
pddl_domain_text = open(PDDL_PATH).read()

data_config = _config.get_config('pi05_libero_skill_reason_fixed')
dataset = LiberoSkillReasonDataset(data_config.data.base_config, data_config.model.action_horizon)

llm_interface, vlm_interface = get_openrouter_interfaces()
scene_graph = TaskSceneGraph(pddl_domain_text, vlm_interface)
verifier = VLAVerifier(scene_graph, llm_interface)

out_dir = SCRIPT_DIR / "data"
os.makedirs(str(out_dir), exist_ok=True)

def image_tensor_to_cv2(image, resolution=(512,512)):
    return cv2.resize(np.array(einops.rearrange(image, "c h w -> h w c") * 255, dtype=np.uint8), resolution, interpolation=cv2.INTER_LANCZOS4)

def get_episode(episode_idx):
    reasonings = dataset.reasoning[episode_idx]
    start_idx = dataset.episode_starts[episode_idx]
    end_idx = dataset.episode_ends[episode_idx]
    data = dataset.hf_dataset[int(start_idx)]
    video_frames = []
    for i in range(start_idx, end_idx):
        img_data = dataset.hf_dataset[i]['image']
        video_frames.append(image_tensor_to_cv2(img_data))
    return reasonings, video_frames

def process_episode(episode, episode_idx):
    reasonings, video_frames = episode
    task = reasonings['instruction'].split(':', 1)[-1].strip()
    print(task)
    scene_graph.read_image(video_frames[0], hint=f"The robot is trying to {task}", ground=False)

    targets = []
    for segment in reasonings['segments']:
        skill = segment['skill']
        splits = re.split('([^a-zA-Z0-9]left[^a-zA-Z0-9]|[^a-zA-Z0-9]right[^a-zA-Z0-9])', skill)
        for i in range(1, len(splits), 2):
            if len(splits[i]) == 6:
                splits[i] = splits[i][0] + 'right' + splits[i][-1]
            elif len(splits[i]) == 7:
                splits[i] = splits[i][0] + 'left' + splits[i][-1]
        skill = ''.join(splits)
        skill_res = verifier.verify_skill(skill)
        print(skill_res)
        if not skill_res.accepted:
            print("Verification failed!")
            break
        name, params = verifier._normalize_grounded_action(skill_res.grounded_action)
        pddl_action = scene_graph.match_grounded_action(name, params)
        if pddl_action is None:
            print("Could not match pddl action!")
            break
        target = None
        match pddl_action.name.value:
            case "pickup_from" | "open" | "place_in" | "turn_on" | "turn_off":
                target = pddl_action.grounding[0].value
            case "place_on":
                target = pddl_action.grounding[2].value
                # TODO: table location is bad
        start_step = segment['start_step']
        end_step = segment['end_step']

        if target in scene_graph.object_data:
            target_object = scene_graph.object_data[target]
            target_info = target_object.to_dict(include_grounding=False)
            result = scene_graph.ground_openrouter(video_frames[start_step], target_object)
            if result['status'] == 'OK':
                target_info['image_point'] = result['position']
        else:
            target_info = None
        targets.append(target_info)
        scene_graph.apply_action(pddl_action)

    # NOTE: This may not contain the full trace, if verification fails...
    with gzip.open(f'{out_dir}/{episode_idx}_targets.json.zip', 'wt', encoding="ascii") as zipfile:
        json.dump(targets, zipfile)

for episode_idx in range(len(dataset.episode_starts)):
    print("Processing episode", episode_idx)
    reasonings, video_frames = get_episode(episode_idx)
    process_episode((reasonings, video_frames), episode_idx)
