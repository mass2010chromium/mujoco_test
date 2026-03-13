"""
Adapted from openpi libero eval script
REQUIRES: robosuite 1.4.0

Adapted from libero_loraFT_skill_test_v3.py
"""
import pathlib
import math
import os

def limit_jax_mem(limit):
	os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"
limit_jax_mem(0.6)

import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import mediapy
import tqdm
import cv2
import re

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

from vla_verify.verifier import VLAVerifier
from vla_verify.scene_graph import TaskSceneGraph
from vla_verify.pddl_parsing import setup_pddl_simulation

from vlm_interfaces import *

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def _parse_plan_steps(plan):

    text = plan.strip()
    if not text:
        return []

    numbered_matches = list(re.finditer(r"(?:^|\s)(\d+)\.\s*", text))
    if numbered_matches:
        steps = []
        for i, match in enumerate(numbered_matches):
            start = match.end()
            end = (
                numbered_matches[i + 1].start()
                if i + 1 < len(numbered_matches)
                else len(text)
            )
            step_text = text[start:end].strip(" \n\t;")
            if step_text:
                steps.append(step_text)
        if steps:
            return steps

    return [
        chunk.strip(" \n\t-")
        for chunk in re.split(r"[\n;]+", text)
        if chunk.strip()
    ]


def create_pi05_skill_reasoning(model_name, checkpoint_dir=None, assets_dir=None):
    vla_config = _config.get_config(model_name)
    if checkpoint_dir is None:
        checkpoint_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{model_name}")
    else:
        checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()

    norm_stats = None
    if assets_dir is not None:
        data_config = vla_config.data.create(vla_config.assets_dirs, vla_config.model)
        if data_config.asset_id is not None:
            from openpi.training import checkpoints as _checkpoints

            assets_path = pathlib.Path(assets_dir).resolve()
            norm_stats = _checkpoints.load_norm_stats(assets_path, data_config.asset_id)

    initial_scene_plan = (
        "Plan: TBD.\n"
        " What I have done: TBD.\n"
        "Now I need to do: TBD.\n"
    )
    return _policy_config.create_trained_skill_reasoning_policy(
        vla_config,
        checkpoint_dir,
        norm_stats=norm_stats,
        initial_scene_plan=initial_scene_plan,
        sample_kwargs={
            "temperature": 0.0,
            "max_reasoning_steps": 256,
            "force_initial_reasoning": True,
            "debug_prefill": True,
        },
    )


def prompt_from_obs(obs, task, scene_plan='', skill='', mode='thinking'):
    """Build observation dict for Pi0Fuse inference.

    The thought prefix must match the training format from cot_simple.json:
      "Instruction: <task>\\n<scene_plan>"
    A 1-element thought list triggers action-mode tokenization (BEGIN_OF_ACTION
    suffix), which prefill() strips before deciding to think or act.
    """
    state_vec = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )

    print("current scene plan: ", scene_plan)

    if mode == 'thinking':
        if scene_plan == '':
            # thought_prefix = task
            thought_prefix = 'Instruction: ' + task
        else:
            thought_prefix = scene_plan

        return {
            'observation/image': obs['agentview_image'][::-1, ::-1, :],
            'observation/wrist_image': obs['robot0_eye_in_hand_image'][::-1, ::-1, :],
            "observation/state": state_vec,
            #"state": state_vec,  # required by LiberoReasonInputs
            'prompt': task,
            'thought': [thought_prefix],
            'mode': mode,
        }
    
    else:
        # skill = 'PICK(top black bowl)'
        print(f"Skill: {skill}")
        return {
            'observation/image': obs['agentview_image'][::-1, ::-1, :],
            'observation/wrist_image': obs['robot0_eye_in_hand_image'][::-1, ::-1, :],
            "observation/state": state_vec,
            #"state": state_vec,  # required by LiberoReasonInputs
            'prompt': task,
            'thought': [skill],
            'mode': mode,
        }


def write_skill_to_frame(obs, current_skill):
    """Render current skill text at the top-right of the agentview frame."""
    frame = np.copy(obs['agentview_image'][::-1, ::-1, :])
    skill_text = f"Skill: {current_skill}" if current_skill else "Skill:"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    margin = 8
    (text_w, text_h), baseline = cv2.getTextSize(skill_text, font, font_scale, thickness)

    x = max(margin, frame.shape[1] - text_w - margin)
    y = margin + text_h

    cv2.rectangle(
        frame,
        (max(0, x - 4), max(0, y - text_h - 4)),
        (min(frame.shape[1] - 1, x + text_w + 4), min(frame.shape[0] - 1, y + baseline + 4)),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, skill_text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 224 # resolution used to render training data
TRIALS_PER_TASK = 1
SEED = 7
EPISODE_LENGTH = 800
FRAME_RATE = 20

if __name__ == "__main__":
    PDDL_PATH = SCRIPT_DIR / "pddl" / "libero_domain.pddl"
    IMAGE_PATH = SCRIPT_DIR / "initial_scene.png"                    # initial scene image

    llm_interface, vlm_interface = get_openrouter_interfaces()

    benchmark_dict = benchmark.get_benchmark_dict()
    # task_suite = benchmark_dict['libero_spatial']()
    # task_suite = benchmark_dict['libero_goal']()
    # task_suite = benchmark_dict['libero_90']()
    task_suite = benchmark_dict['libero_10']()
    num_tasks_in_suite = task_suite.n_tasks

    # Point to openpi assets (where compute_norm_stats.py wrote them)
    openpi_root = pathlib.Path(__file__).resolve().parent.parent / "pace" / "openpi"
    assets_dir = openpi_root / "assets" / "pi05_libero_skill_reason_lora_v2"

    print("Loading pi05 skill reasoning policy...", end="", flush=True)
    policy = create_pi05_skill_reasoning(
        "pi05_libero_skill_reason_lora_v2",
        checkpoint_dir=(
            pathlib.Path(__file__).resolve().parent.parent
            / "pace"
            / "openpi"
            / "checkpoints"
            / "pi05_libero_skill_reason_lora_v2"
            / "pi05_libero_skill_reason_lora_v3"
            / "30000"
        ),
        assets_dir=assets_dir,
    )
    print("Done.")
    trials_success = []

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        if task_id != 3:       # test task 10 --> drawer closed initially
            continue

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, SEED)

        # task_description = "put the black bowl inside the drawer"

        infer_count = 0
        think_freq = 4
        for episode_idx in tqdm.tqdm(range(TRIALS_PER_TASK)):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            if hasattr(policy, "start"):
                policy.start()
            
            # wait for the env to settle
            for _ in range(20):
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            
            # create initial scene graph and verifier
            print("Creating initial scene graph...")
            mediapy.write_image(IMAGE_PATH, obs['agentview_image'][::-1, ::-1, :])
            pddl_domain_text = open(PDDL_PATH).read()
            scene_graph = TaskSceneGraph(pddl_domain_text, vlm_interface)
            scene_graph.read_image(cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB), ground=True)
            verifier = VLAVerifier(scene_graph, llm_interface, vlm_interface)

            scene_plan = ''

            # hardcode scene plan if needed
            scene_plan = '1. OPEN(top drawer) 2. PICK(black bowl) 3. PLACE(black bowl, top drawer, inside) 4. CLOSE(top drawer)'

            current_skill = ''
            
            # first, think once to populate the plan
            if scene_plan == '':
                prompt = prompt_from_obs(obs, task_description, scene_plan=scene_plan, mode='thinking')
                result = policy.infer(prompt)
                infer_count += 1
                subtask = result.get('subtask', None)
                if subtask is not None:
                    scene_plan = subtask
                else:
                    raise Exception("No initial thought generated")
            
            # ============================ VERIFY THE SCENE PLAN ============================
            result = verifier.verify_skill_plan(scene_plan)
            if not result["feasible"]:
                print("Scene plan result: INFEASIBLE")
                print("Failure reason: ", result["failure_reason"])
                #TODO figure out how to steer the plan
            else:
                print("Scene plan result: FEASIBLE")
                verifier.set_skill_plan(scene_plan)
            plan_steps = _parse_plan_steps(scene_plan)

            # start with the first skill in the plan
            current_skill = plan_steps[0]
            print("plan steps: ", plan_steps, "; starting with current skill: ", current_skill)
            verifier.verify_skill_transition(current_skill)
            
            # act for the first time
            prompt = prompt_from_obs(obs, task_description, skill=current_skill, mode='acting')
            result = policy.infer(prompt)
            infer_count += 1
            actions = result['actions']

            print(f"[Task {task_id}] Task: {task_description}")

            rollout = []
            frames = []
            raw_frames = []
            wrist_frames = []
            trajectory_idx = 0
            done = False

            for i in range(EPISODE_LENGTH):

                act = np.copy(actions[trajectory_idx])
                obs, reward, done, info = env.step(act)
                rollout.append(obs)
                frames.append(write_skill_to_frame(obs, current_skill))
                raw_frames.append(obs['agentview_image'][::-1, ::-1, :])
                wrist_frames.append(obs['robot0_eye_in_hand_image'][::-1, ::-1, :])
                trajectory_idx += 1

                if done:
                    break

                if trajectory_idx == (len(actions) // 2):
                    trajectory_idx = 0

                    # when it's time to think, call infer to update plan
                    if infer_count % think_freq == 0:
                        prompt = prompt_from_obs(obs, task_description, scene_plan=scene_plan, mode='thinking')
                        result = policy.infer(prompt)
                        infer_count += 1
                        skill = result.get('subtask', None)

                        # ============================ VERIFY THE SKILL TRANSITION ============================
                        if current_skill != skill:
                            result = verifier.verify_skill_transition(
                                next_skill=skill,
                                image_rgb=obs['agentview_image'][::-1, ::-1, :],
                            )
                            if not result["feasible"]:
                                print(f"skill transition infeasible at step {i}, failure reason: {result['failure_reason']}")
                            else:
                                current_skill = skill
                                print("skill transition feasible")
                        print(f"[Step {i}] Reasoning skill (thinking): {current_skill}")

                    prompt = prompt_from_obs(obs, task_description, scene_plan=scene_plan, skill=current_skill, mode='acting')
                    result = policy.infer(prompt)
                    infer_count += 1
                    actions = result['actions']

                    mediapy.write_image('trial_imgs/frame_agentview' + str(i) + '.png', write_skill_to_frame(obs, current_skill))
                    
                
            if done:
                trials_success.append(1)
            else:
                trials_success.append(0)
            mediapy.write_image('franka_libero_f0.png', raw_frames[0])
            mediapy.write_video('franka_libero_v3_' + str(task_id) + '.mp4', frames, fps=FRAME_RATE)
        #     break
        # break

    print(f"Trials success: {trials_success}")
    print(f"Trials success rate: {sum(trials_success) / len(trials_success)}")