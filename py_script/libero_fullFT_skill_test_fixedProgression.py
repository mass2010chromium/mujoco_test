"""
This script tests the full finetuned model with a fixed skill progression schedule. Namely, it
1. provides a hardcoded skill plan
2. provides a hardcoded skill schedule where each skill executes a fixed number of steps

The purpose is to test the model's capability under the assumption that plan generation and skill selections are correct.

NOTE: for testings, we might add some hardcoded plan or prompt, but the above is the overal purpose of this script
"""
import pathlib
import math
import os
import re

def limit_jax_mem(limit):
	os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"
limit_jax_mem(0.6)

import numpy as np
import mediapy
import tqdm
import cv2

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

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


def _normalize_scene_plan_text(scene_plan):
    text = " ".join(str(scene_plan).split()).strip()
    text = re.sub(r"^\s*Plan:\s*", "", text)
    text = re.split(r"\s*;\s*Done:\s*", text, maxsplit=1)[0].strip(" ;")
    return text


def _format_done_skills(done_skills):
    if not done_skills:
        return "Done: None"
    return f"Done: {', '.join(done_skills)}"


def _build_skill_reasoning_prefix(scene_plan, done_skills):
    normalized_plan = _normalize_scene_plan_text(scene_plan)
    return f"Plan: {normalized_plan}; {_format_done_skills(done_skills)}"


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


def prompt_from_obs(obs, task, scene_plan='', done_skills=None, skill='', mode='thinking'):
    """Build observation dict for Pi0Fuse inference.

    The skill-selection prompt must match LiberoSkillReasonDataset training:
      "Plan: <scene_plan>; Done: <done skills>"
    """
    state_vec = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )

    print("current scene plan: ", scene_plan)
    done_skills = [] if done_skills is None else done_skills

    if mode == 'thinking':
        if scene_plan == '':
            thought_prefix = 'Instruction: ' + task
        else:
            thought_prefix = _build_skill_reasoning_prefix(scene_plan, done_skills)

        print("current done skills: ", done_skills)

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


def write_skill_to_frame(obs, current_skill, t=0):
    """Render current skill text at the top-right of the agentview frame."""
    frame = np.copy(obs['agentview_image'][::-1, ::-1, :])
    skill_text = f"step: {t}; Skill: {current_skill}" if current_skill else "Skill:"

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
EPISODE_LENGTH = 1000
FRAME_RATE = 20

if __name__ == "__main__":
    benchmark_dict = benchmark.get_benchmark_dict()
    # task_suite = benchmark_dict['libero_spatial']()
    # task_suite = benchmark_dict['libero_goal']()
    # task_suite = benchmark_dict['libero_90']()
    task_suite = benchmark_dict['libero_10']()
    num_tasks_in_suite = task_suite.n_tasks

    # Point to openpi assets (where compute_norm_stats.py wrote them)
    openpi_root = pathlib.Path(__file__).resolve().parent.parent / "pace" / "openpi"
    assets_dir = openpi_root / "assets" / "pi05_libero_skill_reason_full_finetune"

    print("Loading full finetuned pi05 skill reasoning policy...", end="", flush=True)
    policy = create_pi05_skill_reasoning(
        "pi05_libero_skill_reason_full_finetune",
        checkpoint_dir=(
            pathlib.Path(__file__).resolve().parent.parent
            / "checkpoints"
            / "pi05_libero_skill_reason_full_finetune"
            / "pi05_libero_skill_reason_full_finetune"
            / "75000"
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

            scene_plan = ''

            # hardcode scene plan if needed
            # scene_plan = '1. OPEN(top drawer) 2. PICK(black bowl) 3. PLACE(black bowl, top drawer, inside) 4. CLOSE(top drawer)'
            # scene_plan = '1. OPEN(top drawer of the cabinet)'
            # scene_plan = '1. PICK(frying pan) 2. PLACE(frying pan, stove, on top of)'
            # scene_plan = '1. OPEN(middle drawer) 2. PICK(black bowl) 3. PLACE(black bowl, middle drawer, inside) 4. CLOSE(middle drawer)'

            # scene_plan = '1. PICK(black bowl) 2. PLACE(black bowl, bottom drawer, inside) 3. CLOSE(bottom drawer)'
            
            # scene_plan = '1. PICK(wine bottle)'
            # scene_plan = '1. GRASP(stove knob)'
            # scene_plan = '1. OPEN(top drawer of the cabinet)'

            # scene_plan = '1. OPEN(top drawer) 2. PICK(black bowl) 3. PLACE(black bowl, top drawer, inside) 4. CLOSE(top drawer)'
            # scene_plan = '1. OPEN(top drawer)'
            # scene_plan = '1. PICK(wine bottle) 2. PLACE(wine bottle, wine_rack, on top of) 3. PICK(black bowl) 4. PLACE(black bowl, top drawer, inside) 5. CLOSE(bottom drawer)'

            # scene_plan = '1. OPEN(top drawer) 2. PICKUP_FROM(black bowl, table) 3. PLACE_IN(black bowl, top drawer) 4. CLOSE(top drawer)'
            scene_plan = '1. CLOSE(bottom drawer) 2. OPEN(top drawer) 3. PICKUP_FROM(black bowl, table) 4. PLACE_IN(black bowl, top drawer)'

            skill_list = ['CLOSE(bottom drawer)', 
                            'OPEN(top drawer)', 
                            'PICKUP_FROM(black bowl, table)', 
                            'PLACE_IN(black bowl, top drawer)', 
                            'CLOSE(top drawer)',
                            ]
            skill_steps = [
                100,
                150,
                400,
                150,
                200
            ]
            cur_skill_idx = 0
            total_steps_per_skill = 200
            cur_skill_steps = 0
            

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
            
            # second, think again to get current skill
            prompt = prompt_from_obs(obs, task_description, scene_plan=scene_plan, mode='thinking')
            result = policy.infer(prompt)
            infer_count += 1
            skill = result.get('subtask', None)
            if skill is not None:
                current_skill = skill
            else:
                raise Exception("No initial thought generated")
            
            # act for the first time
            current_skill_hard = skill_list[cur_skill_idx]
            current_skill = current_skill_hard
            prompt = prompt_from_obs(obs, task_description, skill=current_skill, mode='acting')
            result = policy.infer(prompt)
            infer_count += 1
            actions = result['actions']

            print(f"[Task {task_id}] Task: {task_description}")
            # print(f"[Task {task_id}] Initial reasoning: {subtask}")

            rollout = []
            frames = []
            raw_frames = []
            wrist_frames = []
            trajectory_idx = 0
            done = False

            for i in range(EPISODE_LENGTH):
                cur_skill_steps += 1

                # hardcode skill progression
                # if i != 0 and i % total_steps_per_skill == 0:
                if cur_skill_steps >= skill_steps[cur_skill_idx]:
                    cur_skill_idx += 1
                    cur_skill_steps = 0
                
                if cur_skill_idx >= len(skill_list):
                    cur_skill_idx = len(skill_list) - 1
                current_skill_hard = skill_list[cur_skill_idx]

                act = np.copy(actions[trajectory_idx])
                obs, reward, done, info = env.step(act)
                rollout.append(obs)
                frames.append(write_skill_to_frame(obs, current_skill, t=i))
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
                        if skill is not None:
                            current_skill = skill
                        print(f"[Step {i}] Reasoning skill (thinking): {current_skill}")

                    # current_skill = 'OPEN(bottom drawer)'
                    current_skill = current_skill_hard
                    print("current skill hard: ", current_skill)
                    prompt = prompt_from_obs(obs, task_description, scene_plan=scene_plan, skill=current_skill, mode='acting')
                    result = policy.infer(prompt)
                    infer_count += 1
                    actions = result['actions']
                    # print(f"[Step {i}] Reasoning: {subtask}")

                    mediapy.write_image('trial_imgs/frame_agentview' + str(i) + '.png', write_skill_to_frame(obs, current_skill, t=i))
                    # mediapy.write_image('trial_imgs/frame_wrist' + str(i) + '.png', np.copy(obs['robot0_eye_in_hand_image'][::-1, ::-1, :]))
                    
                
            if done:
                trials_success.append(1)
            else:
                trials_success.append(0)
            mediapy.write_image('franka_libero_f0.png', raw_frames[0])
            mediapy.write_video(
                'franka_libero_fixed_' + str(task_id) + '.mp4',
                frames,
                fps=FRAME_RATE,
                codec="mpeg4",
                bps=1_000_000,
            )
            # mediapy.write_video('franka_libero_wrist_fixed_' + str(task_id) + '.mp4', wrist_frames, fps=FRAME_RATE)
        #     break
        # break

    print(f"Trials success: {trials_success}")
    print(f"Trials success rate: {sum(trials_success) / len(trials_success)}")
