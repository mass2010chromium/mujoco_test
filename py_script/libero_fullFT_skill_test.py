"""
This script test the full finetuned model to operate on its own entirely. Namely, it
1) output a plan given the prompt at t = 0
2) output a skill given the generated plan at t >= 0
3) output actions given the selected skill at t >= 0
No outside mechanisms interfere with the model on these three levels

NOTE: for testings, we might add some hardcoded plan or prompt, but the above is the overal purpose of this script
"""
import pathlib
import math
import os
import re

# def configure_jax_runtime():
#     # Avoid large upfront GPU preallocation during policy load. On ANVIL this can
#     # fail at backend init even when the GPU later appears idle in nvidia-smi.
#     os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
#     os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
#     os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")


# configure_jax_runtime()

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


def _normalize_skill_text(skill):
    return " ".join(str(skill).split()).strip(" ;")


def _parse_plan_steps(plan):
    text = _normalize_scene_plan_text(plan)
    if not text:
        return []

    numbered_matches = list(re.finditer(r"(?:^|\s)(\d+)\.\s*", text))
    if numbered_matches:
        steps = []
        for i, match in enumerate(numbered_matches):
            start = match.end()
            end = numbered_matches[i + 1].start() if i + 1 < len(numbered_matches) else len(text)
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


def _format_done_skills(done_skills):
    if not done_skills:
        return "Done: None"
    return f"Done: {', '.join(done_skills)}"


def _build_skill_reasoning_prefix(scene_plan, done_skills):
    normalized_plan = _normalize_scene_plan_text(scene_plan)
    return f"Plan: {normalized_plan}; {_format_done_skills(done_skills)}"


def _advance_done_skills(plan_steps, done_skills):
    next_done_index = len(done_skills)
    if next_done_index < len(plan_steps):
        done_skills.append(plan_steps[next_done_index])


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

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 224 # resolution used to render training data
TRIALS_PER_TASK = 1
SEED = 7
EPISODE_LENGTH = 400
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
            # scene_plan = '1. OPEN(top drawer of the cabinet) 2. PICK(black bowl) 3. PLACE(black bowl, top drawer of the cabinet, inside) 4. CLOSE(top drawer of the cabinet)'
            # scene_plan = '1. OPEN(top drawer of the cabinet)'
            # scene_plan = '1. PICK(frying pan) 2. PLACE(frying pan, stove, on top of)'

            # scene_plan = '1. PICK(black bowl) 2. PLACE(black bowl, bottom drawer, inside) 3. CLOSE(bottom drawer)'
            
            # scene_plan = '1. PICK(wine bottle)'
            # scene_plan = '1. GRASP(stove knob)'
            # scene_plan = '1. OPEN(top drawer of the cabinet)'

            scene_plan = '1. CLOSE(bottom drawer) 2. OPEN(top drawer) 3. PICKUP_FROM(black bowl, table) 4. PLACE_IN(black bowl, top drawer)'

            current_skill = ''
            plan_steps = []
            done_skills = []
            
            # first, think once to populate the plan
            if scene_plan == '':
                prompt = prompt_from_obs(obs, task_description, scene_plan=scene_plan, mode='thinking')
                result = policy.infer(prompt)
                infer_count += 1
                subtask = result.get('subtask', None)
                if subtask is not None:
                    scene_plan = _normalize_scene_plan_text(subtask)
                else:
                    raise Exception("No initial thought generated")

            scene_plan = _normalize_scene_plan_text(scene_plan)
            plan_steps = _parse_plan_steps(scene_plan)
            print("parsed plan steps: ", plan_steps)
            
            # second, think again to get current skill
            # prompt = prompt_from_obs(
            #     obs,
            #     task_description,
            #     scene_plan=scene_plan,
            #     done_skills=done_skills,
            #     mode='thinking',
            # )
            # result = policy.infer(prompt)
            # infer_count += 1
            # skill = result.get('subtask', None)
            # if skill is not None:
            #     current_skill = _normalize_skill_text(skill)
            # elif plan_steps:
            #     current_skill = plan_steps[0]
            # else:
            #     raise Exception("No initial thought generated")
            
            # start with the first skill in the plan
            current_skill = plan_steps[0]
            print("plan steps: ", plan_steps, "; starting with current skill: ", current_skill)
            
            # act for the first time
            prompt = prompt_from_obs(obs, task_description, skill=current_skill, mode='acting')
            result = policy.infer(prompt)
            infer_count += 1
            actions = result['actions']

            print(f"[Task {task_id}] Task: {task_description}")
            # print(f"[Task {task_id}] Initial reasoning: {subtask}")

            rollout = []
            frames = []
            wrist_frames = []
            trajectory_idx = 0
            done = False

            for i in range(EPISODE_LENGTH):

                act = np.copy(actions[trajectory_idx])
                obs, reward, done, info = env.step(act)
                rollout.append(obs)
                # frames.append(obs['agentview_image'][::-1, ::-1, :])
                frames.append(write_skill_to_frame(obs, current_skill))
                wrist_frames.append(obs['robot0_eye_in_hand_image'][::-1, ::-1, :])
                trajectory_idx += 1

                if done:
                    break

                if trajectory_idx == (len(actions) // 2):
                    trajectory_idx = 0

                    # when it's time to think, call infer to update plan
                    if infer_count % think_freq == 0:
                        prompt = prompt_from_obs(
                            obs,
                            task_description,
                            scene_plan=scene_plan,
                            done_skills=done_skills,
                            mode='thinking',
                        )
                        result = policy.infer(prompt)
                        infer_count += 1
                        skill = result.get('subtask', None)
                        if skill is not None:
                            next_skill = _normalize_skill_text(skill)
                            if next_skill and next_skill != current_skill:
                                _advance_done_skills(plan_steps, done_skills)
                                print("updated done skills: ", done_skills)
                                current_skill = next_skill
                            elif not current_skill:
                                current_skill = next_skill
                        print(f"[Step {i}] Reasoning skill (thinking): {current_skill}")

                    # current_skill = 'OPEN(bottom drawer)'
                    prompt = prompt_from_obs(obs, task_description, scene_plan=scene_plan, skill=current_skill, mode='acting')
                    result = policy.infer(prompt)
                    infer_count += 1
                    actions = result['actions']
                    # print(f"[Step {i}] Reasoning: {subtask}")


                    mediapy.write_image('trial_imgs/frame_agentview' + str(i) + '.png', np.copy(obs['agentview_image'][::-1, ::-1, :]))
                    mediapy.write_image('trial_imgs/frame_wrist' + str(i) + '.png', np.copy(obs['robot0_eye_in_hand_image'][::-1, ::-1, :]))
                    
                
            if done:
                trials_success.append(1)
            else:
                trials_success.append(0)
            mediapy.write_image('franka_libero_f0.png', frames[0])
            # mediapy.write_video('franka_libero_v3_' + str(task_id) + '.mp4', frames, fps=FRAME_RATE)
            mediapy.write_video(
                'franka_libero_v3_fullFT_' + str(task_id) + '.mp4',
                frames,
                fps=FRAME_RATE,
                codec="mpeg4",
                bps=1_000_000,
            )
            # mediapy.write_video('franka_libero_wrist_v3_' + str(task_id) + '.mp4', wrist_frames, fps=FRAME_RATE)
        #     break
        # break

    print(f"Trials success: {trials_success}")
    print(f"Trials success rate: {sum(trials_success) / len(trials_success)}")
