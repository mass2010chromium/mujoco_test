import functools
import json
import math
import os
import pathlib
import re
import sys
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.30"

import numpy as np
import mediapy as media

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import openpi
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

sys.path.append('../py_script')

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
def skill_prompt(obs, task, scene_plan='', skill='', mode='thinking'):
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
            thought_prefix = task
        else:
            thought_prefix = scene_plan
        text_input = thought_prefix

    else:
        print(f"Skill: {skill}")
        text_input = skill

    return {
        'observation/image': obs['agentview_image'][::-1, ::-1, :],
        'observation/wrist_image': obs['robot0_eye_in_hand_image'][::-1, ::-1, :],
        "observation/state": state_vec,
        'prompt': task,
        'thought': [text_input],
        'mode': mode,
    }
def invoke_skill_policy(policy, obs, *args, **kwargs):
    prompt = skill_prompt(obs, *args, **kwargs)
    vla_output = policy.infer(prompt)
    return vla_output

def skill_init(policy, obs, task):
    policy.start()
    plan_output = invoke_skill_policy(policy, obs, "Instruction: " + task, mode="thinking")
    scene_plan = plan_output['subtask']
    policy.task = task
    policy.plan = scene_plan

def skill_infer(policy, obs):
    skill_output = invoke_skill_policy(policy, obs, policy.plan, mode="thinking")
    skill = skill_output['subtask']
    act_output = invoke_skill_policy(policy, obs, '', skill=skill, mode="acting")
    return act_output

# Load pi model
def create_pi05_skill(model_name, checkpoint_dir=None, assets_dir=None):
    vla_config = _config.get_config(model_name)
    if checkpoint_dir is None:
        checkpoint_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{model_name}")
    else:
        checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()

    # Load norm stats from assets dir if checkpoint doesn't have them
    norm_stats = None
    if assets_dir is not None:
        data_config = vla_config.data.create(vla_config.assets_dirs, vla_config.model)
        if data_config.asset_id is not None:
            from openpi.training import checkpoints as _checkpoints
            assets_path = pathlib.Path(assets_dir).resolve()
            norm_stats = _checkpoints.load_norm_stats(assets_path, data_config.asset_id)

    initial_scene_plan = {
        "Plan: TBD.\n"
        " What I have done: TBD.\n"  # Extra space here follows the original dataset formatting...
        "Now I need to do: TBD.\n"
    }
    policy = _policy_config.create_trained_skill_reasoning_policy(
        vla_config,
        checkpoint_dir,
        norm_stats=norm_stats,
        initial_scene_plan=initial_scene_plan,
        sample_kwargs={
            "temperature": 0.0,
            "max_reasoning_steps": 256,
            "force_initial_reasoning": True
        }
    )
    policy.initialize = functools.partial(skill_init, policy)
    policy.run_vla = functools.partial(skill_infer, policy)
    return policy

# EE control
def vanilla_prompt(obs, task): 
    qpos = np.array(obs['robot0_joint_pos'])
    gripper_pos = np.array([obs['robot0_gripper_qpos'][0]])
    return {
        # Flip the ego camera?
        'observation/image': obs['agentview_image'][::-1, ::-1, :],
        'observation/wrist_image': obs['robot0_eye_in_hand_image'][::-1, ::-1, :],
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        'prompt': task
    }

def vanilla_init(policy, obs, task):
    policy.task = task

def vanilla_infer(policy, obs):
    prompt = vanilla_prompt(obs, policy.task)
    return policy.infer(prompt)

def create_pi05(model_name):
    vla_config = _config.get_config(model_name)
    checkpoint_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{model_name}")
    policy = _policy_config.create_trained_policy(vla_config, checkpoint_dir)
    policy.initialize = functools.partial(vanilla_init, policy)
    policy.run_vla = functools.partial(vanilla_infer, policy)
    return policy


def _normalize_scene_plan_text(scene_plan):
    text = " ".join(str(scene_plan).split()).strip()
    text = re.sub(r"^\s*Plan:\s*", "", text)
    text = re.split(r"\s*;\s*Done:\s*", text, maxsplit=1)[0].strip(" ;")
    return text

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

def skill2_prompt(obs, task, scene_plan='', done_skills=None, skill='', mode='thinking'):
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

    def _format_done_skills(done_skills):
        if not done_skills:
            return "Done: None"
        return f"Done: {', '.join(done_skills)}"
    def _build_skill_reasoning_prefix(scene_plan, done_skills):
        normalized_plan = _normalize_scene_plan_text(scene_plan)
        return f"Plan: {normalized_plan}; {_format_done_skills(done_skills)}"

    print("current scene plan: ", scene_plan)
    done_skills = [] if done_skills is None else done_skills

    if mode == 'thinking':
        if scene_plan == '':
            thought_prefix = 'Instruction: ' + task
        else:
            thought_prefix = _build_skill_reasoning_prefix(scene_plan, done_skills)

        print("current done skills: ", done_skills)
        thought = [thought_prefix]
    
    else:
        print(f"Skill: {skill}")
        thought = [skill]
    return {
        'observation/image': obs['agentview_image'][::-1, ::-1, :],
        'observation/wrist_image': obs['robot0_eye_in_hand_image'][::-1, ::-1, :],
        "observation/state": state_vec,
        'prompt': task,
        'thought': thought,
        'mode': mode,
    }
def invoke_skill2_policy(policy, obs, *args, **kwargs):
    prompt = skill2_prompt(obs, *args, **kwargs)
    vla_output = policy.infer(prompt)
    return vla_output

def skill2_init(policy, obs, task):
    if hasattr(policy, "start"):
        print("policy has 'start'")
        policy.start()
    else:
        print("policy has no 'start'")
    plan_output = invoke_skill2_policy(policy, obs, task, scene_plan='', mode="thinking")
    scene_plan = _normalize_scene_plan_text(plan_output['subtask'])
    policy.task = task
    policy.plan_steps = _parse_plan_steps(scene_plan)
    policy.done_skills = []
    policy.plan_text = scene_plan
    policy.current_skill = policy.plan_steps[0]
    policy.infer_count = 0
    policy.think_freq = 4

def _normalize_skill_text(skill):
    return " ".join(str(skill).split()).strip(" ;")

def skill2_infer(policy, obs):
    policy.infer_count += 1
    if policy.infer_count == policy.think_freq:
        policy.infer_count = 0
        skill_output = invoke_skill2_policy(policy, obs, policy.task,
            scene_plan=policy.plan_text,
            done_skills=policy.done_skills,
            mode="thinking"
        )
        skill = skill_output['subtask']
        next_skill = _normalize_skill_text(skill)
        if next_skill and next_skill != policy.current_skill:
            # Advance done skills
            if len(policy.done_skills) < len(policy.plan_steps):
                policy.done_skills.append(policy.plan_steps[len(policy.done_skills)])
            policy.current_skill = next_skill
    act_output = invoke_skill_policy(policy, obs, policy.task,
        scene_plan=policy.plan_text,
        skill=policy.current_skill,
        mode="acting"
    )
    return act_output

def create_pi05_skill2(model_name, checkpoint_dir=None, assets_dir=None):
    vla_config = _config.get_config(model_name)
    if checkpoint_dir is None:
        checkpoint_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{model_name}")
    else:
        checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()

    # Load norm stats from assets dir if checkpoint doesn't have them
    norm_stats = None
    if assets_dir is not None:
        data_config = vla_config.data.create(vla_config.assets_dirs, vla_config.model)
        if data_config.asset_id is not None:
            from openpi.training import checkpoints as _checkpoints
            assets_path = pathlib.Path(assets_dir).resolve()
            norm_stats = _checkpoints.load_norm_stats(assets_path, data_config.asset_id)

    initial_scene_plan = {
        "Plan: TBD.\n"
        " What I have done: TBD.\n"  # Extra space here follows the original dataset formatting...
        "Now I need to do: TBD.\n"
    }
    policy = _policy_config.create_trained_skill_reasoning_policy(
        vla_config,
        checkpoint_dir,
        norm_stats=norm_stats,
        initial_scene_plan=initial_scene_plan,
        sample_kwargs={
            "temperature": 0.0,
            "max_reasoning_steps": 256,
            "force_initial_reasoning": True
        }
    )
    policy.initialize = functools.partial(skill2_init, policy)
    policy.run_vla = functools.partial(skill2_infer, policy)
    return policy


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution, "camera_depths": True}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

class LiberoEnvMaker:
    def __init__(self, suite: str,
                 render_resolution: int = 512, seed: int = 0,
                 repeats: int = 1):
        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[suite]()
        self.repeats = repeats
        self.render_resolution = render_resolution
        self.seed = seed

    def get_num_tasks(self):
        return self.task_suite.n_tasks

    def task_instantiations(self, task_id):
        task = self.task_suite.get_task(task_id)
        print(task, task_id)
        initial_states = self.task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, self.render_resolution, self.seed)
        for episode_idx in range(self.repeats):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            yield obs, env, task_description

def run_benchmark(name, policy, setting, repeats=2):
    libero_envs = LiberoEnvMaker(setting, repeats=repeats)
    # STARTUP section
    MAX_STEPS = 600
    count = 0
    success = 0
    out_dir = f'benchmark/{name}'
    os.makedirs(out_dir, exist_ok=True)
    result_summary = {}
    for task_num in range(libero_envs.get_num_tasks()):
        stats = []
        for j, instance in enumerate(libero_envs.task_instantiations(task_num)):
            rollout = []
            frames = []
            wrist_frames = []
            obs, env, task_description = instance
            LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
            for _ in range(20):
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            policy.initialize(obs, task_description)
            rollout.append(obs)
            frames.append(obs['agentview_image'][::-1, ::-1, :])
            wrist_frames.append(obs['robot0_eye_in_hand_image'][::-1, ::-1, :])
            print(task_description)
            vla_output = policy.run_vla(obs)
            actions = vla_output['actions']
    
            trajectory_idx = 0
            count += 1
            for i in range(MAX_STEPS):
                act = np.copy(actions[trajectory_idx])
                obs, reward, done, info = env.step(act)
                frames.append(obs['agentview_image'][::-1, ::-1, :])
                wrist_frames.append(obs['robot0_eye_in_hand_image'][::-1, ::-1, :])
                if done:
                    break
                trajectory_idx += 1
                if trajectory_idx == (len(actions)//2):
                    vla_output = policy.run_vla(obs)
                    actions = vla_output['actions']
                    trajectory_idx = 0
            print(f"TASK {task_num}: STATUS={done} after {i} steps")
            if done:
                success += 1
            freq = 20
            media.write_video(f'{out_dir}/{task_num}.{j}.mp4', frames, fps=freq)
            media.write_video(f'{out_dir}/{task_num}.{j}_wrist.mp4', wrist_frames, fps=freq)
            stats.append({
                "done": bool(done),
                "length": i
            })
        result_summary[task_description] = stats
    print(f"Overall: {success}/{count}")
    with open(f'{out_dir}/summary.json', 'w') as outfile:
        json.dump(result_summary, outfile)
        
    for task, stats in result_summary.items():
        count = len(stats)
        success = 0
        for stat in stats:
            if stat['done']:
                success += 1
        print(f"{task}: {success}/{count}")
    return result_summary


# # model_name = 'pi05_libero_skill_reason_lora_v2'
# model_name = 'pi05_libero_skill_reason_fixed'
# # Create a trained policy.
# policy = create_pi05_skill(   # Use LoRA weights:
#     model_name,
#     # checkpoint_dir='../pace/openpi/checkpoints/pi05_libero_reason_lora/pi05_libero_reason_lora/7999',
#     checkpoint_dir=f'../pace/openpi/checkpoints/{model_name}/{model_name}/29999',
#     assets_dir=f'../pace/openpi/assets/{model_name}'
#     # 'pi05_libero_10_reason_lora',
#     # checkpoint_dir='../pace/openpi/checkpoints/pi05_libero_10_reason_lora/pi05_libero_10_reason_lora/1500',
#     # assets_dir='../pace/openpi/assets/pi05_libero_10_reason_lora'
# )
# run_benchmark("pi05_skill", policy, "libero_10", repeats=10)

# policy = create_pi05("pi05_libero")
# run_benchmark("pi05", policy, "libero_10", repeats=10)

model_name = 'pi05_libero_skill_reason_full_finetune'
policy = create_pi05_skill2(   # Use LoRA weights:
    model_name,
    # checkpoint_dir='../pace/openpi/checkpoints/pi05_libero_reason_lora/pi05_libero_reason_lora/7999',
    checkpoint_dir=f'../pace/openpi/checkpoints/{model_name}/{model_name}/75000',
    assets_dir=f'../pace/openpi/assets/{model_name}'
    # 'pi05_libero_10_reason_lora',
    # checkpoint_dir='../pace/openpi/checkpoints/pi05_libero_10_reason_lora/pi05_libero_10_reason_lora/1500',
    # assets_dir='../pace/openpi/assets/pi05_libero_10_reason_lora'
)
run_benchmark("pi05_skill2", policy, "libero_10", repeats=10)
