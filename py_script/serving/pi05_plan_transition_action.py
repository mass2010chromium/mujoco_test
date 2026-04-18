import functools
import re

import numpy as np

from openpi.policies import policy_config as _policy_config

from inference_common import quat2axisangle, register_model

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
            quat2axisangle(obs["robot0_eef_quat"]),
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

def create_pi05(model_name, config, checkpoint_dir, norm_stats):
    initial_scene_plan = {
        "Plan: TBD.\n"
        " What I have done: TBD.\n"  # Extra space here follows the original dataset formatting...
        "Now I need to do: TBD.\n"
    }
    policy = _policy_config.create_trained_skill_reasoning_policy(
        config,
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

register_model(create_pi05, "pi05_libero_skill_reason_full_finetune", 75000)
