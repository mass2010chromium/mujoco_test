import functools

from openpi.policies import policy_config as _policy_config

from inference_common import quat2axisangle, register_model

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
            quat2axisangle(obs["robot0_eef_quat"]),
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
def create_pi05(model_name, config, checkpoint_dir, norm_stats):
    # TODO: is this used?
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
    policy.initialize = functools.partial(skill_init, policy)
    policy.run_vla = functools.partial(skill_infer, policy)
    return policy

register_model(create_pi05, "pi05_libero_skill_reason_fixed", 29999)
