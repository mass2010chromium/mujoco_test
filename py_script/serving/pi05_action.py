import functools

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

from inference_common import quat2axisangle, register_base_model, register_model

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
                quat2axisangle(obs["robot0_eef_quat"]),
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

def create_pi05_default(model_name, *args):
    vla_config = _config.get_config(model_name)
    checkpoint_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{model_name}")
    policy = _policy_config.create_trained_policy(vla_config, checkpoint_dir)
    policy.initialize = functools.partial(vanilla_init, policy)
    policy.run_vla = functools.partial(vanilla_infer, policy)
    return policy


def real_infer(policy, obs):
    prompt = {
        'observation/image': obs['image'],
        'observation/wrist_image': obs['wrist_image'],
        'observation/state': obs['state'],
        'prompt': policy.task
    }
    return policy.infer(prompt)

def create_pi05_real(model_name, config, checkpoint_dir, norm_stats):
    # TODO: is this used?
    policy = _policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        norm_stats=norm_stats
    )
    policy.initialize = functools.partial(vanilla_init, policy)
    policy.run_vla = functools.partial(real_infer, policy)
    return policy

register_base_model(create_pi05_default, "pi05_libero")
register_model(create_pi05_real, "pi05_real_lora", 2000)
register_model(create_pi05_real, "pi05_real_full_ft", 29999, _checkpoint_dir="/work/hdd/bgtb/zhong2/checkpoints/pi05_table_tasks/pi05_table_tasks/29999")
