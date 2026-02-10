"""
Adapted from: https://robosuite.ai/docs/modules/environments.html
REQUIRES: robosuite 1.5.2
"""
import os

import numpy as np
import mediapy as media

import robosuite
from robosuite.controllers import load_composite_controller_config

from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

# Joint control
#controller_file = os.path.join(os.path.dirname(__file__), "panda_joint_controller.json")
#controller_config = load_composite_controller_config(controller=controller_file)
# Load pi model
#vla_config = _config.get_config("pi05_droid")
#checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
#def prompt_from_obs(obs):
#    qpos = np.array(obs['robot0_joint_pos'])
#    gripper_pos = np.array(obs['robot0_gripper_qpos'][0])
#    return {
#        # Flip the ego camera?
#        'observation/exterior_image_1_left': obs['agentview_image'][::-1, ::-1, :],
#        'observation/wrist_image_left': np.zeros((224, 224, 3), dtype=np.uint8),
#        'observation/joint_position': qpos,
#        'observation/gripper_position': gripper_pos,
#        'prompt': 'Grab and pick up the red cube'
#    }

# Pose control
controller_config = load_composite_controller_config(controller="BASIC")
# Load pi model
vla_config = _config.get_config("pi05_libero")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
def prompt_from_obs(obs):
    qpos = np.array(obs['robot0_joint_pos'])
    gripper_pos = np.array([obs['robot0_gripper_qpos'][0]])
    return {
        # Flip the ego camera?
        'observation/image': obs['agentview_image'][::-1, ::-1, :],
        'observation/wrist_image': obs['robot0_robotview_image'][::-1, ::-1, :],
        'observation/state': np.concatenate((qpos, gripper_pos)),
        'prompt': 'Grab and pick up the red cube'
    }


freq = 20
episode_length=800
env = robosuite.make(
    "Stack",
    robots=["Panda"],
    gripper_types="default",
    controller_configs=controller_config,
    env_configuration="opposed",    # What are the options?
    has_renderer=False,
    #render_camera="frontview",
    has_offscreen_renderer=True,
    control_freq=freq,
    horizon=episode_length,
    use_object_obs=False,
    use_camera_obs=True,
    camera_names=["agentview", "robot0_robotview"],
    camera_heights=224,
    camera_widths=224,
)

# Create a trained policy.
policy = _policy_config.create_trained_policy(vla_config, checkpoint_dir)


rollout = []
frames = []
wrist_frames = []
act_scale = 0.5
for _ in range(1):
    obs = env.reset()
    prompt = prompt_from_obs(obs)
    actions = policy.infer(prompt)['actions']
    rollout.append(obs)

    trajectory_idx = 0
    for i in range(episode_length):
        act = np.copy(actions[trajectory_idx])
        act[-1] *= 1
        obs, reward, done, info = env.step(act * act_scale)
        rollout.append(obs)
        frames.append(obs['agentview_image'][::-1, ::-1, :])
        wrist_frames.append(obs['robot0_robotview_image'][::-1, ::-1, :])
        trajectory_idx += 1
        if trajectory_idx == (len(actions)//2):
            prompt = prompt_from_obs(obs)
            actions = policy.infer(prompt)['actions']
            trajectory_idx = 0

media.write_video('franka_stacking.mp4', frames, fps=freq)
media.write_video('franka_stacking_wrist.mp4', wrist_frames, fps=freq)
