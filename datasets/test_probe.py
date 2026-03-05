import json
import math
import os
from pathlib import Path
import sys
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".25"

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import mediapy as media
from scipy.spatial.transform import Rotation, RigidTransform

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

from probe_network import ProbeNetwork, LinearProbeNetwork

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
    
# Load pi model
def create_pi05(model_name, checkpoint_dir=None, assets_dir=None):
    vla_config = _config.get_config(model_name)
    if checkpoint_dir is None:
        checkpoint_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{model_name}")
    else:
        checkpoint_dir = Path(checkpoint_dir).resolve()

    # Load norm stats from assets dir if checkpoint doesn't have them
    norm_stats = None
    if assets_dir is not None:
        data_config = vla_config.data.create(vla_config.assets_dirs, vla_config.model)
        if data_config.asset_id is not None:
            from openpi.training import checkpoints as _checkpoints
            assets_path = Path(assets_dir).resolve()
            norm_stats = _checkpoints.load_norm_stats(assets_path, data_config.asset_id)

    initial_scene_plan = {
        "Plan: TBD.\n"
        " What I have done: TBD.\n"  # Extra space here follows the original dataset formatting...
        "Now I need to do: TBD.\n"
    }
    policy = _policy_config.create_trained_reasoning_policy(
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
    return policy

def prompt_from_obs(obs, task, scene_plan=''):
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
    return {
        'observation/image': obs['agentview_image'][::-1, ::-1, :],
        'observation/wrist_image': obs['robot0_eye_in_hand_image'][::-1, ::-1, :],
        "observation/state": state_vec,
        'prompt': task,
    }

def infer_until_action(policy, prompt, max_think_rounds=8):
    """Query the policy until it returns actions (not a thinking response)."""
    result = policy.infer(prompt)
    think_round = 0
    while result.get("isthinking", False):
        think_round += 1
        print(f"[Thinking {think_round}] {result.get('thought', '')}")
        if think_round >= max_think_rounds:
            raise RuntimeError(
                f"Policy is still thinking after {max_think_rounds} rounds; "
                "aborting this rollout."
            )
        result = policy.infer(prompt)
    return result

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
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
        initial_states = self.task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, self.render_resolution, self.seed)
        for episode_idx in range(self.repeats):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            yield obs, env, task_description

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="dataset name (ex. libero_10)")
args = parser.parse_args()

DATASET = args.dataset
N_REPEATS = 1
SEED = 0
DATASET = args.dataset
libero_envs = LiberoEnvMaker(DATASET, render_resolution=224, seed=SEED, repeats=N_REPEATS)

# Read targets generated by libero_get_targets.py
targets_dir = Path(__file__).parent / 'libero_targets'
targets_file = targets_dir / f"{DATASET}.json"
targets_data = json.load(open(targets_file, 'r'))
targets_map = {x['task_name']: x for x in targets_data.values()}    # Remap to task names


# Create a trained policy.
policy = create_pi05(   # Use LoRA weights:
    'pi05_libero_reason_lora',
    # checkpoint_dir='../pace/openpi/checkpoints/pi05_libero_reason_lora/pi05_libero_reason_lora/7999',
    checkpoint_dir='../pace/openpi/checkpoints/pi05_libero_reason_lora/pi05_libero_reason_lora/31799',
    assets_dir='../pace/openpi/assets/pi05_libero_reason_lora'
    # 'pi05_libero_10_reason_lora',
    # checkpoint_dir='../pace/openpi/checkpoints/pi05_libero_10_reason_lora/pi05_libero_10_reason_lora/1500',
    # assets_dir='../pace/openpi/assets/pi05_libero_10_reason_lora'
)

import orbax.checkpoint as ocp
checkpointer = ocp.StandardCheckpointer()
checkpoint_dir = os.path.abspath('checkpoints/state')
probe = ProbeNetwork(nnx.Rngs(0))
graphdef, abstract_state = nnx.split(probe)
state_restored = checkpointer.restore(checkpoint_dir, abstract_state)
probe = nnx.merge(graphdef, state_restored)

n_tasks = libero_envs.get_num_tasks()
for i in range(n_tasks):
    print(f"========== Processing task {i} ==========")
    task = libero_envs.task_suite.get_task(i)
    target_data = targets_map[task.name]
    first_target = target_data['target_objects'][0]
    print(task.name)
    print(task.language)

    for j, instance in enumerate(libero_envs.task_instantiations(i)):
        print(f"  Instantiation {j}")
        obs, env, task_description = instance

        # Try getting the target poses.
        # First, try getting the object by ID.
        target_id = first_target['id']
        target_loc = first_target['location']
        if f"{target_id}_pos" in obs:
            state = dict(pos=obs[f'{target_id}_pos'], quat=obs[f'{target_id}_quat'])
        elif target_loc in env.env.object_states_dict:
            state = env.env.object_states_dict[target_loc].get_geom_state()
        elif target_id in env.env.object_states_dict:
            state = env.env.object_states_dict[target_id].get_geom_state()
        else:
            raise ValueError(f"Could not locate object {first_target}")
        target_pose = RigidTransform.from_components(
            translation=state['pos'],
            rotation=Rotation.from_quat(state['quat'])
        )

        # TODO: Reject if no target_pose
        aligned_robot_frame = RigidTransform.from_components(
            translation=obs["robot0_eef_pos"],
            rotation=Rotation.identity()
        )
        eef_rot = Rotation.from_quat(obs["robot0_eef_quat"])
        # Normal multiplication is composition in scipy
        rel_transform = aligned_robot_frame.inv() * target_pose
        print(f'{obs["robot0_eef_pos"] = }')
        rel_transform = RigidTransform.from_components(
            translation=rel_transform.translation,
            rotation=eef_rot.inv() * rel_transform.rotation
        )
        print(f'{rel_transform.translation = }')

        policy.start()
        prompt = prompt_from_obs(obs, task_description)
        vla_output = infer_until_action(policy, prompt)
        intermediates = policy.saved_intermediates
        actions = vla_output['actions']

        pred_transform = probe(intermediates[0][None, :, 0, -1, :])[0]
        print(f'{first_target = }')
        print(f'{pred_transform = }')
        print(f'{rel_transform.translation = }')
        print(f'loss = {np.mean((pred_transform - rel_transform.translation)**2)}')

        trajectory_idx = 0
        for step in range(200):
            act = np.copy(actions[trajectory_idx])
            obs, reward, done, info = env.step(act)
            trajectory_idx += 1
            if trajectory_idx == (len(actions)//2):
                prompt = prompt_from_obs(obs, task_description)
                vla_output = infer_until_action(policy, prompt)
                actions = vla_output['actions']
                trajectory_idx = 0
                media.write_image("frame.png", obs['agentview_image'][::-1, ::-1, :])
                media.write_image("wrist_frame.png", obs['robot0_eye_in_hand_image'][::-1, ::-1, :])

                intermediates = policy.saved_intermediates

                aligned_robot_frame = RigidTransform.from_components(
                    translation=obs["robot0_eef_pos"],
                    rotation=Rotation.identity()
                )
                eef_rot = Rotation.from_quat(obs["robot0_eef_quat"])
                # Normal multiplication is composition in scipy
                rel_transform = aligned_robot_frame.inv() * target_pose
                print(f'{obs["robot0_eef_pos"] = }')
                rel_transform = RigidTransform.from_components(
                    translation=rel_transform.translation,
                    rotation=eef_rot.inv() * rel_transform.rotation
                )
                print(f'{rel_transform.translation = }')

                pred_transform = probe(intermediates[0][None, :, 0, -1, :])[0]
                print(f'Step {step}')
                print(f'{pred_transform = }')
                print(f'{rel_transform.translation = }')
                print(f'loss = {np.mean((pred_transform - rel_transform.translation)**2)}')
                if input().strip().lower() == 'q':
                    break

        input("Enter to continue next trajectory")
