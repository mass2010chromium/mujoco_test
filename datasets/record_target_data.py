import json
import math
import os
from pathlib import Path
import sys
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

import jax.numpy as jnp
import numpy as np
import mediapy as media
from scipy.spatial.transform import Rotation, RigidTransform

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

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
    policy = _policy_config.create_trained_skill_reasoning_policy(
        vla_config,
        checkpoint_dir,
        norm_stats=norm_stats,
        initial_scene_plan=initial_scene_plan,
        sample_kwargs={
            "temperature": 0.0,
            "max_reasoning_steps": 256,
            "force_initial_reasoning": True,
            "debug_prefill": True,
        }
    )
    return policy

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

    if mode == 'thinking':
        if scene_plan == '':
            thought_prefix = task
        else:
            thought_prefix = scene_plan
        text_input = thought_prefix
    
    else:
        text_input = skill

    return {
        'observation/image': obs['agentview_image'][::-1, ::-1, :],
        'observation/wrist_image': obs['robot0_eye_in_hand_image'][::-1, ::-1, :],
        "observation/state": state_vec,
        'prompt': task,
        'thought': [text_input],
        'mode': mode,
    }

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


def get_component_transform(target_info):
    # Try getting the target poses.
    # First, try getting the object by ID.
    target_id = target_info['id']
    target_loc = target_info['location']
    if f"{target_id}_pos" in obs:
        state = dict(pos=obs[f'{target_id}_pos'], quat=obs[f'{target_id}_quat'])
    elif target_loc in env.env.object_states_dict:
        state = env.env.object_states_dict[target_loc].get_geom_state()
    elif target_id in env.env.object_states_dict:
        state = env.env.object_states_dict[target_id].get_geom_state()
    else:
        raise ValueError(f"Could not locate object {first_target}")
    return RigidTransform.from_components(
        translation=state['pos'],
        rotation=Rotation.from_quat(state['quat'])
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset name (ex. libero_10)")
    parser.add_argument("--repeats", type=int, default=5, help="Number of times to run each experiment")
    parser.add_argument("--seed", type=int, default=0, help="Libero simulation seed")
    parser.add_argument("--token-count", type=int, default=10, help="Number of tokens to probe from")
    args = parser.parse_args()
    PROBE_LAYERS = [9]

    N_REPEATS = args.repeats
    SEED = args.seed
    DATASET = args.dataset
    libero_envs = LiberoEnvMaker(DATASET, render_resolution=224, seed=SEED, repeats=N_REPEATS)

    # Read targets generated by libero_get_targets.py
    targets_dir = Path(__file__).parent / 'libero_targets'
    targets_file = targets_dir / f"{DATASET}.json"
    targets_data = json.load(open(targets_file, 'r'))
    targets_map = {x['task_name']: x for x in targets_data.values()}    # Remap to task names


    output_dir = Path(__file__).parent / 'outputs_and_transforms' / DATASET
    os.makedirs(output_dir, exist_ok=True)
    meta_file = output_dir / "meta.json"

    meta = dict(
        seed=SEED,
        dataset=DATASET,
        repeats=N_REPEATS
    )
    with open(meta_file, 'w') as f:
        json.dump(meta, f)

    openpi_root = Path(__file__).resolve().parent.parent / "pace" / "openpi"
    assets_dir = openpi_root / "assets" / "pi05_libero_skill_reason_fixed"
    checkpoint_dir = (
        openpi_root
        / "checkpoints"
        / "pi05_libero_skill_reason_fixed"
        / "pi05_libero_skill_reason_fixed"
        / "29999"
    )
    # Create a trained policy.
    policy = create_pi05(   # Use LoRA weights:
        'pi05_libero_skill_reason_lora_v2',
        checkpoint_dir=checkpoint_dir,
        assets_dir=assets_dir
    )

    n_tasks = libero_envs.get_num_tasks()
    for i in range(n_tasks):
        print(f"========== Processing task {i} ==========")
        task = libero_envs.task_suite.get_task(i)
        target_data = targets_map[task.name]
        targets = target_data['target_objects']
        print(task.name)
        print(task.language)

        for j, instance in enumerate(libero_envs.task_instantiations(i)):
            print(f"  Instantiation {j}")
            obs, env, task_description = instance

            # Track our step in the symbolic plan / position plan
            target_idx = 0
            target_pose = get_component_transform(targets[target_idx])
            # TODO: Reject if no target_pose
            aligned_robot_frame = RigidTransform.from_components(
                translation=obs["robot0_eef_pos"],
                rotation=Rotation.identity()
            )
            eef_rot = Rotation.from_quat(obs["robot0_eef_quat"])
            # Normal multiplication is composition in scipy
            rel_transform = aligned_robot_frame.inv() * target_pose
            rel_transform = RigidTransform.from_components(
                translation=rel_transform.translation,
                rotation=eef_rot.inv() * rel_transform.rotation
            ).as_exp_coords()

            policy.start()
            # first, think once to populate the plan
            prompt = prompt_from_obs(obs, task_description, scene_plan=f'Instruction: {task.language}', mode='thinking')
            result = policy.infer(prompt)
            scene_plan = result['subtask']  # Error if none... please be a good vla

            prompt = prompt_from_obs(obs, task_description, scene_plan=scene_plan, mode='thinking')
            result = policy.infer(prompt)
            skill = result['subtask']  # Error if none... please be a good vla
            print(f'{scene_plan = }')
            print(f'initial {skill = }')

            prompt = prompt_from_obs(obs, task_description, skill=skill, mode='acting')
            vla_output = policy.infer(prompt)
            intermediates = policy.saved_intermediates[0][:, :, -args.token_count:, :]
            #keys = policy.saved_kv_cache[1][PROBE_LAYERS, :, :, 0, :]
            #values = policy.saved_kv_cache[2][PROBE_LAYERS, :, :, 0, :]
            #intermediates = jnp.concat([keys, values], axis=0)
            actions = vla_output['actions']

            all_intermediates = [intermediates]
            all_actions = [np.array(actions)]
            all_targets = [rel_transform]

            trajectory_idx = 0
            skill_gen_counter = 0
            SKILL_UPDATE_FREQ = 2
            last_transition_time = 0
            step_timeout = 200
            media.write_image("frame0.png", obs['agentview_image'][::-1, ::-1, :])
            media.write_image("wrist_frame0.png", obs['robot0_eye_in_hand_image'][::-1, ::-1, :])
            for step in range(500):
                act = np.copy(actions[trajectory_idx])
                obs, reward, done, info = env.step(act)
                trajectory_idx += 1

                if step - last_transition_time > step_timeout:
                    print(f"{step_timeout} steps without new skill, exiting")
                    break
                if trajectory_idx == (len(actions)//2):
                    skill_gen_counter += 1
                    if skill_gen_counter == SKILL_UPDATE_FREQ:
                        prompt = prompt_from_obs(obs, task_description, scene_plan=scene_plan, mode='thinking')
                        result = policy.infer(prompt)
                        new_skill = result['subtask']  # Error if none... please be a good vla
                        print(f'updated {new_skill = }')
                        if new_skill != skill:
                            print("Updating target pose")
                            # Advance item
                            # A bit of hopium here that the networks don't go batshit insane
                            if target_idx < (len(targets)-1):
                                target_idx += 1
                                last_transition_time = step
                                skill = new_skill
                            else:
                                print("WARNING: ran out of steps to execute!")
                                break
                            print(f"Target object: {targets[target_idx]['id']}, {targets[target_idx]['location']}")
                        skill_gen_counter = 0
                    prompt = prompt_from_obs(obs, task_description, skill=skill, mode='acting')
                    vla_output = policy.infer(prompt)
                    actions = vla_output['actions']
                    trajectory_idx = 0

                    # layer, batch, token, dimension
                    intermediates = policy.saved_intermediates[0][:, :, -args.token_count:, :]
                    #keys = policy.saved_kv_cache[1][PROBE_LAYERS, :, :, 0, :]
                    #values = policy.saved_kv_cache[2][PROBE_LAYERS, :, :, 0, :]
                    #intermediates = jnp.concat([keys, values], axis=0)

                    target_pose = get_component_transform(targets[target_idx])
                    aligned_robot_frame = RigidTransform.from_components(
                        translation=obs["robot0_eef_pos"],
                        rotation=Rotation.identity()
                    )
                    eef_rot = Rotation.from_quat(obs["robot0_eef_quat"])
                    # Normal multiplication is composition in scipy
                    rel_transform = aligned_robot_frame.inv() * target_pose
                    rel_transform = RigidTransform.from_components(
                        translation=rel_transform.translation,
                        rotation=eef_rot.inv() * rel_transform.rotation
                    ).as_exp_coords()

                    all_intermediates.append(intermediates)
                    all_actions.append(np.array(actions))
                    all_targets.append(rel_transform)

            media.write_image("frame.png", obs['agentview_image'][::-1, ::-1, :])
            media.write_image("wrist_frame.png", obs['robot0_eye_in_hand_image'][::-1, ::-1, :])
            jnp.save(output_dir / f"{task.name}_{j}_intermediate.npy", jnp.stack(all_intermediates))
            np.save(output_dir / f"{task.name}_{j}_action.npy", np.stack(all_actions))
            np.save(output_dir / f"{task.name}_{j}_transform.npy", np.stack(all_targets))
