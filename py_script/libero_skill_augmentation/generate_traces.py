import argparse
import json
import os

from pathlib import Path

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from common import (  # noqa: E402
    cumulative_episode_bounds,
    episode_shard_path,
    load_episode_frame,
    load_episode_records,
    load_json,
    load_lerobot_dataset,
    save_json_atomic,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract semantic target points and contact-point traces for an existing LIBERO skill annotation run."
        )
    )
    parser.add_argument(
        "--skill-annotations",
        type=Path,
        default=None,
        help="Explicit skill annotation JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs",
        help="Trace run directory. Defaults to outputs",
    )
    parser.add_argument(
        "--start-idx",
        "--start-episode",
        dest="start_episode",
        type=int,
        default=0,
        help="Inclusive episode start index.",
    )
    parser.add_argument(
        "--end-idx",
        "--end-episode",
        dest="end_episode",
        type=int,
        default=None,
        help="Exclusive episode end index. Defaults to one past the largest annotated episode id.",
    )
    parser.add_argument(
        "--image-key",
        choices=["image", "wrist_image"],
        default="image",
        help="Observation image stream used for target/trace extraction. Use image for the side-view camera.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip episodes whose target-trace shard file already exists in the output dir.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing target-trace shards instead of skipping them.",
    )
    parser.add_argument(
        "--disable-saving-trace-scenes",
        action="store_true",
        help="Disable saving start-frame target/trace overlay visualizations.",
    )
    parser.add_argument(
        "--libero-bddl-root",
        type=Path,
        default=None,
        help="Optional LIBERO bddl_files root for resolving per-task camera calibration.",
    )
    return parser.parse_args()

class LiberoEnvMaker:
    def __init__(self, suite: str,
                 render_resolution: int = 256, seed: int = 0,
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

def main() -> int:
    args = parse_args()

    with open(args.skill_annotations) as f:
        annotation_data = json.load(f)
    repo_id = annotation_data['source_repo_id']
    dataset_root = os.path.dirname(args.skill_annotations)

    records = load_episode_records(dataset_root)
    episode_bounds = cumulative_episode_bounds(records)
    print("loading lerobot dataset...", flush=True)
    dataset = load_lerobot_dataset(repo_id, dataset_root)
    print("dataset loaded", flush=True)

    DATASET = "libero_90"
    N_REPEATS = 9999
    libero_envs = LiberoEnvMaker(DATASET, repeats=N_REPEATS)

    task_generators = [libero_envs.task_instantiations(i) for i in range(libero_envs.get_num_tasks())]

    for counter, (episode_index, (start, end)) in enumerate(episode_bounds.items()):
        record = records[episode_index]
        obs, env, task_description = next(task_generators[counter % len(task_generators)])
        video_frames = []
        for i in range(30):
            row = dataset.hf_dataset[start + i]
            action = row['actions']
            obs, reward, done, info = env.step(action)
            video_frames.append(np.copy(obs['agentview_image'][::-1, ::-1, :]))
        media.write_video("out.mp4", video_frames)
        break

if __name__ == "__main__":
    raise SystemExit(main())
