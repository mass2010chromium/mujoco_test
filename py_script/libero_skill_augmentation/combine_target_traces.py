#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import DEFAULT_REPO_ID, list_episode_shards, load_json, save_json_atomic  # noqa: E402
from common_trace import aggregate_episode_target_traces, load_skill_annotation_episodes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine episode-wise target-trace annotation shards into skill_target_traces.json, "
            "aligned with an existing skill_annotations.json file."
        )
    )
    parser.add_argument(
        "shard_dirs",
        nargs="+",
        type=Path,
        help="One or more directories containing target-trace episode shard files.",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        required=True,
        help="Folder containing the existing skill_annotations.json. skill_target_traces.json is written here by default.",
    )
    parser.add_argument(
        "--skill-annotations",
        type=Path,
        default=None,
        help="Explicit skill annotation JSON path. Defaults to annotation-dir/skill_annotations.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the combined target-trace JSON. Defaults to annotation-dir/skill_target_traces.json.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot dataset repo id.")
    parser.add_argument("--start-idx", "--start-episode", dest="start_episode", type=int, default=0)
    parser.add_argument("--end-idx", "--end-episode", dest="end_episode", type=int, default=None)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing target-trace shards inside the requested annotated episode range.",
    )
    return parser.parse_args()


def load_shards_from_dirs(shard_dirs: list[Path]) -> dict[int, dict[str, Any]]:
    combined: dict[int, dict[str, Any]] = {}
    for shard_dir in shard_dirs:
        for shard_path in list_episode_shards(shard_dir):
            episode = load_json(shard_path)
            episode_index = int(episode["episode_index"])
            if episode_index in combined:
                raise ValueError(
                    f"Duplicate target-trace shard for episode {episode_index}: "
                    f"already loaded once before seeing {shard_path}"
                )
            combined[episode_index] = episode
    return combined


def validate_target_trace_episode_against_skill(
    target_episode: dict[str, Any],
    skill_episode: dict[str, Any],
) -> None:
    episode_index = int(skill_episode["episode_index"])
    if int(target_episode["episode_index"]) != episode_index:
        raise ValueError(
            f"Target-trace episode index {target_episode.get('episode_index')} "
            f"does not match skill episode {episode_index}."
        )

    skill_segments = skill_episode.get("segments")
    target_segments = target_episode.get("segments")
    target_traces = target_episode.get("target_traces")
    if not isinstance(skill_segments, list) or not isinstance(target_segments, list) or not isinstance(target_traces, list):
        raise ValueError(f"Episode {episode_index} must contain segment and target_traces lists.")
    if target_segments != skill_segments:
        raise ValueError(f"Episode {episode_index} target-trace segments do not exactly match skill annotation segments.")
    if len(target_traces) != len(skill_segments):
        raise ValueError(
            f"Episode {episode_index} has {len(target_traces)} target traces but {len(skill_segments)} skill segments."
        )

    for idx, (entry, segment) in enumerate(zip(target_traces, skill_segments, strict=True)):
        if int(entry["skill_index"]) != idx:
            raise ValueError(f"Episode {episode_index} entry {idx} has wrong skill_index {entry.get('skill_index')}.")
        if str(entry["skill"]) != str(segment["skill"]):
            raise ValueError(f"Episode {episode_index} entry {idx} skill does not match segment skill.")
        if int(entry["start_step"]) != int(segment["start_step"]) or int(entry["end_step"]) != int(segment["end_step"]):
            raise ValueError(f"Episode {episode_index} entry {idx} interval does not match segment interval.")
        semantic_enabled = bool(target_episode.get("semantic_target_enabled", True))
        if semantic_enabled and not isinstance(entry.get("semantic_target"), dict):
            raise ValueError(f"Episode {episode_index} entry {idx} is missing semantic_target.")
        if target_episode.get("end_effector_trace_enabled") and not isinstance(entry.get("end_effector_trace"), dict):
            raise ValueError(f"Episode {episode_index} entry {idx} is missing end_effector_trace.")


def main() -> int:
    args = parse_args()
    annotation_dir = args.annotation_dir.expanduser().resolve()
    skill_annotations_path = (
        args.skill_annotations.expanduser().resolve()
        if args.skill_annotations is not None
        else annotation_dir / "skill_annotations.json"
    )
    skill_data, skill_episodes = load_skill_annotation_episodes(skill_annotations_path)
    loaded_traces = load_shards_from_dirs([path.expanduser().resolve() for path in args.shard_dirs])

    start_episode = max(0, int(args.start_episode))
    end_episode = max(skill_episodes) + 1 if args.end_episode is None else int(args.end_episode)
    if not (0 <= start_episode < end_episode):
        raise ValueError(f"Invalid episode range [{start_episode}, {end_episode}).")

    requested_indices = [idx for idx in range(start_episode, end_episode) if idx in skill_episodes]
    missing = [idx for idx in requested_indices if idx not in loaded_traces]
    if missing and not args.allow_missing:
        sample = ", ".join(str(idx) for idx in missing[:20])
        raise ValueError(
            f"Missing {len(missing)} target-trace shard(s) in the requested annotated range "
            f"[{start_episode}, {end_episode}). Sample: {sample}"
        )

    selected: list[dict[str, Any]] = []
    for episode_index in requested_indices:
        if episode_index not in loaded_traces:
            continue
        target_episode = loaded_traces[episode_index]
        validate_target_trace_episode_against_skill(target_episode, skill_episodes[episode_index])
        selected.append(target_episode)

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else annotation_dir / "skill_target_traces.json"
    )
    source_repo_id = str(skill_data.get("source_repo_id", args.repo_id))
    dataset_root = str(skill_data.get("dataset_root", ""))
    combined = aggregate_episode_target_traces(
        selected,
        skill_annotations_path=skill_annotations_path,
        source_repo_id=source_repo_id,
        dataset_root=dataset_root,
    )
    save_json_atomic(output_path, combined)

    print(f"combined_target_trace_shards={len(selected)}", flush=True)
    print(f"output={output_path}", flush=True)
    if missing:
        print(f"missing_episodes={len(missing)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
