#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    DEFAULT_REPO_ID,
    aggregate_episode_annotations,
    aggregate_training_annotations,
    list_episode_shards,
    load_episode_records,
    load_json,
    resolve_dataset_root,
    save_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine episode-wise CALVIN annotation shard JSON files into a canonical skill annotation JSON "
            "and a training-oriented cot-style skill annotation JSON."
        )
    )
    parser.add_argument(
        "shard_dirs",
        nargs="+",
        type=Path,
        help="One or more directories containing episode shard files.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot dataset repo id.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Existing LeRobot dataset root. Defaults to the cached snapshot for --repo-id.",
    )
    parser.add_argument(
        "--canonical-output",
        type=Path,
        default=None,
        help="Output path for the combined canonical annotation JSON.",
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        default=None,
        help="Output path for the combined training-oriented cot JSON.",
    )
    parser.add_argument(
        "--boundary-window",
        type=int,
        default=10,
        help="Number of frames at the start of each skill to expose as a reasoning update window.",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="Inclusive lower episode bound to include when combining.",
    )
    parser.add_argument(
        "--end-episode",
        type=int,
        default=None,
        help="Exclusive upper episode bound to include when combining.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing episodes inside the requested range instead of erroring.",
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
                    f"Duplicate episode shard for episode {episode_index}: "
                    f"already loaded once before seeing {shard_path}"
                )
            combined[episode_index] = episode
    return combined


def main() -> int:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.repo_id, args.dataset_root)
    records = load_episode_records(dataset_root)
    total_episodes = len(records)
    end_episode = total_episodes if args.end_episode is None else min(args.end_episode, total_episodes)
    if not (0 <= args.start_episode < end_episode <= total_episodes):
        raise ValueError(
            f"Invalid range [{args.start_episode}, {end_episode}) for dataset with {total_episodes} episodes."
        )

    loaded = load_shards_from_dirs([path.expanduser().resolve() for path in args.shard_dirs])
    requested_indices = list(range(args.start_episode, end_episode))
    missing = [idx for idx in requested_indices if idx not in loaded]
    if missing and not args.allow_missing:
        sample = ", ".join(str(idx) for idx in missing[:20])
        raise ValueError(
            f"Missing {len(missing)} episode shard(s) in the requested range [{args.start_episode}, {end_episode}). "
            f"Sample: {sample}"
        )

    selected = [loaded[idx] for idx in requested_indices if idx in loaded]
    canonical_output = (
        args.canonical_output.expanduser().resolve()
        if args.canonical_output is not None
        else args.shard_dirs[0].expanduser().resolve().parent / "skill_annotations.json"
    )
    training_output = (
        args.training_output.expanduser().resolve()
        if args.training_output is not None
        else args.shard_dirs[0].expanduser().resolve().parent / "cot_skill.json"
    )

    canonical = aggregate_episode_annotations(
        selected,
        source_repo_id=args.repo_id,
        dataset_root=dataset_root,
    )
    training = aggregate_training_annotations(
        selected,
        boundary_window=args.boundary_window,
        source_repo_id=args.repo_id,
        dataset_root=dataset_root,
    )

    save_json_atomic(canonical_output, canonical)
    save_json_atomic(training_output, training)

    print(f"combined_shards={len(selected)}", flush=True)
    print(f"canonical_output={canonical_output}", flush=True)
    print(f"training_output={training_output}", flush=True)
    if missing:
        print(f"missing_episodes={len(missing)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
