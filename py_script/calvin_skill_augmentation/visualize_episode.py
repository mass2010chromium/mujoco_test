#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    DEFAULT_REPO_ID,
    load_dataset_info,
    load_episode_records,
    load_episode_rows,
    render_episode_video,
    resolve_dataset_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a CALVIN episode video with the same frame-index overlay used by the "
            "annotation pipeline."
        )
    )
    parser.add_argument("episode_index", type=int, help="Episode index to visualize.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot dataset repo id.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Existing LeRobot dataset root. Defaults to the cached snapshot for --repo-id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where episode_xxxxxx_video.mp4 will be written. Defaults to the current directory.",
    )
    parser.add_argument(
        "--image-key",
        choices=["top", "wrist", "observation.images.top", "observation.images.wrist"],
        default="top",
        help="Observation image stream to render. Defaults to the same top view used by annotation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.repo_id, args.dataset_root)
    records = load_episode_records(dataset_root)

    if not (0 <= args.episode_index < len(records)):
        raise ValueError(
            f"Episode index {args.episode_index} is out of bounds for dataset with {len(records)} episodes."
        )

    record = records[args.episode_index]
    dataset_info = load_dataset_info(dataset_root)
    episode_rows = load_episode_rows(
        dataset_root,
        args.episode_index,
        dataset_info=dataset_info,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_path = output_dir / f"episode_{args.episode_index:06d}_video.mp4"
    render_episode_video(
        episode_rows,
        record=record,
        output_path=output_path,
        image_key=args.image_key,
        overlay_text=True,
    )

    print(f"task_name={record.task_name}", flush=True)
    print(f"raw_instruction={record.raw_instruction}", flush=True)
    print(f"instruction={record.instruction}", flush=True)
    print(f"num_steps={record.length}", flush=True)
    print(f"output_video={output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
