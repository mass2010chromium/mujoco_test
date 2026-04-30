#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    DEFAULT_FPS,
    DEFAULT_REPO_ID,
    SKILL_ARG_COUNTS,
    as_uint8_hwc,
    decode_image_value,
    load_dataset_info,
    load_episode_records,
    load_episode_rows,
    load_json,
    overlay_step_text,
    resolve_dataset_root,
    resolve_image_key,
    validate_skill_expr,
)

RAW_PLAN_ITEM_RE = re.compile(r"(\d+)\.\s*")
RAW_SKILL_EXPR_RE = re.compile(r"^\s*([A-Z_]+)\s*\(")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render episode videos for CALVIN annotations, overlaying the active skill on each frame, "
            "and also write a concatenated combined video."
        )
    )
    parser.add_argument(
        "skill_annotations_json",
        type=Path,
        help="Canonical CALVIN skill_annotations.json file to visualize.",
    )
    parser.add_argument(
        "--num-step-plans",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="If set, only visualize episodes whose plan has exactly this many skills.",
    )
    parser.add_argument(
        "--task-keyword",
        default=None,
        help=(
            "Case-insensitive task_name substring filter. For example, --task-keyword unstack "
            "keeps only episodes whose task_name contains 'unstack'."
        ),
    )
    parser.add_argument(
        "--must-contain",
        dest="must_contain",
        action="append",
        nargs="+",
        default=None,
        help=(
            "Keep only episodes containing this skill name. Repeat the flag for multiple required skills, "
            "for example: --must-contain MOVE_SLIDER PICKUP_FROM."
        ),
    )
    parser.add_argument(
        "--skill-at",
        dest="skill_at",
        action="append",
        nargs=2,
        metavar=("POSITION", "SKILL_NAME"),
        default=None,
        help=(
            "Keep only episodes where the 1-based skill POSITION has this skill name. "
            "For example: --skill-at 1 MOVE_SLIDER. Repeat the flag for multiple positions."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of selected episodes to visualize after all filtering.",
    )
    parser.add_argument(
        "--annotation-view",
        action="store_true",
        help=(
            "Render the same overlays used by the annotation pipeline: step number, task prompt, "
            "and gripper state only."
        ),
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Optional repo id override. Defaults to the annotation metadata, then to the standard CALVIN repo id.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional dataset root override. Defaults to the annotation metadata, then to the cached snapshot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for episode videos and episode_combined_video.mp4. "
            "Defaults to a sibling directory named <input_stem>_visualizations."
        ),
    )
    parser.add_argument(
        "--image-key",
        choices=["top", "wrist", "observation.images.top", "observation.images.wrist"],
        default="top",
        help="Observation image stream to render. Defaults to the same top view used by annotation.",
    )
    return parser.parse_args()


def iter_annotation_episodes(data: dict[str, Any]) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for key in sorted((k for k in data if isinstance(k, str) and k.isdigit()), key=lambda item: int(item)):
        value = data[key]
        if not isinstance(value, dict):
            continue
        episode = dict(value)
        episode.setdefault("episode_index", int(key))
        episodes.append(episode)
    return episodes


def skill_count_for_episode(episode: dict[str, Any]) -> int:
    return len(raw_skill_strings_for_episode(episode))


def raw_plan_skill_strings(plan: str | list[Any]) -> list[str]:
    if isinstance(plan, list):
        return [str(skill).strip() for skill in plan if str(skill).strip()]

    text = " ".join(str(plan).strip().split())
    if text.startswith("Plan:"):
        text = text[len("Plan:") :].strip()
    matches = list(RAW_PLAN_ITEM_RE.finditer(text))
    if not matches:
        return [text] if text else []

    skills: list[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        skill = text[start:end].strip()
        if skill:
            skills.append(skill)
    return skills


def raw_skill_strings_for_episode(episode: dict[str, Any]) -> list[str]:
    plan = episode.get("plan")
    if isinstance(plan, (str, list)):
        return raw_plan_skill_strings(plan)

    segments = episode.get("segments")
    if isinstance(segments, list):
        skills = [str(segment.get("skill", "")).strip() for segment in segments if isinstance(segment, dict)]
        return [skill for skill in skills if skill]

    raise ValueError(f"Episode {episode.get('episode_index')} has neither a parseable plan nor segments.")


def skill_name_from_expr(skill: str) -> str:
    match = RAW_SKILL_EXPR_RE.match(skill)
    if not match:
        raise ValueError(f"Could not extract skill name from {skill!r}.")
    return match.group(1)


def skill_names_for_episode(episode: dict[str, Any]) -> set[str]:
    return {skill_name_from_expr(skill) for skill in raw_skill_strings_for_episode(episode)}


def skill_name_sequence_for_episode(episode: dict[str, Any]) -> list[str]:
    return [skill_name_from_expr(skill) for skill in raw_skill_strings_for_episode(episode)]


def normalize_task_keyword(raw_keyword: str | None) -> str | None:
    if raw_keyword is None:
        return None
    keyword = " ".join(raw_keyword.strip().split())
    if not keyword:
        raise ValueError("--task-keyword must be non-empty when provided.")
    return keyword.casefold()


def episode_matches_task_keyword(episode: dict[str, Any], task_keyword: str) -> bool:
    return task_keyword in str(episode.get("task_name", "")).casefold()


def flatten_required_skill_names(raw_skill_names: list[list[str]] | None) -> list[str]:
    if not raw_skill_names:
        return []
    return [skill_name for group in raw_skill_names for skill_name in group]


def normalize_required_skill_names(raw_skill_names: list[list[str]] | None) -> set[str]:
    flattened_skill_names = flatten_required_skill_names(raw_skill_names)
    if not flattened_skill_names:
        return set()

    normalized: set[str] = set()
    for raw_name in flattened_skill_names:
        skill_name = raw_name.strip().upper()
        if not skill_name:
            raise ValueError("--must-contain skill names must be non-empty.")
        if skill_name not in SKILL_ARG_COUNTS:
            choices = ", ".join(sorted(SKILL_ARG_COUNTS))
            raise ValueError(f"Unsupported --must-contain skill name {raw_name!r}. Expected one of: {choices}")
        normalized.add(skill_name)
    return normalized


def normalize_skill_at_constraints(raw_constraints: list[list[str]] | None) -> dict[int, str]:
    if not raw_constraints:
        return {}

    constraints: dict[int, str] = {}
    for raw_position, raw_name in raw_constraints:
        try:
            position = int(raw_position)
        except ValueError as exc:
            raise ValueError(f"--skill-at POSITION must be an integer, got {raw_position!r}.") from exc
        if position <= 0:
            raise ValueError(f"--skill-at POSITION is 1-based and must be positive, got {position}.")

        skill_name = raw_name.strip().upper()
        if not skill_name:
            raise ValueError("--skill-at skill names must be non-empty.")
        if skill_name not in SKILL_ARG_COUNTS:
            choices = ", ".join(sorted(SKILL_ARG_COUNTS))
            raise ValueError(f"Unsupported --skill-at skill name {raw_name!r}. Expected one of: {choices}")

        existing = constraints.get(position)
        if existing is not None and existing != skill_name:
            raise ValueError(
                f"Conflicting --skill-at constraints for position {position}: {existing} and {skill_name}."
            )
        constraints[position] = skill_name
    return constraints


def episode_matches_skill_at_constraints(episode: dict[str, Any], constraints: dict[int, str]) -> bool:
    if not constraints:
        return True

    skill_names = skill_name_sequence_for_episode(episode)
    for position, expected_skill_name in constraints.items():
        skill_idx = position - 1
        if skill_idx >= len(skill_names):
            return False
        if skill_names[skill_idx] != expected_skill_name:
            return False
    return True


def skill_for_display(skill: str) -> str:
    try:
        return validate_skill_expr(skill)
    except ValueError:
        return " ".join(skill.strip().split())


def wrapped_skill_text(draw: Any, skill: str, image_width: int) -> str:
    for wrap_width in (28, 24, 20, 18, 16, 14):
        lines = textwrap.wrap(
            skill,
            width=wrap_width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        text = "\n".join(lines) if lines else skill
        try:
            left, top, right, bottom = draw.multiline_textbbox((0, 0), text, spacing=1)
        except AttributeError:
            bbox = draw.textbbox((0, 0), text)
            left, top, right, bottom = bbox
        if (right - left) <= max(40, image_width - 16) and (bottom - top) <= 30:
            return text
    return skill


def overlay_skill_text(frame: Any, skill: str) -> Any:
    from PIL import Image, ImageDraw

    image = Image.fromarray(as_uint8_hwc(frame))
    draw = ImageDraw.Draw(image)
    skill_text = wrapped_skill_text(draw, skill_for_display(skill), image.width)
    try:
        left, top, right, bottom = draw.multiline_textbbox((0, 0), skill_text, spacing=1)
    except AttributeError:
        bbox = draw.textbbox((0, 0), skill_text)
        left, top, right, bottom = bbox

    text_width = right - left
    text_height = bottom - top
    x0 = max(0, image.width - text_width - 14)
    y0 = max(0, 4)
    x1 = min(image.width, image.width - 4)
    y1 = min(image.height, y0 + text_height + 6)
    draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
    draw.multiline_text((x0 + 3, y0 + 2), skill_text, fill=(255, 255, 255), spacing=1)
    return image


def overlay_episode_index_text(frame: Any, episode_index: int) -> Any:
    from PIL import Image, ImageDraw

    image = Image.fromarray(as_uint8_hwc(frame))
    draw = ImageDraw.Draw(image)
    label = f"episode {int(episode_index):06d}"
    try:
        left, top, right, bottom = draw.textbbox((0, 0), label)
    except AttributeError:  # pragma: no cover - older Pillow fallback
        left, top, right, bottom = (0, 0, 8 * len(label), 11)

    text_width = right - left
    text_height = bottom - top
    x0 = max(0, image.width - text_width - 14)
    y0 = max(0, image.height - text_height - 8)
    x1 = min(image.width, image.width - 4)
    y1 = min(image.height, image.height - 4)
    draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
    draw.text((x0 + 3, y0 + 2), label, fill=(255, 255, 255))
    return image


def active_skill_for_step(segments: list[dict[str, Any]], local_step: int, segment_idx: int) -> tuple[str, int]:
    while segment_idx + 1 < len(segments) and local_step >= int(segments[segment_idx]["end_step"]):
        segment_idx += 1

    current_segment = segments[segment_idx]
    start_step = int(current_segment["start_step"])
    end_step = int(current_segment["end_step"])
    if not (start_step <= local_step < end_step):
        raise ValueError(
            f"Step {local_step} is not covered by the current segment {segment_idx}: {current_segment!r}"
        )
    return skill_for_display(str(current_segment["skill"])), segment_idx


def render_episode_with_skill_overlay(
    *,
    episode_index: int,
    episode_rows: list[dict[str, Any]],
    record: Any,
    segments: list[dict[str, Any]],
    output_path: Path,
    combined_writer: Any,
    image_key: str,
    fps: int,
    annotation_view: bool,
) -> None:
    import imageio.v2 as imageio

    resolved_image_key = resolve_image_key(image_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segment_idx = 0
    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=7) as writer:
        for local_step, item in enumerate(episode_rows):
            skill, segment_idx = active_skill_for_step(segments, local_step, segment_idx)
            frame = decode_image_value(item[resolved_image_key])
            frame = overlay_step_text(
                frame,
                step_idx=local_step,
                total_steps=record.length,
                instruction=record.instruction,
                gripper_state=item["observation.state"] if annotation_view else None,
            )
            if not annotation_view:
                frame = overlay_skill_text(frame, skill)
                frame = overlay_episode_index_text(frame, episode_index)
            encoded_frame = as_uint8_hwc(frame)
            writer.append_data(encoded_frame)
            combined_writer.append_data(encoded_frame)


def resolve_dataset_root_from_metadata(
    annotation_data: dict[str, Any],
    *,
    repo_id_override: str | None,
    dataset_root_override: Path | None,
) -> tuple[str, Path]:
    repo_id = repo_id_override or str(annotation_data.get("source_repo_id") or DEFAULT_REPO_ID)
    metadata_root = annotation_data.get("dataset_root")
    if dataset_root_override is not None:
        dataset_root = resolve_dataset_root(repo_id, dataset_root_override)
    elif isinstance(metadata_root, str) and metadata_root.strip():
        dataset_root = resolve_dataset_root(repo_id, metadata_root)
    else:
        dataset_root = resolve_dataset_root(repo_id, None)
    return repo_id, dataset_root


def main() -> int:
    import imageio.v2 as imageio

    args = parse_args()
    annotation_path = args.skill_annotations_json.expanduser().resolve()
    data = load_json(annotation_path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a top-level JSON object in {annotation_path}, got {type(data).__name__}.")

    _, dataset_root = resolve_dataset_root_from_metadata(
        data,
        repo_id_override=args.repo_id,
        dataset_root_override=args.dataset_root,
    )
    dataset_info = load_dataset_info(dataset_root)
    records = load_episode_records(dataset_root)
    fps = int(data.get("fps", DEFAULT_FPS))

    episodes = iter_annotation_episodes(data)
    if args.num_step_plans is not None:
        episodes = [episode for episode in episodes if skill_count_for_episode(episode) == args.num_step_plans]
    task_keyword = normalize_task_keyword(args.task_keyword)
    if task_keyword is not None:
        episodes = [episode for episode in episodes if episode_matches_task_keyword(episode, task_keyword)]
    required_skill_names = normalize_required_skill_names(args.must_contain)
    if required_skill_names:
        episodes = [
            episode
            for episode in episodes
            if required_skill_names.issubset(skill_names_for_episode(episode))
        ]
    skill_at_constraints = normalize_skill_at_constraints(args.skill_at)
    if skill_at_constraints:
        episodes = [
            episode
            for episode in episodes
            if episode_matches_skill_at_constraints(episode, skill_at_constraints)
        ]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(f"--limit must be positive, got {args.limit}.")
        episodes = episodes[: args.limit]

    if not episodes:
        filter_parts: list[str] = []
        if args.num_step_plans is not None:
            filter_parts.append(f"num-step-plans={args.num_step_plans}")
        if task_keyword is not None:
            filter_parts.append(f"task-keyword={task_keyword}")
        if required_skill_names:
            filter_parts.append(f"must-contain={','.join(sorted(required_skill_names))}")
        if skill_at_constraints:
            filter_parts.append(
                "skill-at="
                + ",".join(
                    f"{position}:{skill_name}" for position, skill_name in sorted(skill_at_constraints.items())
                )
            )
        filter_desc = f" with {' and '.join(filter_parts)}" if filter_parts else ""
        raise ValueError(f"No episodes found in {annotation_path}{filter_desc}.")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else annotation_path.parent / f"{annotation_path.stem}_visualizations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "episode_combined_video.mp4"
    selected_episodes_path = output_dir / "selected_episodes.txt"

    print(f"input_json={annotation_path}", flush=True)
    print(f"dataset_root={dataset_root}", flush=True)
    print(f"selected_episodes={len(episodes)}", flush=True)
    if args.num_step_plans is not None:
        print(f"num_step_plans={args.num_step_plans}", flush=True)
    if task_keyword is not None:
        print(f"task_keyword={task_keyword}", flush=True)
    if required_skill_names:
        print(f"must_contain={','.join(sorted(required_skill_names))}", flush=True)
    if skill_at_constraints:
        print(
            "skill_at="
            + ",".join(f"{position}:{skill_name}" for position, skill_name in sorted(skill_at_constraints.items())),
            flush=True,
        )
    if args.limit is not None:
        print(f"limit={args.limit}", flush=True)
    print(f"annotation_view={args.annotation_view}", flush=True)
    print(f"image_key={resolve_image_key(args.image_key)}", flush=True)
    print(f"output_dir={output_dir}", flush=True)

    with imageio.get_writer(combined_path, fps=fps, codec="libx264", quality=7) as combined_writer:
        for order_idx, episode in enumerate(episodes, start=1):
            episode_index = int(episode["episode_index"])
            if not (0 <= episode_index < len(records)):
                raise ValueError(
                    f"Episode index {episode_index} from {annotation_path} is out of bounds for dataset "
                    f"with {len(records)} episodes."
                )

            segments = episode.get("segments")
            if not isinstance(segments, list) or not segments:
                raise ValueError(f"Episode {episode_index} has no canonical segments.")

            record = records[episode_index]
            episode_rows = load_episode_rows(
                dataset_root,
                episode_index,
                dataset_info=dataset_info,
            )
            if len(episode_rows) != record.length:
                raise ValueError(
                    f"Episode parquet length mismatch for episode {episode_index}: "
                    f"meta length={record.length}, parquet rows={len(episode_rows)}"
                )

            output_path = output_dir / f"episode_{episode_index:06d}_video.mp4"
            render_episode_with_skill_overlay(
                episode_index=episode_index,
                episode_rows=episode_rows,
                record=record,
                segments=segments,
                output_path=output_path,
                combined_writer=combined_writer,
                image_key=args.image_key,
                fps=fps,
                annotation_view=args.annotation_view,
            )

            print(
                f"[{order_idx}/{len(episodes)}] episode={episode_index} "
                f"num_skills={skill_count_for_episode(episode)} "
                f"task_name={record.task_name} output_video={output_path}",
                flush=True,
            )

    with selected_episodes_path.open("w", encoding="utf-8") as f:
        for episode in episodes:
            f.write(f"{int(episode['episode_index'])}\n")

    print(f"combined_output={combined_path}", flush=True)
    print(f"selected_episodes_output={selected_episodes_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
