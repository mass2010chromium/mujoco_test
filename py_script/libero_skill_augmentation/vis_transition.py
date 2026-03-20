#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    load_json,
    parse_plan_skills,
    transition_scene_paths,
    validate_skill_expr,
)


EPISODE_DIR_RE = re.compile(r"^episode_(\d{6})$")
ANNOTATION_CANDIDATES = ("cot_skill.json", "skill_annotations.json")
OUTPUT_IMAGE_NAME = "vis_transition.png"


@dataclass(frozen=True)
class VisualizationExample:
    episode_index: int
    view_kind: str
    transition_index: int
    segment_index: int
    boundary_step: int
    instruction: str | None
    plan: str
    image_path: Path
    skill_before: str | None = None
    skill_after: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize saved LIBERO initial-scene and skill-transition frames one at a time. "
            f"The script writes {OUTPUT_IMAGE_NAME} in the current working directory and overwrites it "
            "on each iteration."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Completed run directory containing a combined annotation JSON and transition_scenes roots.",
    )
    parser.add_argument(
        "--random-order",
        action="store_true",
        help=(
            "Traverse episodes sequentially from the beginning. When omitted, each iteration samples "
            "a random episode and then a random transition inside that episode."
        ),
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help=(
            "Optional starting episode index. In sequential mode, traversal starts from this episode. "
            "In random mode, the script first shows this episode's initial scene and transitions, then "
            "continues with random transition sampling."
        ),
    )
    return parser.parse_args()


def load_combined_episodes(path: Path) -> tuple[dict[int, dict[str, Any]], Path]:
    annotation_path = None
    for name in ANNOTATION_CANDIDATES:
        candidate = path / name
        if candidate.is_file():
            annotation_path = candidate.resolve()
            break
    if annotation_path is None:
        tried = ", ".join(str(path / name) for name in ANNOTATION_CANDIDATES)
        raise FileNotFoundError(f"Could not find a combined annotation JSON under {path}. Tried: {tried}")

    data = load_json(annotation_path)
    if not isinstance(data, dict):
        raise ValueError(f"Combined annotation file must contain a JSON object: {annotation_path}")

    episodes: dict[int, dict[str, Any]] = {}
    for key, value in data.items():
        if not (isinstance(key, str) and key.isdigit() and isinstance(value, dict)):
            continue
        episode = dict(value)
        episode.setdefault("episode_index", int(key))
        episodes[int(key)] = episode

    if not episodes:
        raise ValueError(f"No episode records found in {annotation_path}")
    return episodes, annotation_path


def resolve_episode_plan(episode: dict[str, Any]) -> str:
    if isinstance(episode.get("plan"), str):
        return episode["plan"]

    segments = episode.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"Episode {episode.get('episode_index')} has no segments to recover a plan from.")

    segment_plans = [segment.get("plan") for segment in segments if isinstance(segment.get("plan"), str)]
    if not segment_plans:
        raise ValueError(f"Episode {episode.get('episode_index')} has no recoverable plan string.")

    first_plan = segment_plans[0]
    first_plan_skills = parse_plan_skills(first_plan)
    for idx, segment_plan in enumerate(segment_plans[1:], start=1):
        if parse_plan_skills(segment_plan) != first_plan_skills:
            raise ValueError(
                f"Episode {episode.get('episode_index')} has inconsistent segment plan strings at index {idx}."
            )
    return first_plan


def extract_instruction(episode: dict[str, Any]) -> str | None:
    instruction = episode.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()

    segments = episode.get("segments")
    if not isinstance(segments, list):
        return None

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        for field_name in ("instruction", "updated_content_w_instruction", "content"):
            candidate = segment.get(field_name)
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            text = candidate.strip()
            for line in text.splitlines():
                if line.startswith("Instruction: "):
                    return line[len("Instruction: ") :].strip()
            if field_name == "instruction":
                return text
    return None


def canonical_segments_from_episode(episode: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = episode.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(f"Episode {episode.get('episode_index')} has no segments.")

    canonical: list[dict[str, Any]] = []
    for idx, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            raise ValueError(f"Episode {episode.get('episode_index')} segment {idx} is not a JSON object.")

        try:
            start_step = int(raw_segment["start_step"])
            end_step = int(raw_segment["end_step"])
        except Exception as exc:
            raise ValueError(
                f"Episode {episode.get('episode_index')} segment {idx} is missing integer start/end steps."
            ) from exc

        skill = validate_skill_expr(str(raw_segment["skill"]))
        segment = {"start_step": start_step, "end_step": end_step, "skill": skill}

        if end_step <= start_step:
            raise ValueError(f"Episode {episode.get('episode_index')} segment {idx} has non-positive length.")

        if not canonical:
            if start_step != 0:
                raise ValueError(f"Episode {episode.get('episode_index')} first segment must start at 0.")
            canonical.append(segment)
            continue

        if start_step != canonical[-1]["end_step"]:
            raise ValueError(
                f"Episode {episode.get('episode_index')} segments are not contiguous at segment {idx}: "
                f"expected start_step {canonical[-1]['end_step']}, got {start_step}."
            )

        if skill == canonical[-1]["skill"]:
            canonical[-1]["end_step"] = end_step
        else:
            canonical.append(segment)

    return canonical


def discover_transition_scene_roots(run_dir: Path) -> list[Path]:
    resolved = run_dir.expanduser().resolve()
    candidates: list[Path] = []
    if resolved.name == "transition_scenes" and resolved.is_dir():
        candidates.append(resolved)
    direct = resolved / "transition_scenes"
    if direct.is_dir():
        candidates.append(direct.resolve())
    if resolved.is_dir():
        for candidate in resolved.rglob("transition_scenes"):
            if candidate.is_dir():
                candidates.append(candidate.resolve())

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in sorted(candidates):
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def build_episode_transition_root_map(roots: list[Path]) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for root in roots:
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            match = EPISODE_DIR_RE.match(child.name)
            if match is None:
                continue
            episode_index = int(match.group(1))
            if episode_index in mapping and mapping[episode_index] != root:
                raise ValueError(
                    f"Episode {episode_index} exists under multiple transition_scenes roots: "
                    f"{mapping[episode_index]} and {root}"
                )
            mapping[episode_index] = root
    return mapping


def build_visualization_examples(
    episodes: dict[int, dict[str, Any]],
    *,
    episode_to_root: dict[int, Path],
) -> tuple[dict[int, list[VisualizationExample]], dict[int, list[VisualizationExample]]]:
    by_episode: dict[int, list[VisualizationExample]] = {}
    transition_only_by_episode: dict[int, list[VisualizationExample]] = {}

    for episode_index in sorted(episodes):
        episode = episodes[episode_index]
        canonical_segments = canonical_segments_from_episode(episode)
        if episode_index not in episode_to_root:
            raise FileNotFoundError(f"Could not find saved transition scenes for episode {episode_index}.")

        plan = resolve_episode_plan(episode)
        instruction = extract_instruction(episode)
        transition_paths = transition_scene_paths(
            episode_to_root[episode_index],
            episode_index=episode_index,
            segments=canonical_segments,
        )
        if not transition_paths:
            raise ValueError(f"Episode {episode_index} has no transition scene paths.")

        initial_image_path = transition_paths[0]
        if not initial_image_path.is_file():
            raise FileNotFoundError(f"Missing initial-scene image for episode {episode_index}: {initial_image_path}")

        episode_examples = [
            VisualizationExample(
                episode_index=episode_index,
                view_kind="initial_scene",
                transition_index=0,
                segment_index=0,
                boundary_step=0,
                instruction=instruction,
                plan=plan,
                image_path=initial_image_path,
                skill_after=str(canonical_segments[0]["skill"]),
            )
        ]

        transition_examples: list[VisualizationExample] = []
        for transition_index in range(1, len(canonical_segments)):
            image_path = transition_paths[transition_index]
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Missing transition image for episode {episode_index}, transition {transition_index}: {image_path}"
                )
            example = VisualizationExample(
                episode_index=episode_index,
                view_kind="skill_transition",
                transition_index=transition_index,
                segment_index=transition_index,
                boundary_step=int(canonical_segments[transition_index]["start_step"]),
                instruction=instruction,
                plan=plan,
                image_path=image_path,
                skill_before=str(canonical_segments[transition_index - 1]["skill"]),
                skill_after=str(canonical_segments[transition_index]["skill"]),
            )
            episode_examples.append(example)
            transition_examples.append(example)

        by_episode[episode_index] = episode_examples
        if transition_examples:
            transition_only_by_episode[episode_index] = transition_examples

    if not by_episode:
        raise ValueError("No episode visualizations were found in the provided run directory.")

    return by_episode, transition_only_by_episode


def load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for font_name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)

    wrapped: list[str] = []
    for line in lines:
        if draw.textlength(line, font=font) <= max_width:
            wrapped.append(line)
            continue
        wrapped.extend(textwrap.wrap(line, width=max(10, len(line) // 2)))
    return wrapped or [text]


def overlay_lines(example: VisualizationExample) -> list[str]:
    if example.view_kind == "initial_scene":
        task_prompt = example.instruction if example.instruction is not None else "(instruction unavailable)"
        return [
            f"Task: {task_prompt}",
            f"Plan: {example.plan}",
        ]
    return [
        f"{example.skill_before}",
        f"{example.skill_after}",
    ]


def render_overlay(example: VisualizationExample, output_path: Path) -> None:
    with Image.open(example.image_path) as source_image:
        image = source_image.convert("RGBA")

    width, _ = image.size
    font_size = max(10, min(34, width // 36))
    font = load_font(font_size)
    draw = ImageDraw.Draw(image)
    max_text_width = max(200, width - 48)

    lines: list[str] = []
    for raw_line in overlay_lines(example):
        lines.extend(wrap_text(draw, raw_line, font, max_text_width))

    line_heights: list[int] = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])
    spacing = max(6, font_size // 5)
    block_height = sum(line_heights) + spacing * max(0, len(lines) - 1)
    padding_x = 18
    padding_y = 14
    rectangle_height = block_height + 2 * padding_y

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        [(10, 10), (width - 10, 10 + rectangle_height)],
        radius=14,
        fill=(0, 0, 0, 175),
    )
    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)

    cursor_y = 10 + padding_y
    for line, line_height in zip(lines, line_heights):
        draw.text((10 + padding_x, cursor_y), line, fill=(255, 255, 255, 255), font=font)
        cursor_y += line_height + spacing

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def print_example(example: VisualizationExample, output_path: Path) -> None:
    print(f"view_kind: {example.view_kind}", flush=True)
    print(f"episode_index: {example.episode_index}", flush=True)
    print(f"segment_index: {example.segment_index}", flush=True)
    print(f"transition_index: {example.transition_index}", flush=True)
    print(f"boundary_step: {example.boundary_step}", flush=True)
    if example.view_kind == "initial_scene":
        task_prompt = example.instruction if example.instruction is not None else "(instruction unavailable)"
        print(f"task_prompt: {task_prompt}", flush=True)
        print(f"first_skill: {example.skill_after}", flush=True)
    else:
        print(f"skill_before: {example.skill_before}", flush=True)
        print(f"skill_after: {example.skill_after}", flush=True)
    print(f"plan: {example.plan}", flush=True)
    print(f"source_image: {example.image_path}", flush=True)
    print(f"output_image: {output_path}", flush=True)
    print(flush=True)


def prompt_for_next() -> str:
    response = input("Press Enter or 'n' for next, or 'q' to quit: ").strip().lower()
    if response == "q":
        return "quit"
    return "next"


def run_examples(examples: list[VisualizationExample], output_path: Path) -> bool:
    for example in examples:
        render_overlay(example, output_path)
        print_example(example, output_path)
        if prompt_for_next() == "quit":
            return False
    return True


def run_sequential_mode(
    examples_by_episode: dict[int, list[VisualizationExample]],
    output_path: Path,
    start_episode: int,
) -> int:
    for episode_index in sorted(idx for idx in examples_by_episode if idx >= start_episode):
        if not run_examples(examples_by_episode[episode_index], output_path):
            return 0

    print("Reached the end of the sequential visualization list.", flush=True)
    return 0


def run_random_mode(
    transition_examples_by_episode: dict[int, list[VisualizationExample]],
    output_path: Path,
    start_examples: list[VisualizationExample] | None = None,
) -> int:
    if start_examples is not None and not run_examples(start_examples, output_path):
        return 0

    if not transition_examples_by_episode:
        print("No skill transitions are available for random sampling.", flush=True)
        return 0

    rng = random.Random()
    episode_indices = sorted(transition_examples_by_episode)

    while True:
        episode_index = rng.choice(episode_indices)
        example = rng.choice(transition_examples_by_episode[episode_index])
        render_overlay(example, output_path)
        print_example(example, output_path)
        if prompt_for_next() == "quit":
            return 0


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run directory does not exist: {run_dir}")

    episodes, annotation_path = load_combined_episodes(run_dir)
    if args.episode is not None and args.episode not in episodes:
        raise ValueError(f"Episode {args.episode} was not found in {annotation_path}")

    transition_scene_roots = discover_transition_scene_roots(run_dir)
    if not transition_scene_roots:
        raise FileNotFoundError(f"Could not find any transition_scenes directories under {run_dir}")

    episode_to_root = build_episode_transition_root_map(transition_scene_roots)
    examples_by_episode, transition_examples_by_episode = build_visualization_examples(
        episodes,
        episode_to_root=episode_to_root,
    )
    output_path = Path.cwd() / OUTPUT_IMAGE_NAME

    total_visualizations = sum(len(items) for items in examples_by_episode.values())
    total_transitions = sum(len(items) for items in transition_examples_by_episode.values())

    print(f"annotation_file: {annotation_path}", flush=True)
    print(f"transition_scene_roots: {len(transition_scene_roots)}", flush=True)
    print(f"episodes_with_visualizations: {len(examples_by_episode)}", flush=True)
    print(f"episodes_with_transitions: {len(transition_examples_by_episode)}", flush=True)
    print(f"total_visualizations: {total_visualizations}", flush=True)
    print(f"total_transitions: {total_transitions}", flush=True)
    print(f"mode: {'sequential' if args.random_order else 'random'}", flush=True)
    if args.episode is not None:
        print(f"start_episode: {args.episode}", flush=True)
    print(flush=True)

    if args.random_order:
        start_examples = examples_by_episode[args.episode] if args.episode is not None else None
        return run_random_mode(transition_examples_by_episode, output_path, start_examples=start_examples)
    
    start_episode = args.episode if args.episode is not None else min(examples_by_episode)
    return run_sequential_mode(examples_by_episode, output_path, start_episode)

if __name__ == "__main__":
    raise SystemExit(main())
