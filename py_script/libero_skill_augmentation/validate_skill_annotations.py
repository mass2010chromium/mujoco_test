#!/usr/bin/env python3
"""
Structural validator for LIBERO skill-annotation JSON files.

This script checks:
- that each episode has a recoverable numbered plan
- that segment skill sequences agree with that plan
- that individual skill expressions are syntactically valid
- that segment-level plan / skill references stay consistent with the episode plan
- that segment text fields do not mention stray skills outside the plan
- that the first planned skill is not one of the forbidden skill types that cannot start an episode
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    DEFAULT_REPO_ID,
    list_episode_shards,
    load_json,
    parse_plan_skills,
    validate_skill_expr,
)


SKILL_NAMES = [
    "PLACE_ON",
    "PLACE_IN",
    "PICKUP_FROM",
    "OPEN",
    "CLOSE",
    "TURN_ON",
    "TURN_OFF",
]

FORBIDDEN_FIRST_SKILLS = {
    "PLACE_ON",
    "PLACE_IN",
}


@dataclass
class ValidationFailure:
    source: Path
    episode_index: int | None
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate skill-annotation JSON files by checking plan recoverability, segment-plan consistency, "
            "skill-expression validity, stray skill mentions, and basic skill-order constraints."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Annotation file(s) or directory/directories containing episode shard files.",
    )
    parser.add_argument(
        "--stop-on-first-error",
        action="store_true",
        help="Exit immediately after the first validation failure.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final summary unless a failure is found.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Unused for validation logic, retained for consistency with the rest of the tooling.",
    )
    return parser.parse_args()


def iter_skill_mentions(text: str) -> list[str]:
    mentions: list[str] = []
    for name in SKILL_NAMES:
        start = 0
        marker = f"{name}("
        while True:
            idx = text.find(marker, start)
            if idx == -1:
                break
            depth = 0
            end = None
            for pos in range(idx, len(text)):
                char = text[pos]
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        end = pos + 1
                        break
            if end is None:
                break
            candidate = text[idx:end]
            try:
                mentions.append(validate_skill_expr(candidate))
            except ValueError:
                pass
            start = end
    return mentions


def compressed_segment_skills(segments: list[dict[str, Any]]) -> list[str]:
    compressed: list[str] = []
    for segment in segments:
        skill = validate_skill_expr(str(segment["skill"]))
        if not compressed or compressed[-1] != skill:
            compressed.append(skill)
    return compressed


def resolve_episode_plan(episode: dict[str, Any]) -> str:
    if "plan" in episode and isinstance(episode["plan"], str):
        return episode["plan"]

    segments = episode.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Episode has no segments to recover a plan from.")

    segment_plans = [segment.get("plan") for segment in segments if isinstance(segment.get("plan"), str)]
    if not segment_plans:
        raise ValueError("Episode has no top-level plan and no segment-level plan strings.")

    first_plan = segment_plans[0]
    for idx, segment_plan in enumerate(segment_plans[1:], start=1):
        if parse_plan_skills(segment_plan) != parse_plan_skills(first_plan):
            raise ValueError(f"Segment plan strings disagree within the same trajectory at segment-plan index {idx}.")
    return first_plan


def validate_episode(episode: dict[str, Any], source: Path) -> list[ValidationFailure]:
    failures: list[ValidationFailure] = []
    episode_index = episode.get("episode_index")
    segments = episode.get("segments")
    if not isinstance(segments, list) or not segments:
        return [ValidationFailure(source, int(episode_index) if isinstance(episode_index, int) else None, "Missing or empty segments list.")]

    try:
        plan_string = resolve_episode_plan(episode)
        plan_skills = parse_plan_skills(plan_string)
    except Exception as exc:
        return [ValidationFailure(source, int(episode_index) if isinstance(episode_index, int) else None, f"Could not parse plan: {exc}")]

    first_skill_name = plan_skills[0].split("(", 1)[0]
    if first_skill_name in FORBIDDEN_FIRST_SKILLS:
        failures.append(
            ValidationFailure(
                source,
                int(episode_index) if isinstance(episode_index, int) else None,
                "The first planned skill cannot be one of "
                f"{sorted(FORBIDDEN_FIRST_SKILLS)}; got {plan_skills[0]!r}.",
            )
        )

    compressed_skills = compressed_segment_skills(segments)
    if compressed_skills != plan_skills:
        failures.append(
            ValidationFailure(
                source,
                int(episode_index) if isinstance(episode_index, int) else None,
                "Compressed segment skill sequence does not exactly match the plan. "
                f"plan={plan_skills}, compressed_segments={compressed_skills}",
            )
        )

    for seg_idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            failures.append(
                ValidationFailure(
                    source,
                    int(episode_index) if isinstance(episode_index, int) else None,
                    f"Segment {seg_idx} is not a JSON object.",
                )
            )
            continue

        try:
            skill = validate_skill_expr(str(segment["skill"]))
        except Exception as exc:
            failures.append(
                ValidationFailure(
                    source,
                    int(episode_index) if isinstance(episode_index, int) else None,
                    f"Segment {seg_idx} has invalid skill: {exc}",
                )
            )
            continue

        if skill not in plan_skills:
            failures.append(
                ValidationFailure(
                    source,
                    int(episode_index) if isinstance(episode_index, int) else None,
                    f"Segment {seg_idx} skill {skill!r} is not present in the plan {plan_skills}.",
                )
            )

        if "updated_skill" in segment and segment["updated_skill"] is not None:
            try:
                updated_skill = validate_skill_expr(str(segment["updated_skill"]))
                if updated_skill not in plan_skills:
                    failures.append(
                        ValidationFailure(
                            source,
                            int(episode_index) if isinstance(episode_index, int) else None,
                            f"Segment {seg_idx} updated_skill {updated_skill!r} is not present in the plan {plan_skills}.",
                        )
                    )
            except Exception as exc:
                failures.append(
                    ValidationFailure(
                        source,
                        int(episode_index) if isinstance(episode_index, int) else None,
                        f"Segment {seg_idx} has invalid updated_skill: {exc}",
                    )
                )

        if "plan" in segment and isinstance(segment["plan"], str):
            try:
                segment_plan_skills = parse_plan_skills(segment["plan"])
                if segment_plan_skills != plan_skills:
                    failures.append(
                        ValidationFailure(
                            source,
                            int(episode_index) if isinstance(episode_index, int) else None,
                            f"Segment {seg_idx} plan string does not match episode plan. "
                            f"segment_plan={segment_plan_skills}, episode_plan={plan_skills}",
                        )
                    )
            except Exception as exc:
                failures.append(
                    ValidationFailure(
                        source,
                        int(episode_index) if isinstance(episode_index, int) else None,
                        f"Segment {seg_idx} has invalid plan string: {exc}",
                    )
                )

        for field_name in ["content", "updated_content", "updated_content_w_instruction"]:
            if field_name not in segment or not isinstance(segment[field_name], str):
                continue
            mentioned_skills = iter_skill_mentions(segment[field_name])
            stray_skills = [mentioned for mentioned in mentioned_skills if mentioned not in plan_skills]
            if stray_skills:
                failures.append(
                    ValidationFailure(
                        source,
                        int(episode_index) if isinstance(episode_index, int) else None,
                        f"Segment {seg_idx} field {field_name!r} mentions skill(s) not in the plan: {stray_skills}",
                    )
                )

    return failures


def iter_episodes_from_file(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict) and "episode_index" in data and "segments" in data:
        return [data]

    if isinstance(data, dict):
        episodes: list[dict[str, Any]] = []
        for key in sorted((k for k in data if isinstance(k, str) and k.isdigit()), key=lambda item: int(item)):
            value = data[key]
            if isinstance(value, dict):
                episode = dict(value)
                episode.setdefault("episode_index", int(key))
                episodes.append(episode)
        if episodes:
            return episodes

    raise ValueError(f"Unsupported annotation JSON structure in {path}")


def expand_inputs(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_path in inputs:
        path = input_path.expanduser().resolve()
        if path.is_dir():
            files.extend(list_episode_shards(path))
        else:
            files.append(path)
    return files


def main() -> int:
    args = parse_args()
    files = expand_inputs(args.inputs)
    if not files:
        raise FileNotFoundError("No annotation files found in the provided inputs.")

    total_files = 0
    total_episodes = 0
    failures: list[ValidationFailure] = []

    for path in files:
        total_files += 1
        episodes = iter_episodes_from_file(path)
        if not args.quiet:
            print(f"checking {path} ({len(episodes)} episode(s))", flush=True)
        for episode in episodes:
            total_episodes += 1
            episode_failures = validate_episode(episode, path)
            if episode_failures:
                failures.extend(episode_failures)
                for failure in episode_failures:
                    prefix = f"{failure.source}"
                    if failure.episode_index is not None:
                        prefix += f" episode={failure.episode_index}"
                    print(f"[FAIL] {prefix}: {failure.message}", flush=True)
                if args.stop_on_first_error:
                    print(f"checked_files={total_files} checked_episodes={total_episodes} failures={len(failures)}", flush=True)
                    return 1

    print(
        f"checked_files={total_files} checked_episodes={total_episodes} failures={len(failures)}",
        flush=True,
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
