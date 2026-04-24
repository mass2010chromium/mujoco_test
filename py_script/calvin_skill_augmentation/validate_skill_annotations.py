#!/usr/bin/env python3
"""
Structural validator for CALVIN skill-annotation JSON files.

This script checks:
- that each episode has a recoverable numbered plan
- that segment skill sequences agree with that plan
- that individual skill expressions are syntactically valid
- that deterministic state-machine constraints implied by the CALVIN skill set hold
- that canonical shard segments carry the requested state/action payloads
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
    parse_skill_expr,
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
    "MOVE_SLIDER",
    "PUSH",
    "PUSH_INTO",
    "TURN_OBJECT",
]

REQUIRES_HOLDING = {"PLACE_ON", "PLACE_IN", "TURN_OBJECT"}
REQUIRES_FREE_GRIPPER = {"PICKUP_FROM", "OPEN", "CLOSE", "TURN_ON", "TURN_OFF", "MOVE_SLIDER", "PUSH", "PUSH_INTO"}


@dataclass
class ValidationFailure:
    source: Path
    episode_index: int | None
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate CALVIN skill-annotation JSON files by checking plan recoverability, segment-plan "
            "consistency, skill-expression validity, canonical state/action payloads, and deterministic "
            "skill-order constraints."
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
    first_plan_skills = parse_plan_skills(first_plan)
    for idx, segment_plan in enumerate(segment_plans[1:], start=1):
        if parse_plan_skills(segment_plan) != first_plan_skills:
            raise ValueError(f"Segment plan strings disagree within the same trajectory at segment-plan index {idx}.")
    return first_plan


def validate_segment_timing(episode: dict[str, Any]) -> list[str]:
    segments = episode.get("segments")
    if not isinstance(segments, list) or not segments:
        return ["Missing or empty segments list."]

    failures: list[str] = []
    prev_end: int | None = None
    for idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            failures.append(f"Segment {idx} is not a JSON object.")
            continue
        try:
            start_step = int(segment["start_step"])
            end_step = int(segment["end_step"])
        except Exception as exc:
            failures.append(f"Segment {idx} is missing integer start/end steps: {exc}")
            continue

        if idx == 0 and start_step != 0:
            failures.append(f"First segment must start at 0, got {start_step}.")
        if prev_end is not None and start_step != prev_end:
            failures.append(
                f"Segments must be contiguous. Segment {idx} starts at {start_step}, expected {prev_end}."
            )
        if end_step <= start_step:
            failures.append(f"Segment {idx} has non-positive length: {segment!r}")
        prev_end = end_step

    if "num_steps" in episode and prev_end is not None:
        try:
            num_steps = int(episode["num_steps"])
            if prev_end != num_steps:
                failures.append(f"Last segment end_step {prev_end} does not match num_steps {num_steps}.")
        except Exception:
            failures.append(f"Episode num_steps is not an integer: {episode.get('num_steps')!r}")

    return failures


def infer_initial_holding(plan_skills: list[str]) -> bool:
    if not plan_skills:
        raise ValueError("Plan is empty.")
    first_name, _ = parse_skill_expr(plan_skills[0])
    return first_name in REQUIRES_HOLDING


def validate_skill_sequence_rules(plan_skills: list[str]) -> list[str]:
    failures: list[str] = []
    holding = infer_initial_holding(plan_skills)
    held_object: str | None = None

    for idx, skill in enumerate(plan_skills):
        name, args = parse_skill_expr(skill)
        human_idx = idx + 1

        if name in REQUIRES_HOLDING and not holding:
            failures.append(f"Plan step {human_idx} {skill!r} requires a grasped object, but the gripper is free.")

        if name in REQUIRES_FREE_GRIPPER and holding:
            failures.append(f"Plan step {human_idx} {skill!r} requires a free gripper, but the robot is already holding an object.")

        if name == "PICKUP_FROM":
            held_object = args[0]
            holding = True
            continue

        if name in {"PLACE_ON", "PLACE_IN"}:
            target = args[0]
            if held_object is not None and target == held_object:
                failures.append(f"Plan step {human_idx} {skill!r} tries to place an object onto or into itself.")
            holding = False
            held_object = None
            continue

        if name == "TURN_OBJECT":
            holding = True
            continue

        holding = False
        held_object = None

    return failures


def validate_canonical_state_action_payloads(episode: dict[str, Any]) -> list[str]:
    if "plan" not in episode or "num_steps" not in episode:
        return []

    failures: list[str] = []
    segments = episode.get("segments", [])
    for idx, segment in enumerate(segments):
        state = segment.get("state")
        action = segment.get("action")
        if not isinstance(state, list):
            failures.append(f"Canonical segment {idx} is missing list-valued 'state'.")
        elif len(state) != 15 or not all(isinstance(value, (int, float)) for value in state):
            failures.append(f"Canonical segment {idx} has invalid state payload; expected 15 numeric values.")

        if not isinstance(action, list):
            failures.append(f"Canonical segment {idx} is missing list-valued 'action'.")
        elif len(action) != 7 or not all(isinstance(value, (int, float)) for value in action):
            failures.append(f"Canonical segment {idx} has invalid action payload; expected 7 numeric values.")
    return failures


def validate_episode(episode: dict[str, Any], source: Path) -> list[ValidationFailure]:
    failures: list[ValidationFailure] = []
    episode_index = episode.get("episode_index")
    segments = episode.get("segments")
    episode_ref = int(episode_index) if isinstance(episode_index, int) else None
    if not isinstance(segments, list) or not segments:
        return [ValidationFailure(source, episode_ref, "Missing or empty segments list.")]

    try:
        plan_string = resolve_episode_plan(episode)
        plan_skills = parse_plan_skills(plan_string)
    except Exception as exc:
        return [ValidationFailure(source, episode_ref, f"Could not parse plan: {exc}")]

    compressed_skills = compressed_segment_skills(segments)
    if compressed_skills != plan_skills:
        failures.append(
            ValidationFailure(
                source,
                episode_ref,
                "Compressed segment skill sequence does not exactly match the plan. "
                f"plan={plan_skills}, compressed_segments={compressed_skills}",
            )
        )

    for message in validate_segment_timing(episode):
        failures.append(ValidationFailure(source, episode_ref, message))

    for message in validate_skill_sequence_rules(plan_skills):
        failures.append(ValidationFailure(source, episode_ref, message))

    for message in validate_canonical_state_action_payloads(episode):
        failures.append(ValidationFailure(source, episode_ref, message))

    for seg_idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            failures.append(ValidationFailure(source, episode_ref, f"Segment {seg_idx} is not a JSON object."))
            continue
        try:
            skill = validate_skill_expr(str(segment["skill"]))
        except Exception as exc:
            failures.append(ValidationFailure(source, episode_ref, f"Segment {seg_idx} has invalid skill: {exc}"))
            continue

        if skill not in plan_skills:
            failures.append(
                ValidationFailure(
                    source,
                    episode_ref,
                    f"Segment {seg_idx} skill {skill!r} is not present in the plan {plan_skills}.",
                )
            )

        if "updated_skill" in segment and segment["updated_skill"] is not None:
            try:
                updated_skill = validate_skill_expr(str(segment["updated_skill"]))
            except Exception as exc:
                failures.append(
                    ValidationFailure(source, episode_ref, f"Segment {seg_idx} has invalid updated_skill: {exc}")
                )
            else:
                if updated_skill not in plan_skills:
                    failures.append(
                        ValidationFailure(
                            source,
                            episode_ref,
                            f"Segment {seg_idx} updated_skill {updated_skill!r} is not present in the plan {plan_skills}.",
                        )
                    )

        if "plan" in segment and isinstance(segment["plan"], str):
            try:
                segment_plan_skills = parse_plan_skills(segment["plan"])
            except Exception as exc:
                failures.append(
                    ValidationFailure(source, episode_ref, f"Segment {seg_idx} has invalid plan string: {exc}")
                )
            else:
                if segment_plan_skills != plan_skills:
                    failures.append(
                        ValidationFailure(
                            source,
                            episode_ref,
                            f"Segment {seg_idx} plan string does not match episode plan. "
                            f"segment_plan={segment_plan_skills}, episode_plan={plan_skills}",
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
            if not episode_failures:
                continue
            failures.extend(episode_failures)
            for failure in episode_failures:
                print(
                    f"[fail] source={failure.source} episode={failure.episode_index} {failure.message}",
                    flush=True,
                )
            if args.stop_on_first_error:
                print(
                    f"summary: files={total_files} episodes={total_episodes} failures={len(failures)}",
                    flush=True,
                )
                return 1

    print(
        f"summary: files={total_files} episodes={total_episodes} failures={len(failures)}",
        flush=True,
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
