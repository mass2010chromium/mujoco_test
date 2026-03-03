#!/usr/bin/env python3
"""Validate cot_skill.json against cot_simple.json conversion rules."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PLAN_MARKER = "Plan:"
DONE_MARKER = "What I have done:"
NOW_MARKER = "Now I need to do:"
INSTRUCTION_MARKER = "Instruction:"

TIME_KEYS = {
    "start_step",
    "end_step",
    "reference_start_step",
    "reference_end_step",
    "outdated_reference_start_step",
    "outdated_reference_end_step",
}

SKILL_RE = re.compile(r"^(PICK|PLACE|OPEN|CLOSE|ROTATE|GRASP|RELEASE)\((.*)\)$")
STEP_RE = re.compile(r"^\s*(\d+)\.\s*")


def is_index_key(key: str) -> bool:
    return key.isdigit()


def strip_step_prefix(line: str) -> str:
    return STEP_RE.sub("", line.strip())


def is_placeholder(text: str) -> bool:
    return text.strip().rstrip(".").strip().lower() in {"tbd", "nothing"}


def is_skill(text: str) -> bool:
    m = SKILL_RE.match(text.strip())
    if not m:
        return False
    name = m.group(1)
    args_raw = m.group(2)
    args = [a.strip() for a in args_raw.split(",")]
    if name == "PLACE":
        return len(args) == 3 and all(args)
    return len(args) == 1 and bool(args[0])


def has_plan_structure(text: str) -> bool:
    return PLAN_MARKER in text and DONE_MARKER in text and NOW_MARKER in text


def parse_instruction_prefix(text: str) -> str | None:
    inst_idx = text.find(INSTRUCTION_MARKER)
    plan_idx = text.find(PLAN_MARKER)
    if inst_idx == -1 or plan_idx == -1 or inst_idx > plan_idx:
        return None
    return text[:plan_idx]


def parse_plan_tail(text: str) -> str | None:
    plan_idx = text.find(PLAN_MARKER)
    if plan_idx == -1:
        return None
    return text[plan_idx:]


def parse_section(text: str, start: str, end: str | None = None) -> str:
    s_idx = text.find(start)
    if s_idx == -1:
        raise ValueError(f"Missing marker: {start}")
    s_idx += len(start)
    if end is None:
        return text[s_idx:].strip()
    e_idx = text.find(end, s_idx)
    if e_idx == -1:
        raise ValueError(f"Missing marker: {end}")
    return text[s_idx:e_idx].strip()


def parse_items(section_text: str) -> list[str]:
    lines = [line.strip() for line in section_text.splitlines() if line.strip()]
    return [strip_step_prefix(line) for line in lines]


def parse_all_action_items(text: str) -> list[str]:
    """Return all action items from Plan/Done/Now sections in order."""
    plan_items = parse_items(parse_section(text, PLAN_MARKER, DONE_MARKER))
    done_items = parse_items(parse_section(text, DONE_MARKER, NOW_MARKER))
    now_items = parse_items(parse_section(text, NOW_MARKER, None))
    return plan_items + done_items + now_items


def now_item(text: str) -> str:
    items = parse_items(parse_section(text, NOW_MARKER, None))
    if not items:
        raise ValueError("Empty 'Now I need to do' section.")
    if len(items) > 1:
        raise ValueError("'Now I need to do' must contain a single item.")
    return items[0]


def ensure_field_skill_rules(
    original_text: str,
    translated_text: str,
    field_name: str,
    failures: list[str],
) -> None:
    if not isinstance(translated_text, str):
        failures.append(f"{field_name}: translated value is not string.")
        return
    if not has_plan_structure(translated_text):
        failures.append(f"{field_name}: missing Plan/Done/Now markers.")
        return

    orig_instruction = parse_instruction_prefix(original_text)
    if orig_instruction is not None:
        trans_instruction = parse_instruction_prefix(translated_text)
        if trans_instruction != orig_instruction:
            failures.append(f"{field_name}: Instruction changed.")

    try:
        plan_items = parse_items(parse_section(translated_text, PLAN_MARKER, DONE_MARKER))
        done_items = parse_items(parse_section(translated_text, DONE_MARKER, NOW_MARKER))
        now = now_item(translated_text)
    except ValueError as exc:
        failures.append(f"{field_name}: section parse error: {exc}")
        return

    for item in plan_items:
        if not (is_placeholder(item) or is_skill(item)):
            failures.append(f"{field_name}: invalid plan item '{item}'")
    for item in done_items:
        if not (is_placeholder(item) or is_skill(item)):
            failures.append(f"{field_name}: invalid done item '{item}'")
    if not (is_placeholder(now) or is_skill(now)):
        failures.append(f"{field_name}: invalid now item '{now}'")

    plan_skills = {item for item in plan_items if is_skill(item)}
    done_skills = {item for item in done_items if is_skill(item)}
    if not done_skills.issubset(plan_skills):
        failures.append(f"{field_name}: done skills are not subset of plan skills.")
    if is_skill(now) and plan_skills and now not in plan_skills:
        failures.append(f"{field_name}: now skill is not in plan.")


def validate_episode(
    idx: str,
    original_episode: dict[str, Any],
    skill_episode: dict[str, Any],
    failures: list[str],
) -> None:
    # Non-segment episode keys should remain untouched.
    for key, orig_val in original_episode.items():
        if key == "segments":
            continue
        if key not in skill_episode:
            failures.append(f"[idx={idx}] missing episode key: {key}")
            continue
        if skill_episode[key] != orig_val:
            failures.append(f"[idx={idx}] episode key changed: {key}")

    if "segments" not in skill_episode or not isinstance(skill_episode["segments"], list):
        failures.append(f"[idx={idx}] missing/invalid segments list.")
        return
    if not isinstance(original_episode.get("segments"), list):
        failures.append(f"[idx={idx}] original segments malformed.")
        return

    original_segments = original_episode["segments"]
    skill_segments = skill_episode["segments"]
    if len(original_segments) != len(skill_segments):
        failures.append(
            f"[idx={idx}] segment length mismatch: {len(original_segments)} vs {len(skill_segments)}"
        )
        return

    for s_i, (orig_seg, skill_seg) in enumerate(zip(original_segments, skill_segments)):
        prefix = f"[idx={idx} seg={s_i}]"
        if not isinstance(orig_seg, dict) or not isinstance(skill_seg, dict):
            failures.append(f"{prefix} non-object segment encountered.")
            continue

        # All original keys must remain present.
        for key in orig_seg.keys():
            if key not in skill_seg:
                failures.append(f"{prefix} missing key: {key}")

        # Timestep and all non-translated fields must remain exactly unchanged.
        translatable_fields = {
            key
            for key, value in orig_seg.items()
            if isinstance(value, str)
            and PLAN_MARKER in value
            and DONE_MARKER in value
            and NOW_MARKER in value
        }
        for key, orig_val in orig_seg.items():
            if key in translatable_fields:
                continue
            if key == "updated_content" and orig_val is None:
                # Null updated_content is expected unchanged.
                if skill_seg.get(key, None) is not None:
                    failures.append(f"{prefix} updated_content should remain null.")
                continue
            if skill_seg.get(key) != orig_val:
                failures.append(f"{prefix} unchanged field modified: {key}")

        for key in TIME_KEYS:
            if key in orig_seg and skill_seg.get(key) != orig_seg[key]:
                failures.append(f"{prefix} timestep field changed: {key}")

        # Validate translated string fields.
        for key in translatable_fields:
            ensure_field_skill_rules(orig_seg[key], skill_seg.get(key), f"{prefix} {key}", failures)

        # Validate updated_content_w_instruction correspondence when present.
        if "updated_content_w_instruction" in orig_seg:
            if "updated_content_w_instruction" not in skill_seg:
                failures.append(f"{prefix} missing updated_content_w_instruction.")
            elif orig_seg["updated_content_w_instruction"] is None:
                if skill_seg["updated_content_w_instruction"] is not None:
                    failures.append(f"{prefix} updated_content_w_instruction nullness changed.")
            else:
                if skill_seg.get("updated_content") is None:
                    failures.append(f"{prefix} updated_content_w_instruction exists but updated_content null.")
                else:
                    instruction = parse_instruction_prefix(skill_seg.get("content", ""))
                    updated_tail = parse_plan_tail(skill_seg.get("updated_content", ""))
                    if instruction is None or updated_tail is None:
                        failures.append(f"{prefix} cannot parse instruction/tail for correspondence check.")
                    else:
                        expected = instruction + updated_tail
                        if skill_seg.get("updated_content_w_instruction") != expected:
                            failures.append(
                                f"{prefix} updated_content_w_instruction does not equal "
                                "instruction + updated_content tail."
                            )

        # Validate skill field (must come from content's Now section).
        if "skill" not in skill_seg:
            failures.append(f"{prefix} missing skill.")
        else:
            skill_value = skill_seg.get("skill")
            if not isinstance(skill_value, str):
                failures.append(f"{prefix} skill must be string, got {type(skill_value).__name__}.")
            elif not isinstance(skill_seg.get("content"), str):
                failures.append(f"{prefix} cannot validate skill: content is missing/non-string.")
            else:
                skill_items = [line.strip() for line in skill_value.splitlines() if line.strip()]
                if len(skill_items) != 1:
                    failures.append(
                        f"{prefix} skill must contain exactly one item, got {len(skill_items)}."
                    )
                try:
                    expected_from_content = now_item(skill_seg["content"])
                    if skill_value != expected_from_content:
                        failures.append(
                            f"{prefix} skill mismatch: expected '{expected_from_content}', "
                            f"got '{skill_value}'"
                        )
                except ValueError as exc:
                    failures.append(f"{prefix} cannot parse now item from content: {exc}")

                if not (is_placeholder(skill_value) or is_skill(skill_value)):
                    failures.append(f"{prefix} skill is not a valid skill/placeholder.")

        # Validate updated_skill field (must come from updated_content if it exists).
        updated_content = skill_seg.get("updated_content")
        if updated_content is None:
            if "updated_skill" in skill_seg and skill_seg["updated_skill"] is not None:
                failures.append(
                    f"{prefix} updated_content is null but updated_skill is present/non-null."
                )
        else:
            if not isinstance(updated_content, str):
                failures.append(
                    f"{prefix} updated_content must be string or null, got {type(updated_content).__name__}."
                )
            if "updated_skill" not in skill_seg:
                failures.append(f"{prefix} missing updated_skill while updated_content exists.")
            else:
                updated_skill = skill_seg.get("updated_skill")
                if not isinstance(updated_skill, str):
                    failures.append(
                        f"{prefix} updated_skill must be string, got {type(updated_skill).__name__}."
                    )
                else:
                    updated_skill_items = [
                        line.strip() for line in updated_skill.splitlines() if line.strip()
                    ]
                    if len(updated_skill_items) != 1:
                        failures.append(
                            f"{prefix} updated_skill must contain exactly one item, "
                            f"got {len(updated_skill_items)}."
                        )
                    try:
                        expected_from_updated = now_item(updated_content)
                        if updated_skill != expected_from_updated:
                            failures.append(
                                f"{prefix} updated_skill mismatch with updated_content now item: "
                                f"expected '{expected_from_updated}', got '{updated_skill}'"
                            )
                    except ValueError as exc:
                        failures.append(f"{prefix} cannot parse now item from updated_content: {exc}")

                    updated_w_inst = skill_seg.get("updated_content_w_instruction")
                    if not isinstance(updated_w_inst, str):
                        failures.append(
                            f"{prefix} updated_content exists but updated_content_w_instruction is "
                            "missing/non-string."
                        )
                    else:
                        try:
                            expected_from_updated_w_inst = now_item(updated_w_inst)
                            if updated_skill != expected_from_updated_w_inst:
                                failures.append(
                                    f"{prefix} updated_skill mismatch with "
                                    f"updated_content_w_instruction now item: "
                                    f"expected '{expected_from_updated_w_inst}', got '{updated_skill}'"
                                )
                        except ValueError as exc:
                            failures.append(
                                f"{prefix} cannot parse now item from updated_content_w_instruction: {exc}"
                            )

                    if not (is_placeholder(updated_skill) or is_skill(updated_skill)):
                        failures.append(f"{prefix} updated_skill is not a valid skill/placeholder.")


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Validate cot_skill.json conversion quality and invariants."
    )
    parser.add_argument(
        "--orig",
        type=Path,
        default=default_dir / "cot_simple.json",
        help="Path to original cot_simple.json",
    )
    parser.add_argument(
        "--skill",
        type=Path,
        default=default_dir / "cot_skill_fixed.json",
        help="Path to generated cot_skill.json",
    )
    parser.add_argument(
        "--strict-skill-only",
        action="store_true",
        help=(
            "Require every action item in Plan/Done/Now to be a skill expression "
            "(disallow placeholders like TBD./Nothing.)."
        ),
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=200,
        help="Max number of error lines to print (default: 200).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.orig.is_file():
        print(f"ERROR: original JSON missing: {args.orig}", file=sys.stderr)
        return 2
    if not args.skill.is_file():
        print(f"ERROR: skill JSON missing: {args.skill}", file=sys.stderr)
        return 2

    with args.orig.open("r", encoding="utf-8") as f:
        original = json.load(f)
    with args.skill.open("r", encoding="utf-8") as f:
        skill = json.load(f)

    failures: list[str] = []
    warnings: list[str] = []

    # Non-index top-level keys in output should match original.
    original_non_index = {k: v for k, v in original.items() if not is_index_key(k)}
    skill_non_index = {k: v for k, v in skill.items() if not is_index_key(k)}
    for key, value in original_non_index.items():
        if key not in skill_non_index:
            failures.append(f"Missing top-level non-index key: {key}")
        elif skill_non_index[key] != value:
            failures.append(f"Top-level non-index key changed: {key}")

    # Validate all indices that exist in the generated skill file.
    skill_indices = sorted((k for k in skill.keys() if is_index_key(k)), key=lambda x: int(x))
    if not skill_indices:
        failures.append("[indices_present] No numeric indices found in cot_skill.json.")
    else:
        idx_ints = [int(k) for k in skill_indices]
        if len(set(idx_ints)) != len(idx_ints):
            failures.append("[indices_non_overlapping] Duplicate numeric indices found.")
        expected = list(range(idx_ints[0], idx_ints[-1] + 1))
        if idx_ints != expected:
            missing = sorted(set(expected) - set(idx_ints))
            preview = missing[:30]
            suffix = "" if len(missing) <= 30 else f" ... and {len(missing) - 30} more"
            failures.append(
                "[indices_no_gaps] Missing indices between "
                f"{idx_ints[0]} and {idx_ints[-1]}: {preview}{suffix}"
            )

    for idx in skill_indices:
        if idx not in original:
            failures.append(f"Generated index not found in original: {idx}")
            continue
        validate_episode(idx, original[idx], skill[idx], failures)

        # Additional per-segment tests from the newer validator.
        episode = skill[idx]
        if not isinstance(episode, dict):
            continue
        segments = episode.get("segments")
        if not isinstance(segments, list):
            failures.append(f"[segment_schema] [idx={idx}] missing or invalid segments list.")
            continue

        for seg_i, seg in enumerate(segments):
            prefix = f"[idx={idx} seg={seg_i}]"
            if not isinstance(seg, dict):
                failures.append(f"[segment_schema] {prefix} segment is not an object.")
                continue

            # Required field/type checks.
            for req_key in ("start_step", "end_step", "reference_start_step", "reference_end_step"):
                if req_key not in seg:
                    failures.append(f"[segment_required_fields] {prefix} missing key '{req_key}'.")
                elif not isinstance(seg[req_key], int):
                    failures.append(
                        f"[segment_required_fields] {prefix} key '{req_key}' must be int, "
                        f"got {type(seg[req_key]).__name__}."
                    )

            # Timestep order sanity checks (allow end_step == -1 sentinel).
            if (
                isinstance(seg.get("start_step"), int)
                and isinstance(seg.get("end_step"), int)
                and seg["end_step"] != -1
                and seg["start_step"] > seg["end_step"]
            ):
                warnings.append(
                    f"[timestep_order] {prefix} start_step ({seg['start_step']}) > "
                    f"end_step ({seg['end_step']})."
                )
            if (
                isinstance(seg.get("reference_start_step"), int)
                and isinstance(seg.get("reference_end_step"), int)
                and seg["reference_start_step"] > seg["reference_end_step"]
            ):
                warnings.append(
                    f"[timestep_order] {prefix} reference_start_step ({seg['reference_start_step']}) > "
                    f"reference_end_step ({seg['reference_end_step']})."
                )

            # skill should be exactly one item.
            base_skill = seg.get("skill")
            if isinstance(base_skill, str):
                base_skill_items = [line.strip() for line in base_skill.splitlines() if line.strip()]
                if len(base_skill_items) != 1:
                    failures.append(
                        f"[skill_single_skill] {prefix} skill should contain "
                        f"exactly one skill, got {len(base_skill_items)}."
                    )

            # updated_skill should be exactly one item when present.
            if "updated_skill" in seg and isinstance(seg.get("updated_skill"), str):
                upd_skill_items = [
                    line.strip() for line in seg["updated_skill"].splitlines() if line.strip()
                ]
                if len(upd_skill_items) != 1:
                    failures.append(
                        f"[updated_skill_single_skill] {prefix} updated_skill should contain "
                        f"exactly one skill, got {len(upd_skill_items)}."
                    )

            # Optional strict test: every action item in plan fields must be skill expression.
            if args.strict_skill_only:
                for field in ("content", "updated_content", "updated_content_w_instruction"):
                    value = seg.get(field)
                    if not isinstance(value, str) or not has_plan_structure(value):
                        continue
                    try:
                        all_items = parse_all_action_items(value)
                    except ValueError as exc:
                        failures.append(
                            f"[strict_skill_only] {prefix} {field} parse error during strict skill check: {exc}"
                        )
                        continue
                    for item in all_items:
                        if not is_skill(item):
                            failures.append(
                                f"[strict_skill_only] {prefix} {field} has non-skill item '{item}'."
                            )
                for skill_field in ("skill", "updated_skill"):
                    sval = seg.get(skill_field)
                    if sval is None:
                        continue
                    if not isinstance(sval, str):
                        failures.append(
                            f"[strict_skill_only] {prefix} {skill_field} is non-string "
                            f"({type(sval).__name__})."
                        )
                    elif not is_skill(sval):
                        failures.append(
                            f"[strict_skill_only] {prefix} {skill_field} has non-skill item '{sval}'."
                        )

    if warnings:
        print(f"Validation WARNINGS with {len(warnings)} issue(s):")
        for msg in warnings[: args.max_errors]:
            print(f"- {msg}")
        if len(warnings) > args.max_errors:
            print(f"... and {len(warnings) - args.max_errors} more.")
        print("")

    if failures:
        print(f"Validation FAILED with {len(failures)} issue(s):")
        for msg in failures[: args.max_errors]:
            print(f"- {msg}")
        if len(failures) > args.max_errors:
            print(f"... and {len(failures) - args.max_errors} more.")
        return 1

    print(
        "Validation PASSED.\n"
        f"Checked {len(skill_indices)} generated index/indices against {args.orig.name}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
