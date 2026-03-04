#!/usr/bin/env python3
"""Ad-hoc post-processing patch for cot_skill JSON schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PLAN_MARKER = "Plan:"
DONE_MARKER = "What I have done:"
NOW_MARKER = "Now I need to do:"
STEP_RE = __import__("re").compile(r"^\s*(\d+)\.\s*")


def is_index_key(key: str) -> bool:
    return key.isdigit()


def parse_section(text: str, start_marker: str, end_marker: str | None = None) -> str:
    s_idx = text.find(start_marker)
    if s_idx == -1:
        raise ValueError(f"Missing marker: {start_marker}")
    s_idx += len(start_marker)
    if end_marker is None:
        return text[s_idx:].strip()
    e_idx = text.find(end_marker, s_idx)
    if e_idx == -1:
        raise ValueError(f"Missing marker: {end_marker}")
    return text[s_idx:e_idx].strip()


def strip_step_prefix(line: str) -> str:
    return STEP_RE.sub("", line.strip())


def parse_items(section_text: str) -> list[str]:
    lines = [line.strip() for line in section_text.splitlines() if line.strip()]
    return [strip_step_prefix(line) for line in lines]


def now_item(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Expected string text for now-item extraction.")
    now_items = parse_items(parse_section(text, NOW_MARKER, None))
    if len(now_items) != 1:
        raise ValueError(f"'Now I need to do' must have exactly one item, got {len(now_items)}")
    return now_items[0]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object.")
    return data


def save_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Fix cot_skill schema by replacing current_skill with skill/updated_skill and "
            "writing a new JSON file."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_dir / "cot_skill.json",
        help="Input cot_skill JSON file (default: ./cot_skill.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dir / "cot_skill_fixed.json",
        help="Output fixed JSON file (default: ./cot_skill_fixed.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    data = load_json(args.input)
    index_keys = sorted((k for k in data.keys() if is_index_key(k)), key=lambda x: int(x))

    seg_count = 0
    updated_skill_count = 0
    removed_current_skill = 0

    for idx in index_keys:
        episode = data[idx]
        if not isinstance(episode, dict):
            raise ValueError(f"[idx={idx}] episode is not an object.")
        segments = episode.get("segments")
        if not isinstance(segments, list):
            raise ValueError(f"[idx={idx}] missing/invalid segments list.")

        for seg_i, seg in enumerate(segments):
            seg_count += 1
            if not isinstance(seg, dict):
                raise ValueError(f"[idx={idx} seg={seg_i}] segment is not an object.")

            if "current_skill" in seg:
                seg.pop("current_skill", None)
                removed_current_skill += 1

            content = seg.get("content")
            if not isinstance(content, str):
                raise ValueError(f"[idx={idx} seg={seg_i}] missing/non-string content.")
            seg["skill"] = now_item(content)

            updated_content = seg.get("updated_content")
            if updated_content is None:
                seg.pop("updated_skill", None)
                continue
            if not isinstance(updated_content, str):
                raise ValueError(
                    f"[idx={idx} seg={seg_i}] updated_content must be string or null."
                )

            updated_now = now_item(updated_content)
            updated_w_instruction = seg.get("updated_content_w_instruction")
            if not isinstance(updated_w_instruction, str):
                raise ValueError(
                    f"[idx={idx} seg={seg_i}] updated_content exists but "
                    "updated_content_w_instruction is missing/non-string."
                )
            updated_w_instruction_now = now_item(updated_w_instruction)
            if updated_now != updated_w_instruction_now:
                raise ValueError(
                    f"[idx={idx} seg={seg_i}] updated_content and updated_content_w_instruction "
                    "have different Now I need to do values."
                )

            seg["updated_skill"] = updated_now
            updated_skill_count += 1

    save_json_atomic(args.output, data)
    print(
        f"Fixed schema for {len(index_keys)} index/indices, {seg_count} segment(s). "
        f"Removed current_skill from {removed_current_skill} segment(s). "
        f"Set updated_skill on {updated_skill_count} segment(s). "
        f"Output: {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
