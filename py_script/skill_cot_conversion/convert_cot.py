#!/usr/bin/env python3
"""Convert cot_simple.json plans into skill-based plans via OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PLAN_MARKER = "Plan:"
DONE_MARKER = "What I have done:"
NOW_MARKER = "Now I need to do:"
INSTRUCTION_MARKER = "Instruction:"
ALLOWED_SKILLS = {"PICK", "PLACE", "OPEN", "CLOSE", "ROTATE", "GRASP", "RELEASE"}
SKILL_RE = re.compile(r"^(PICK|PLACE|OPEN|CLOSE|ROTATE|GRASP|RELEASE)\((.*)\)$")
SKILL_PREFIX_RE = re.compile(r"^\s*((?:PICK|PLACE|OPEN|CLOSE|ROTATE|GRASP|RELEASE)\([^)]*\))")
STEP_RE = re.compile(r"^\s*(\d+)\.\s*")

SKILL_GUIDE = """\
Allowed skill set and exact syntax:
1) PLACE(object1, object2, preposition)
- object1: the object being placed
- object2: destination/support object
- preposition: relative relation (e.g., inside, on top of)
- Example: PLACE(black bowl, top drawer, inside)

2) PICK(object)
- object: the object being picked
- should be a single object with no commas in the description
- Example: PICK(red and yellow mug)

3) OPEN(object)
- object: the object being opened
- should be a single object with no commas in the description
- Example: OPEN(middle drawer)

4) CLOSE(object)
- object: the object being closed
- should be a single object with no commas in the description
- Example: CLOSE(drawer)

5) ROTATE(object)
- object: the object being rotated
- should be a single object with no commas in the description
- Example: ROTATE(stove knob)

6) GRASP(object)
- object: the object being grasped
- should be a single object with no commas in the description
- Example: GRASP(stove knob)

7) RELEASE(object)
- object: the object being released
- should be a single object with no commas in the description
- Example: RELEASE(stove knob)
"""


def is_index_key(key: str) -> bool:
    return key.isdigit()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_step_prefix(line: str) -> str:
    return STEP_RE.sub("", line.strip())


def split_items(section_text: str) -> list[str]:
    lines = [line.strip() for line in section_text.splitlines() if line.strip()]
    if not lines:
        return []
    return [strip_step_prefix(line) for line in lines]


def section_uses_numbering(section_text: str) -> bool:
    for line in section_text.splitlines():
        if STEP_RE.match(line.strip()):
            return True
    return False


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


def parse_section(text: str, start_marker: str, end_marker: str | None = None) -> str:
    start_idx = text.find(start_marker)
    if start_idx == -1:
        raise ValueError(f"Missing marker '{start_marker}'.")
    start_idx += len(start_marker)
    if end_marker is None:
        return text[start_idx:].strip()
    end_idx = text.find(end_marker, start_idx)
    if end_idx == -1:
        raise ValueError(f"Missing marker '{end_marker}' after '{start_marker}'.")
    return text[start_idx:end_idx].strip()


def canonicalize_structured_text(text: str) -> str:
    if not has_plan_structure(text):
        raise ValueError("Cannot canonicalize text without Plan/Done/Now structure.")

    instruction_prefix = parse_instruction_prefix(text)
    plan_section = parse_section(text, PLAN_MARKER, DONE_MARKER)
    done_section = parse_section(text, DONE_MARKER, NOW_MARKER)
    now_section = parse_section(text, NOW_MARKER, None)

    plan_numbered = section_uses_numbering(plan_section)
    done_numbered = section_uses_numbering(done_section)

    plan_items = [canonicalize_item(item) for item in split_items(plan_section)]
    done_items = [canonicalize_item(item) for item in split_items(done_section)]
    now_items = [canonicalize_item(item) for item in split_items(now_section)]

    if not now_items:
        raise ValueError("Empty 'Now I need to do' section after canonicalization.")
    if len(now_items) > 1:
        print("text", text)
        raise ValueError("Expected single 'Now I need to do' item after canonicalization.")

    out_parts: list[str] = []
    if instruction_prefix is not None:
        out_parts.append(instruction_prefix)
    out_parts.append(f"Plan: {format_items(plan_items, plan_numbered)}\n")
    out_parts.append(f" What I have done: {format_items(done_items, done_numbered)}\n")
    out_parts.append(f"Now I need to do: {now_items[0]}\n")
    return "".join(out_parts)


def is_placeholder(item: str) -> bool:
    norm = item.strip().lower()
    return norm.startswith("tbd") or norm.startswith("nothing")


def canonical_placeholder(item: str) -> str | None:
    norm = item.strip().lower()
    if norm.startswith("tbd"):
        return "TBD."
    if norm.startswith("nothing"):
        return "Nothing."
    return None


def canonicalize_item(item: str) -> str:
    stripped = strip_step_prefix(item)
    placeholder = canonical_placeholder(stripped)
    if placeholder is not None:
        return placeholder
    if is_skill_expr(stripped):
        return stripped
    prefix_match = SKILL_PREFIX_RE.match(stripped)
    if prefix_match:
        return prefix_match.group(1).strip()
    return stripped


def format_items(items: list[str], numbered: bool) -> str:
    if not items:
        return "TBD."
    if numbered:
        return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))
    if len(items) == 1:
        return items[0]
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))


def is_skill_expr(item: str) -> bool:
    match = SKILL_RE.match(item.strip())
    if not match:
        return False
    skill_name, args_raw = match.group(1), match.group(2)
    if skill_name not in ALLOWED_SKILLS:
        return False
    args = [a.strip() for a in args_raw.split(",")]
    if skill_name == "PLACE":
        return len(args) == 3 and all(args)
    return len(args) == 1 and bool(args[0])


def has_plan_structure(text: str) -> bool:
    return PLAN_MARKER in text and DONE_MARKER in text and NOW_MARKER in text


def extract_now_item(text: str) -> str:
    now_section = parse_section(text, NOW_MARKER, None)
    now_items = split_items(now_section)
    if not now_items:
        raise ValueError("Empty 'Now I need to do' section.")
    if len(now_items) > 1:
        raise ValueError("Expected a single item in 'Now I need to do' section.")
    return now_items[0]


def validate_field(original_text: str, translated_text: str, field_name: str) -> None:
    if not has_plan_structure(translated_text):
        raise ValueError(f"{field_name}: missing one or more required markers.")

    orig_prefix = parse_instruction_prefix(original_text)
    out_prefix = parse_instruction_prefix(translated_text)
    if orig_prefix is not None and out_prefix != orig_prefix:
        raise ValueError(f"{field_name}: Instruction part changed.")

    plan_items = split_items(parse_section(translated_text, PLAN_MARKER, DONE_MARKER))
    done_items = split_items(parse_section(translated_text, DONE_MARKER, NOW_MARKER))
    now_item = extract_now_item(translated_text)

    for item in plan_items:
        if not (is_placeholder(item) or is_skill_expr(item)):
            print("translated text", translated_text)
            raise ValueError(f"{field_name}: invalid plan item '{item}'.")
    for item in done_items:
        if not (is_placeholder(item) or is_skill_expr(item)):
            raise ValueError(f"{field_name}: invalid completed item '{item}'.")
    if not (is_placeholder(now_item) or is_skill_expr(now_item)):
        raise ValueError(f"{field_name}: invalid now item '{now_item}'.")

    plan_skills = {item for item in plan_items if is_skill_expr(item)}
    done_skills = {item for item in done_items if is_skill_expr(item)}
    if not done_skills.issubset(plan_skills):
        raise ValueError(f"{field_name}: 'What I have done' contains skills not in plan.")
    if is_skill_expr(now_item) and plan_skills and now_item not in plan_skills:
        raise ValueError(f"{field_name}: 'Now I need to do' is not in plan.")


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
    print("text", text)
    raise ValueError("Unbalanced JSON object in model output.")


def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    req = Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenRouter HTTP {exc.code}: {body[:500]}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter connection error: {exc}") from exc

    parsed = json.loads(raw)
    choices = parsed.get("choices")
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {raw[:500]}")
    content = choices[0].get("message", {}).get("content")
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict):
                chunks.append(str(part.get("text", "")))
            else:
                chunks.append(str(part))
        return "".join(chunks)
    if content is None:
        raise RuntimeError(f"OpenRouter returned empty content: {raw[:500]}")
    return str(content)


def build_prompts(field_map: dict[str, str]) -> tuple[str, str]:
    system_prompt = (
        "You convert robot plans to skill-based plans and must follow instructions exactly. "
        "Return JSON only."
    )
    user_prompt = f"""You must translate plan text into the allowed skill syntax.

Important rules:
1. ONLY translate action text in "Plan:", "What I have done:", and "Now I need to do:".
2. The "Now I need to do:" part should only include a single skill.
3. Keep the "Instruction: ..." part EXACTLY unchanged for fields that contain it.
4. DO NOT change any timesteps or non-text metadata (not provided here).
5. Keep field-level correspondence:
   - If fields describe the same plan progression, keep the same skill sequence.
   - "updated_content_w_instruction" must represent instruction + updated_content.
6. The output skills can ONLY come from this skill set, following the exact skill name and syntax:
{SKILL_GUIDE}
7. Keep numbering when list numbering exists.
8. If text says TBD/Nothing and no concrete action exists, preserve that status.
9. Return STRICT JSON object with exactly the same keys as input, values as translated strings.

Input fields JSON:
{json.dumps(field_map, ensure_ascii=False, indent=2)}
"""
    return system_prompt, user_prompt


def translate_fields(
    field_map: dict[str, str],
    api_key: str,
    model: str,
    timeout: float,
    max_retries: int,
) -> dict[str, str]:
    system_prompt, user_prompt = build_prompts(field_map)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        # print("system prompt", system_prompt)
        # print("user prompt", user_prompt)
        try:
            raw_output = call_openrouter(api_key, model, system_prompt, user_prompt, timeout)
            obj = extract_json_object(raw_output)

            if set(obj.keys()) != set(field_map.keys()):
                raise ValueError(
                    f"Model output keys mismatch. expected={sorted(field_map.keys())} "
                    f"got={sorted(obj.keys())}"
                )

            translated: dict[str, str] = {}
            for key, original_text in field_map.items():
                out_val = obj.get(key)
                if not isinstance(out_val, str):
                    raise ValueError(f"Field '{key}' must be a string.")
                out_text = out_val

                # Hard-guard exact Instruction retention for fields that have Instruction.
                original_instruction = parse_instruction_prefix(original_text)
                if original_instruction is not None:
                    out_tail = parse_plan_tail(out_text)
                    if out_tail is None:
                        raise ValueError(f"Field '{key}' missing Plan after translation.")
                    out_text = original_instruction + out_tail

                out_text = canonicalize_structured_text(out_text)
                translated[key] = out_text

            # Ensure updated_content_w_instruction is deterministic combination when present.
            if (
                "updated_content_w_instruction" in translated
                and "updated_content" in translated
                and "content" in field_map
            ):
                content_instruction = parse_instruction_prefix(field_map["content"])
                updated_tail = parse_plan_tail(translated["updated_content"])
                if content_instruction is None or updated_tail is None:
                    raise ValueError(
                        "Cannot reconstruct updated_content_w_instruction from content/updated_content."
                    )
                translated["updated_content_w_instruction"] = content_instruction + updated_tail

            # Validate translated fields.
            for key, original_text in field_map.items():
                validate_field(original_text, translated[key], key)

            return translated
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries:
                break
            sleep_s = min(2 ** (attempt - 1), 8)
            print(
                f"  [retry {attempt}/{max_retries}] translation failed: {exc}. "
                f"Sleeping {sleep_s}s...",
                flush=True,
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"Translation failed after {max_retries} attempts: {last_error}")


def save_json(path: Path, data: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def current_skill_from_segment(segment: dict[str, Any]) -> str:
    source_text = segment.get("updated_content")
    if not isinstance(source_text, str):
        source_text = segment.get("content")
    if not isinstance(source_text, str):
        raise ValueError("Segment missing usable 'content'/'updated_content' for current_skill.")
    return extract_now_item(source_text)


def should_translate_field(value: Any) -> bool:
    return isinstance(value, str) and has_plan_structure(value)


def process_episode(
    episode_obj: dict[str, Any],
    api_key: str,
    model: str,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    out_episode = deepcopy(episode_obj)
    segments = out_episode.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Episode has no 'segments' list.")

    for seg_idx, segment in enumerate(segments):
        print("processing segment", seg_idx, "of", len(segments))
        if not isinstance(segment, dict):
            raise ValueError(f"Segment {seg_idx} is not an object.")
        field_map = {
            key: value
            for key, value in segment.items()
            if should_translate_field(value)
        }
        if field_map:
            translated = translate_fields(
                field_map=field_map,
                api_key=api_key,
                model=model,
                timeout=timeout,
                max_retries=max_retries,
            )
            for key, value in translated.items():
                segment[key] = value

        segment["current_skill"] = current_skill_from_segment(segment)

    return out_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate cot_simple.json plan strings to skill-based plans via OpenRouter/Gemini. "
            "Saves cot_skill.json after each completed index."
        )
    )
    default_input = Path(__file__).resolve().with_name("cot_simple.json")
    default_output = Path(__file__).resolve().with_name("cot_skill.json")

    parser.add_argument("--input", type=Path, default=default_input, help="Path to cot_simple.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path to output cot_skill.json",
    )
    parser.add_argument(
        "--model",
        type=str,
        # default="google/gemini-2.5-pro",
        # default="google/gemini-2.5-flash",
        default="google/gemini-3-flash-preview",
        help="OpenRouter model name (default: google/gemini-2.5-pro)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index (inclusive). Default: first index (0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "End index (exclusive). Default: no upper bound. "
            "Example: --limit 10 processes indices < 10."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Maximum retries per segment translation (default: 5).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="OpenRouter HTTP timeout in seconds (default: 120).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set.", file=sys.stderr)
        print('Set it with: export OPENROUTER_API_KEY="sk-or-v1-..."', file=sys.stderr)
        return 2

    if not args.input.is_file():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2

    source = load_json(args.input)
    index_keys = sorted((k for k in source.keys() if is_index_key(k)), key=lambda x: int(x))
    if not index_keys:
        print("ERROR: No numeric episode indices found.", file=sys.stderr)
        return 2

    min_idx = int(index_keys[0])
    max_exclusive = int(index_keys[-1]) + 1
    start_idx = min_idx if args.start is None else args.start
    end_idx = max_exclusive if args.limit is None else args.limit

    if start_idx < min_idx:
        raise ValueError(f"--start must be >= {min_idx}, got {start_idx}")
    if end_idx > max_exclusive:
        end_idx = max_exclusive
    if start_idx >= end_idx:
        raise ValueError(f"Empty processing range: start={start_idx}, end={end_idx}")

    selected = [k for k in index_keys if start_idx <= int(k) < end_idx]
    if not selected:
        raise ValueError("No indices selected to process.")

    # Non-index header keys are preserved exactly.
    output: dict[str, Any] = {
        key: deepcopy(value)
        for key, value in source.items()
        if not is_index_key(key)
    }

    # Resume mode: keep previously translated episodes if starting from >0.
    if start_idx > min_idx and args.output.exists():
        try:
            existing = load_json(args.output)
            for key, value in existing.items():
                if is_index_key(key):
                    output[key] = value
            print(f"Resuming from existing output: {args.output}")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: Could not load existing output ({exc}); starting without resume.")

    total = len(selected)
    print(
        f"Processing {total} indices ({start_idx} <= idx < {end_idx}) "
        f"using model '{args.model}'."
    )

    for processed, index_key in enumerate(selected, start=1):
        out_episode = process_episode(
            episode_obj=source[index_key],
            api_key=api_key,
            model=args.model,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        output[index_key] = out_episode
        save_json(args.output, output)

        pct = (processed / total) * 100.0
        print(
            f"[{processed}/{total}] index {index_key} complete "
            f"({pct:.1f}%) -> saved {args.output}",
            flush=True,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
