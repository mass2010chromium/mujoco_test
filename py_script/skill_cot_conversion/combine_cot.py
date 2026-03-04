#!/usr/bin/env python3
"""Combine partitioned cot_skill shard JSON files into one cot_skill.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def is_index_key(key: str) -> bool:
    return key.isdigit()


def parse_input_tokens(tokens: list[str]) -> list[Path]:
    paths: list[Path] = []
    for token in tokens:
        for part in token.split(","):
            cleaned = part.strip().strip("[]").strip().strip("'\"")
            if cleaned:
                paths.append(Path(cleaned))
    return paths


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSON must be an object: {path}")
    return data


def save_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def combine_files(input_paths: list[Path]) -> dict[str, Any]:
    if not input_paths:
        raise ValueError("No input files provided.")

    headers: dict[str, Any] = {}
    header_source: dict[str, Path] = {}
    indices: dict[str, Any] = {}
    seen_in: dict[str, Path] = {}
    dupes: list[tuple[str, Path, Path]] = []

    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        data = load_json(path)

        for key, value in data.items():
            if is_index_key(key):
                if key in seen_in:
                    dupes.append((key, seen_in[key], path))
                else:
                    seen_in[key] = path
                    indices[key] = value
            else:
                if key not in headers:
                    headers[key] = value
                    header_source[key] = path
                elif headers[key] != value:
                    raise ValueError(
                        f"Non-index key '{key}' differs across files "
                        f"({header_source[key]} vs {path})."
                    )

    if dupes:
        lines = [
            "Duplicate episode indices detected. Aborting without writing output:",
            f"  total duplicate indices: {len(dupes)}",
        ]
        max_show = 25
        for idx, first_path, second_path in dupes[:max_show]:
            lines.append(f"  index {idx}: {first_path} and {second_path}")
        if len(dupes) > max_show:
            lines.append(f"  ... and {len(dupes) - max_show} more duplicates.")
        raise ValueError("\n".join(lines))

    combined: dict[str, Any] = {}
    combined.update(headers)
    for key in sorted(indices.keys(), key=lambda x: int(x)):
        combined[key] = indices[key]
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine cot_skill shard JSON files into one file. "
            "Fails if duplicate numeric indices are found."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Input shard files. Supports either plain args "
            "(f1.json f2.json) or list-style token ([cot_skill_1100.json, cot_skill_2200.json, cot_skill_3300.json, cot_skill_4400.json])."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().with_name("cot_skill.json"),
        help="Output combined file path (default: ./cot_skill.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = parse_input_tokens(args.inputs)
    if not input_paths:
        raise ValueError("No valid input file paths parsed from arguments.")

    # Resolve relative paths against current working directory.
    resolved_inputs = [p.resolve() for p in input_paths]
    combined = combine_files(resolved_inputs)
    save_json_atomic(args.output.resolve(), combined)

    count = len([k for k in combined.keys() if is_index_key(k)])
    print(
        f"Combined {len(resolved_inputs)} files into {args.output.resolve()} "
        f"with {count} episode indices."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
