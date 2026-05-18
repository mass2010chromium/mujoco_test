#!/usr/bin/env python3
"""
Interactive visualization of combined LIBERO skill target/trace annotations.

Walks through each skill segment in the requested episode range. For each
segment, writes the per-segment overlay frame saved by
annotate_libero_target_traces.py (semantic target + prediction trace +
extraction trace + EE trace) to a single output PNG, with the skill name
drawn above the frame. The PNG is overwritten in place at every step so it
can be opened once in an IDE preview pane (or scp'd) and refreshed.

Advances on a terminal key press. Designed for headless / SSH use: no GUI
window is created. Read-only with respect to annotation outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import load_json  # noqa: E402
from common_trace import target_trace_scene_episode_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively visualize combined LIBERO skill target/trace annotations. "
            "Steps skill-by-skill across episodes; advances on a terminal key press."
        )
    )
    parser.add_argument(
        "traces_json",
        type=Path,
        help="Path to skill_target_traces.json (combined) or a single target-trace shard JSON.",
    )
    parser.add_argument(
        "--start-idx",
        "--start-episode",
        dest="start_episode",
        type=int,
        default=0,
        help="Inclusive episode start index.",
    )
    parser.add_argument(
        "--end-idx",
        "--end-episode",
        dest="end_episode",
        type=int,
        default=None,
        help="Exclusive episode end index. Defaults to one past the largest episode in the JSON.",
    )
    parser.add_argument(
        "--trace-scene-root",
        action="append",
        dest="trace_scene_roots",
        default=None,
        help=(
            "Roots that contain target_trace_scenes/ folders. Repeatable. "
            "Auto-discovered from sibling target_trace_run_*/ dirs if omitted."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help=(
            "If set, only visualize episodes whose task instruction equals this string "
            "(whitespace-collapsed, case-sensitive). Example: "
            "--task \"pick up the book and place it in the back compartment of the caddy\"."
        ),
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=None,
        help="Override the skill-text font size (defaults scale with image height).",
    )
    parser.add_argument(
        "--output",
        "--vis-path",
        dest="output_path",
        type=Path,
        default=Path("vis.png"),
        help="Path to write (and overwrite) the current visualization frame. Default: ./vis.png.",
    )
    return parser.parse_args()


def discover_trace_scene_roots(traces_json_path: Path) -> list[Path]:
    """Search common locations for target_trace_scenes/ folders.

    Looks at the JSON's parent (combined-output layout) and grandparent
    (single-shard layout: <run>/episode_shards/<file>.json).
    """
    candidates: list[Path] = []
    seen: set[Path] = set()
    for parent in (traces_json_path.parent, traces_json_path.parent.parent):
        for path in sorted(parent.glob("target_trace_run_*/target_trace_scenes")):
            resolved = path.resolve()
            if path.is_dir() and resolved not in seen:
                seen.add(resolved)
                candidates.append(path)
        direct = parent / "target_trace_scenes"
        resolved = direct.resolve()
        if direct.is_dir() and resolved not in seen:
            seen.add(resolved)
            candidates.append(direct)
    return candidates


def find_scene_image(
    roots: list[Path],
    *,
    episode_index: int,
    skill_index: int,
    start_step: int,
) -> Path | None:
    filename = f"skill_{skill_index:03d}_start_step_{start_step:06d}.png"
    for root in roots:
        candidate = target_trace_scene_episode_dir(root, episode_index) / filename
        if candidate.exists():
            return candidate
    return None


def load_episodes(data: Any) -> dict[int, dict[str, Any]]:
    if isinstance(data, dict) and "episode_index" in data and "target_traces" in data:
        episode = dict(data)
        return {int(episode["episode_index"]): episode}
    if isinstance(data, dict):
        episodes: dict[int, dict[str, Any]] = {}
        for key, value in data.items():
            if isinstance(key, str) and key.isdigit() and isinstance(value, dict):
                episode = dict(value)
                episode.setdefault("episode_index", int(key))
                episodes[int(key)] = episode
        if episodes:
            return episodes
    raise ValueError("Could not parse target trace JSON: no episode entries found.")


def _wrap_text_to_width(text: str, *, font: Any, max_width: int, draw: Any) -> list[str]:
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def overlay_skill_text(image: Any, text: str, font_size: int | None) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    base = image.convert("RGB")
    width, height = base.size
    size = font_size if font_size is not None else max(14, height // 14)

    font = None
    for candidate in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            font = ImageFont.truetype(candidate, size=size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    pad = 6
    measure_draw = ImageDraw.Draw(base)
    lines = _wrap_text_to_width(text, font=font, max_width=width - 2 * pad, draw=measure_draw)
    line_metrics = [measure_draw.textbbox((0, 0), line, font=font) for line in lines]
    line_height = max(bbox[3] - bbox[1] for bbox in line_metrics) if line_metrics else 0
    line_spacing = max(2, line_height // 4)
    bar_h = 2 * pad + len(lines) * line_height + max(0, len(lines) - 1) * line_spacing

    canvas = Image.new("RGB", (width, height + bar_h), (0, 0, 0))
    canvas.paste(base, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    cursor_y = pad
    for line, bbox in zip(lines, line_metrics):
        draw.text((pad, cursor_y - bbox[1]), line, fill=(255, 255, 255), font=font)
        cursor_y += line_height + line_spacing
    return canvas


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def build_segment_items(
    episodes: dict[int, dict[str, Any]],
    *,
    start_episode: int,
    end_episode: int,
    roots: list[Path],
    task: str | None = None,
) -> list[dict[str, Any]]:
    target_task = _normalize_text(task) if task is not None else None
    items: list[dict[str, Any]] = []
    for episode_index in sorted(episodes):
        if not (start_episode <= episode_index < end_episode):
            continue
        episode = episodes[episode_index]
        instruction = str(episode.get("instruction", ""))
        if target_task is not None and _normalize_text(instruction) != target_task:
            continue
        target_traces = episode.get("target_traces") or []
        skill_count = len(target_traces)
        for entry in target_traces:
            skill_index = int(entry["skill_index"])
            start_step = int(entry["start_step"])
            end_step = int(entry["end_step"])
            skill = str(entry["skill"])
            image_path = find_scene_image(
                roots,
                episode_index=episode_index,
                skill_index=skill_index,
                start_step=start_step,
            )
            items.append(
                {
                    "episode_index": episode_index,
                    "skill_index": skill_index,
                    "skill_count": skill_count,
                    "skill": skill,
                    "start_step": start_step,
                    "end_step": end_step,
                    "image_path": image_path,
                    "instruction": instruction,
                }
            )
    return items


def main() -> int:
    args = parse_args()
    traces_json_path = args.traces_json.expanduser().resolve()
    if not traces_json_path.is_file():
        print(f"target traces JSON not found: {traces_json_path}", file=sys.stderr)
        return 1

    episodes = load_episodes(load_json(traces_json_path))

    start_episode = max(0, int(args.start_episode))
    end_episode = (
        max(episodes) + 1 if args.end_episode is None else int(args.end_episode)
    )
    if not (0 <= start_episode < end_episode):
        print(f"Invalid episode range [{start_episode}, {end_episode}).", file=sys.stderr)
        return 1

    if args.trace_scene_roots:
        roots = [Path(p).expanduser().resolve() for p in args.trace_scene_roots]
    else:
        roots = discover_trace_scene_roots(traces_json_path)
    if not roots:
        print(
            "Could not auto-discover any target_trace_scenes/ root. "
            "Pass one or more --trace-scene-root paths.",
            file=sys.stderr,
        )
        return 1
    print("trace_scene_roots:")
    for root in roots:
        print(f"  - {root}")

    items = build_segment_items(
        episodes,
        start_episode=start_episode,
        end_episode=end_episode,
        roots=roots,
        task=args.task,
    )
    if not items:
        if args.task is not None:
            target = _normalize_text(args.task)
            print(
                f"No skill segments to visualize in [{start_episode}, {end_episode}) "
                f"matching --task {target!r}."
            )
            unique_instructions = sorted(
                {_normalize_text(ep.get("instruction", "")) for ep in episodes.values()}
            )
            from difflib import get_close_matches

            suggestions = get_close_matches(target, unique_instructions, n=5, cutoff=0.4)
            if suggestions:
                print("Closest task instructions:")
                for suggestion in suggestions:
                    print(f"  - {suggestion}")
        else:
            print(f"No skill segments to visualize in [{start_episode}, {end_episode}).")
        return 1

    episode_count = len({item["episode_index"] for item in items})
    missing_count = sum(1 for item in items if item["image_path"] is None)
    print(
        f"loaded {len(items)} skill segment(s) across {episode_count} episode(s)"
        + (f" ({missing_count} missing scene image(s))" if missing_count else "")
    )
    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing current frame to {output_path} (overwritten each step)")
    print(
        "Navigation: <Enter>=next, p=prev, b=next-episode, q=quit. "
        "Pressing any other key advances by one segment."
    )

    from PIL import Image

    i = 0
    while 0 <= i < len(items):
        item = items[i]
        position = i + 1

        if item["image_path"] is None:
            print(
                f"[{position}/{len(items)}] missing visualization frame for "
                f"ep={item['episode_index']} skill={item['skill_index']} "
                f"start_step={item['start_step']}",
                file=sys.stderr,
            )
        else:
            image = Image.open(item["image_path"])
            overlay_text = (
                f"ep {item['episode_index']} | skill {item['skill_index']}: {item['skill']}"
            )
            display_image = overlay_skill_text(image, overlay_text, args.font_size)
            tmp_path = output_path.with_name(output_path.name + ".tmp.png")
            display_image.save(tmp_path, format="PNG")
            tmp_path.replace(output_path)

        prompt = (
            f"[{position}/{len(items)}] ep={item['episode_index']} "
            f"skill={item['skill_index']} ({item['skill']})  "
            "<Enter>=next p=prev b=next-episode q=quit > "
        )
        try:
            key = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if key in ("", "n"):
            i += 1
        elif key == "p":
            i = max(0, i - 1)
        elif key == "b":
            current_episode = item["episode_index"]
            j = i + 1
            while j < len(items) and items[j]["episode_index"] == current_episode:
                j += 1
            i = j
        elif key == "q":
            break
        else:
            i += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
