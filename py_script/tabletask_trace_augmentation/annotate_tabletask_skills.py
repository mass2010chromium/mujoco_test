#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    DEFAULT_ANNOTATION_FPS,
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_REPO_ID,
    PROMPT_VERSION,
    build_annotation_schema,
    build_episode_annotation,
    build_multimodal_prompt,
    build_video_data_url,
    episode_shard_path,
    episode_video_path,
    load_dataset_info,
    load_episode_records,
    load_json,
    normalize_model_steps,
    rebuild_combined_video,
    render_annotation_video,
    render_visualization_video,
    resolve_dataset_root,
    save_json_atomic,
    save_transition_scene_images,
    transition_scene_episode_dir,
    transition_scene_paths,
    visualization_video_path,
)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate LeRobot-formatted tabletop manipulation episodes (n5zhong/table_tasks) with "
            "skill segments by querying google/gemini-3.1-pro-preview through OpenRouter."
        )
    )
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
        required=True,
        help="Run directory for shard outputs, manifests, videos, transition scenes, and error logs.",
    )
    parser.add_argument("--start-episode", type=int, default=0, help="Inclusive episode start index.")
    parser.add_argument(
        "--end-episode",
        type=int,
        default=None,
        help="Exclusive episode end index. Defaults to the dataset episode count.",
    )
    parser.add_argument("--model", default=DEFAULT_OPENROUTER_MODEL, help="OpenRouter model id.")
    parser.add_argument(
        "--image-key",
        choices=["image", "wrist_image"],
        default="image",
        help="Camera stream to render into the annotation and visualization video.",
    )
    parser.add_argument(
        "--annotation-fps",
        type=int,
        default=DEFAULT_ANNOTATION_FPS,
        help="Playback FPS for the rendered annotation/visualization mp4. Default matches the libero pipeline.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum request / validation retries per episode.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=8.0,
        help="Base retry sleep in seconds for API or validation failures.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the OpenRouter request.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16000,
        help="Maximum completion tokens for the OpenRouter response.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip episodes whose shard file already exists in the output dir.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing episode shard files instead of skipping them.",
    )
    parser.add_argument(
        "--keep-annotation-videos",
        action="store_true",
        help="Keep the raw step-overlaid annotation videos (the ones sent to the model) "
        "under output-dir/annotation_videos.",
    )
    parser.add_argument(
        "--no-overlay-step-text",
        action="store_true",
        help="Disable per-frame step overlays in the rendered annotation mp4.",
    )
    parser.add_argument(
        "--disable-structured-output",
        action="store_true",
        help="Do not send a response_format json_schema request. Use only if the provider rejects structured output.",
    )
    parser.add_argument(
        "--disable-saving-transition-scene",
        action="store_true",
        help="Disable saving clean camera frames for each skill-boundary transition under "
        "output-dir/transition_scenes.",
    )
    parser.add_argument(
        "--disable-saving-visualization-video",
        action="store_true",
        help="Disable saving the per-episode visualization mp4 (and skip rebuilding combined.mp4).",
    )
    parser.add_argument(
        "--no-combined-video",
        action="store_true",
        help="Disable rebuilding combined.mp4 even if individual visualization videos are produced.",
    )
    return parser.parse_args()


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    return stripped


def extract_message_content(response_json: dict[str, Any]) -> str:
    try:
        message = response_json["choices"][0]["message"]["content"]
    except Exception as exc:
        raise ValueError(f"Unexpected OpenRouter response shape: {response_json}") from exc

    if isinstance(message, str):
        return message
    if isinstance(message, list):
        text_parts: list[str] = []
        for item in message:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        if text_parts:
            return "\n".join(text_parts)
    raise ValueError(f"Could not extract text content from response: {message!r}")


def parse_response_json(text: str) -> dict[str, Any]:
    return json.loads(strip_json_fence(text))


def build_request_payload(
    *,
    model: str,
    prompt: str,
    video_data_url: str,
    temperature: float,
    max_tokens: int,
    structured_output: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": video_data_url}},
                ],
            }
        ],
    }
    if structured_output:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": build_annotation_schema(),
        }
        payload["provider"] = {"require_parameters": True}
        payload["plugins"] = [{"id": "response-healing"}]
    return payload


def call_openrouter(
    *,
    api_key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openrouter.ai",
        "X-Title": "tabletask-skill-augmentation",
    }
    request = Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def annotate_episode(
    *,
    api_key: str,
    record,
    video_path: Path,
    annotation_fps: int,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_sleep_seconds: float,
    structured_output: bool,
) -> dict[str, Any]:
    prompt = build_multimodal_prompt(record.instruction, record.length, fps=annotation_fps)
    video_data_url = build_video_data_url(video_path)
    last_error: Exception | None = None
    attempted_fallback = False

    for attempt in range(1, max_retries + 1):
        request_prompt = prompt
        if last_error is not None:
            request_prompt = (
                prompt
                + "\n\nThe previous output failed validation. Regenerate from scratch and fix these issues:\n"
                + f"- {last_error}\n"
                + "Return valid JSON only."
            )

        payload = build_request_payload(
            model=model,
            prompt=request_prompt,
            video_data_url=video_data_url,
            temperature=temperature,
            max_tokens=max_tokens,
            structured_output=structured_output and not attempted_fallback,
        )

        try:
            response_json = call_openrouter(api_key=api_key, payload=payload)
            message_text = extract_message_content(response_json)
            parsed = parse_response_json(message_text)
            normalized = normalize_model_steps(parsed, record.length)
            return {
                "raw_response": response_json,
                "parsed_output": parsed,
                "normalized": normalized,
            }
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if (
                structured_output
                and not attempted_fallback
                and exc.code in (400, 422)
                and "response_format" in body
            ):
                attempted_fallback = True
                print(
                    f"  structured-output rejected for episode {record.episode_index}; retrying once without json_schema",
                    flush=True,
                )
                last_error = ValueError(body)
                continue
            last_error = RuntimeError(f"HTTP {exc.code}: {body}")
        except URLError as exc:
            last_error = RuntimeError(f"Network error: {exc}")
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc

        if attempt < max_retries:
            sleep_seconds = retry_sleep_seconds * attempt
            print(
                f"  retry {attempt}/{max_retries - 1} for episode {record.episode_index} after error: {last_error}",
                flush=True,
            )
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise last_error


def ensure_manifest(path: Path, *, args: argparse.Namespace, dataset_root: Path, total_episodes: int) -> None:
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_version": PROMPT_VERSION,
        "repo_id": args.repo_id,
        "dataset_root": str(dataset_root),
        "model": args.model,
        "image_key": args.image_key,
        "annotation_fps": args.annotation_fps,
        "start_episode": args.start_episode,
        "end_episode": args.end_episode if args.end_episode is not None else total_episodes,
        "total_episodes": total_episodes,
        "structured_output": not args.disable_structured_output,
        "save_transition_scenes": not args.disable_saving_transition_scene,
        "save_visualization_videos": not args.disable_saving_visualization_video,
    }
    save_json_atomic(path, manifest)


def main() -> int:
    args = parse_args()
    if args.skip_existing and args.overwrite_existing:
        raise ValueError("Use either --skip-existing or --overwrite-existing, not both.")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    dataset_root = resolve_dataset_root(args.repo_id, args.dataset_root)
    info = load_dataset_info(dataset_root)
    records = load_episode_records(dataset_root)
    total_episodes = len(records)

    start_episode = max(0, int(args.start_episode))
    end_episode = total_episodes if args.end_episode is None else min(int(args.end_episode), total_episodes)
    if not (0 <= start_episode < end_episode <= total_episodes):
        raise ValueError(
            f"Invalid episode range [{start_episode}, {end_episode}) for dataset with {total_episodes} episodes."
        )

    output_dir = args.output_dir.expanduser().resolve()
    shard_dir = output_dir / "episode_shards"
    error_dir = output_dir / "errors"
    annotation_video_dir = output_dir / "annotation_videos"
    transition_scene_dir = output_dir / "transition_scenes"
    visualization_video_dir = output_dir / "videos"
    shard_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)
    if args.keep_annotation_videos:
        annotation_video_dir.mkdir(parents=True, exist_ok=True)
    if not args.disable_saving_transition_scene:
        transition_scene_dir.mkdir(parents=True, exist_ok=True)
    save_visualization_videos = not args.disable_saving_visualization_video
    if save_visualization_videos:
        visualization_video_dir.mkdir(parents=True, exist_ok=True)

    ensure_manifest(
        output_dir / "run_manifest.json",
        args=args,
        dataset_root=dataset_root,
        total_episodes=total_episodes,
    )

    print(f"dataset_root={dataset_root}", flush=True)
    print(f"episodes={total_episodes}, range=[{start_episode}, {end_episode})", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"openrouter_model={args.model}", flush=True)
    print(f"annotation_fps={args.annotation_fps}", flush=True)

    processed = 0
    skipped = 0
    failed = 0
    newly_produced_visualization = False
    start_time = time.time()

    for episode_index in range(start_episode, end_episode):
        record = records[episode_index]
        shard_path = episode_shard_path(shard_dir, episode_index)
        error_path = error_dir / f"episode_{episode_index:06d}.error.json"
        episode_transition_dir = (
            transition_scene_episode_dir(transition_scene_dir, episode_index)
            if not args.disable_saving_transition_scene
            else None
        )
        episode_visualization_path = (
            visualization_video_path(visualization_video_dir, episode_index)
            if save_visualization_videos
            else None
        )
        source_video_path = episode_video_path(dataset_root, info, episode_index, args.image_key)

        if shard_path.exists():
            if args.skip_existing and not args.overwrite_existing:
                # Best-effort backfill of transition scenes and visualization video for already-annotated episodes.
                try:
                    existing_episode = load_json(shard_path)
                    segments = existing_episode["segments"]
                    if episode_transition_dir is not None:
                        expected_paths = transition_scene_paths(
                            transition_scene_dir,
                            episode_index=episode_index,
                            segments=segments,
                        )
                        if not all(path.exists() for path in expected_paths):
                            save_transition_scene_images(
                                record=record,
                                source_video_path=source_video_path,
                                segments=segments,
                                output_dir=transition_scene_dir,
                            )
                    if episode_visualization_path is not None and not episode_visualization_path.exists():
                        render_visualization_video(
                            record=record,
                            segments=segments,
                            source_video_path=source_video_path,
                            output_path=episode_visualization_path,
                            total_episodes=total_episodes,
                            fps=args.annotation_fps,
                        )
                        newly_produced_visualization = True
                except Exception as exc:
                    print(
                        f"  warning: could not backfill aux artifacts for episode {episode_index}: {exc}",
                        flush=True,
                    )
                skipped += 1
                print(f"[skip] episode={episode_index} shard already exists", flush=True)
                continue
            if args.overwrite_existing:
                shard_path.unlink()

        annotation_video_path = (
            annotation_video_dir / f"episode_{episode_index:06d}.mp4"
            if args.keep_annotation_videos
            else output_dir / f".episode_{episode_index:06d}.annotation.mp4"
        )
        if annotation_video_path.exists() and args.overwrite_existing:
            annotation_video_path.unlink()
        if episode_transition_dir is not None and episode_transition_dir.exists():
            shutil.rmtree(episode_transition_dir)
        if episode_visualization_path is not None and episode_visualization_path.exists():
            episode_visualization_path.unlink()

        episode_t0 = time.time()
        print(
            f"[{episode_index - start_episode + 1}/{end_episode - start_episode}] "
            f"episode={episode_index} task_index={record.task_index} length={record.length}",
            flush=True,
        )

        try:
            if not source_video_path.exists():
                raise FileNotFoundError(
                    f"Source video missing for episode {episode_index}: {source_video_path}"
                )

            if not annotation_video_path.exists():
                render_annotation_video(
                    record=record,
                    source_video_path=source_video_path,
                    output_path=annotation_video_path,
                    fps=args.annotation_fps,
                    overlay_text=not args.no_overlay_step_text,
                )

            result = annotate_episode(
                api_key=api_key,
                record=record,
                video_path=annotation_video_path,
                annotation_fps=args.annotation_fps,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                retry_sleep_seconds=args.retry_sleep_seconds,
                structured_output=not args.disable_structured_output,
            )

            episode_annotation = build_episode_annotation(
                record=record,
                normalized=result["normalized"],
                model=args.model,
                source_repo_id=args.repo_id,
                dataset_root=dataset_root,
                annotation_fps=args.annotation_fps,
                prompt_version=PROMPT_VERSION,
                raw_response=result["parsed_output"],
            )

            if episode_transition_dir is not None:
                save_transition_scene_images(
                    record=record,
                    source_video_path=source_video_path,
                    segments=episode_annotation["segments"],
                    output_dir=transition_scene_dir,
                )
            if episode_visualization_path is not None:
                render_visualization_video(
                    record=record,
                    segments=episode_annotation["segments"],
                    source_video_path=source_video_path,
                    output_path=episode_visualization_path,
                    total_episodes=total_episodes,
                    fps=args.annotation_fps,
                )
                newly_produced_visualization = True

            save_json_atomic(shard_path, episode_annotation)
            if error_path.exists():
                error_path.unlink()
            if not args.keep_annotation_videos and annotation_video_path.exists():
                annotation_video_path.unlink()

            processed += 1
            elapsed = time.time() - episode_t0
            print(
                f"  saved {shard_path.name} with {len(episode_annotation['segments'])} skill segment(s) "
                f"in {elapsed:.1f}s",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            error_payload = {
                "episode_index": episode_index,
                "task_index": record.task_index,
                "instruction": record.instruction,
                "length": record.length,
                "error": f"{type(exc).__name__}: {exc}",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            save_json_atomic(error_path, error_payload)
            # Ensure no half-baked visualization remains for this failure.
            if episode_visualization_path is not None and episode_visualization_path.exists():
                episode_visualization_path.unlink()
            if not args.keep_annotation_videos and annotation_video_path.exists():
                try:
                    annotation_video_path.unlink()
                except FileNotFoundError:
                    pass
            print(f"  failed episode={episode_index}: {error_payload['error']}", flush=True)

    if save_visualization_videos and not args.no_combined_video:
        try:
            combined = rebuild_combined_video(visualization_video_dir)
            if combined is not None:
                print(f"combined_video={combined}", flush=True)
        except Exception as exc:
            print(f"warning: combined video rebuild failed: {exc}", flush=True)

    total_elapsed = time.time() - start_time
    print(
        f"done: processed={processed} skipped={skipped} failed={failed} elapsed={total_elapsed/60.0:.1f}m",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
