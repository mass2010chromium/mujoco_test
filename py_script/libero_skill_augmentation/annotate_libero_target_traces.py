#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

THIS_DIR = Path(__file__).resolve().parent
PY_SCRIPT_DIR = THIS_DIR.parent
for path in (THIS_DIR, PY_SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (  # noqa: E402
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_REPO_ID,
    cumulative_episode_bounds,
    episode_shard_path,
    load_episode_frame,
    load_episode_records,
    load_json,
    load_lerobot_dataset,
    save_json_atomic,
)
from common_trace import (  # noqa: E402
    CONTACT_POINT_LABEL,
    DEFAULT_EE_TRACE_MAX_OUT_OF_BOUNDS_FRACTION,
    DEFAULT_EE_REPROJECTION_MAX_ERROR_METERS,
    DEFAULT_EE_ROUNDED_REPROJECTION_MAX_ERROR_METERS,
    DEFAULT_EE_TRACE_MAX_STEP_DELTA_PIXELS,
    DEFAULT_TRACE_FRAME_COUNT,
    END_EFFECTOR_TRACE_KIND,
    EXTRACTION_TRACE_LABEL,
    PREDICTION_TRACE_LABEL,
    SEMANTIC_TARGET_LABEL,
    TARGET_TRACE_COORDINATE_RESOLUTION,
    TARGET_TRACE_PROMPT_VERSION,
    build_contact_point_prompt,
    build_bddl_language_index,
    build_episode_target_trace_annotation,
    build_extraction_trace_prompt,
    build_image_data_url,
    build_point_schema,
    build_projected_end_effector_trace,
    build_prediction_trace_prompt,
    build_semantic_target_prompt,
    build_trace_schema,
    build_video_data_url,
    camera_name_for_image_key,
    compute_libero_camera_calibration,
    load_skill_annotation_episodes,
    normalize_point_response,
    normalize_trace_response,
    parse_skill_name,
    parse_skill_object_descriptions,
    render_sampled_segment_video,
    resolve_bddl_path_for_instruction,
    resolve_dataset_root_from_skill_data,
    save_target_trace_scene_images,
    select_evenly_spaced_frame_indices,
    target_trace_scene_episode_dir,
    target_trace_scene_paths,
    validate_skill_episode_shape,
)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {parsed}.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract semantic target points and contact-point traces for an existing LIBERO skill annotation run."
        )
    )
    parser.add_argument(
        "annotation_dir",
        type=Path,
        help="Folder containing a completed skill_annotations.json from the Libero skill annotation pipeline.",
    )
    parser.add_argument(
        "--skill-annotations",
        type=Path,
        default=None,
        help="Explicit skill annotation JSON path. Defaults to annotation_dir/skill_annotations.json.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot dataset repo id.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Existing LeRobot dataset root. Defaults to the path recorded in skill_annotations.json, then HF cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Trace run directory. Defaults to annotation_dir/target_trace_run_<start>_<end>.",
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
        help="Exclusive episode end index. Defaults to one past the largest annotated episode id.",
    )
    parser.add_argument("--model", default=DEFAULT_OPENROUTER_MODEL, help="OpenRouter VLM model id.")
    parser.add_argument(
        "--image-key",
        choices=["image", "wrist_image"],
        default="image",
        help="Observation image stream used for target/trace extraction. Use image for the side-view camera.",
    )
    parser.add_argument(
        "--query-image-width",
        type=positive_int_arg,
        default=None,
        help="Width to resize first-frame images and sampled video frames before sending them to Gemini.",
    )
    parser.add_argument(
        "--query-image-height",
        type=positive_int_arg,
        default=None,
        help="Height to resize first-frame images and sampled video frames before sending them to Gemini.",
    )
    parser.add_argument(
        "--model-coordinate-resolution",
        type=positive_int_arg,
        default=TARGET_TRACE_COORDINATE_RESOLUTION,
        help="Square coordinate grid Gemini should use before rescaling to original image pixels.",
    )
    parser.add_argument(
        "--trace-frame-count",
        type=positive_int_arg,
        default=DEFAULT_TRACE_FRAME_COUNT,
        help=(
            "Maximum number of evenly spaced skill-segment frames to include in each trace video. "
            "Shorter skills use every frame."
        ),
    )
    parser.add_argument("--trace-video-fps", type=positive_int_arg, default=10, help="FPS for sampled trace videos.")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum request / validation retries per Gemini point or trace query.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=8.0,
        help="Base retry sleep in seconds for API or validation failures.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=16000, help="Maximum completion tokens per request.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip episodes whose target-trace shard file already exists in the output dir.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing target-trace shards instead of skipping them.",
    )
    parser.add_argument(
        "--disable-structured-output",
        action="store_true",
        help="Do not send response_format json_schema. Use only if the provider rejects structured output.",
    )
    parser.add_argument(
        "--disable-saving-trace-scenes",
        action="store_true",
        help="Disable saving start-frame target/trace overlay visualizations.",
    )
    parser.add_argument(
        "--keep-videos",
        action="store_true",
        help="Keep sampled skill videos under output-dir/videos.",
    )
    parser.add_argument(
        "--no-contact-prediction",
        action="store_true",
        help="Disable predicted-contact-point tracking trace generation.",
    )
    parser.add_argument(
        "--no-contact-extraction",
        action="store_true",
        help="Disable direct extraction contact-point trace generation.",
    )
    parser.add_argument(
        "--predict-contact-only",
        action="store_true",
        help=(
            "Predict the first-frame contact point without tracking it. Independent of "
            "--no-contact-prediction: the prediction runs whenever this flag is set, even when "
            "--no-contact-prediction also disables the tracking trace."
        ),
    )
    parser.add_argument(
        "--use-place-video",
        action="store_true",
        help=(
            "Also send the skill-segment video as a hint when annotating PLACE_ON / PLACE_IN skills, "
            "the same way it is already sent for PICKUP_FROM. Default off; when off, semantic-target "
            "annotation for PLACE_* skills uses only the first frame, exactly as before."
        ),
    )
    parser.add_argument(
        "--no-ee-trace",
        action="store_true",
        help="Disable dense projected end-effector trace generation.",
    )
    parser.add_argument(
        "--ee-max-step-delta-pixels",
        type=float,
        default=DEFAULT_EE_TRACE_MAX_STEP_DELTA_PIXELS,
        help="Reject projected EE traces whose largest adjacent-frame pixel jump exceeds this value.",
    )
    parser.add_argument(
        "--ee-max-out-of-bounds-fraction",
        type=float,
        default=DEFAULT_EE_TRACE_MAX_OUT_OF_BOUNDS_FRACTION,
        help=(
            "Reject projected EE traces if more than this fraction of raw projected points lie outside "
            "the image before clipping. Default is permissive because the gripper can legitimately leave frame."
        ),
    )
    parser.add_argument(
        "--ee-max-reprojection-error-meters",
        type=float,
        default=DEFAULT_EE_REPROJECTION_MAX_ERROR_METERS,
        help="Reject EE traces whose exact project-and-reproject max 3D error exceeds this value.",
    )
    parser.add_argument(
        "--ee-max-rounded-reprojection-error-meters",
        type=float,
        default=DEFAULT_EE_ROUNDED_REPROJECTION_MAX_ERROR_METERS,
        help=(
            "Reject EE traces whose rounded in-bounds pixel project-and-reproject max 3D error exceeds this value."
        ),
    )
    parser.add_argument(
        "--libero-bddl-root",
        type=Path,
        default=None,
        help="Optional LIBERO bddl_files root for resolving per-task camera calibration.",
    )
    parser.add_argument(
        "--store-full-openrouter-response",
        action="store_true",
        help="Store full OpenRouter responses. This can make episode shards much larger.",
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
    except Exception as exc:  # pragma: no cover - defensive
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


def parse_response_json(text: str) -> Any:
    return json.loads(strip_json_fence(text))


def build_request_payload(
    *,
    model: str,
    prompt: str,
    media_data_url: str,
    media_type: str,
    schema: dict[str, Any],
    temperature: float,
    max_tokens: int,
    structured_output: bool,
    extra_image_data_urls: list[str] | None = None,
    extra_video_data_url: str | None = None,
) -> dict[str, Any]:
    if media_type not in {"image", "video"}:
        raise ValueError(f"Unsupported media_type: {media_type}")
    media_key = "image_url" if media_type == "image" else "video_url"
    content_type = "image_url" if media_type == "image" else "video_url"
    if extra_image_data_urls and media_type != "image":
        raise ValueError("extra_image_data_urls is only supported when media_type='image'.")
    if extra_video_data_url and media_type != "image":
        raise ValueError("extra_video_data_url is only supported when media_type='image'.")
    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {"type": content_type, media_key: {"url": media_data_url}},
    ]
    if extra_image_data_urls:
        for url in extra_image_data_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
    if extra_video_data_url:
        content.append({"type": "video_url", "video_url": {"url": extra_video_data_url}})
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    if structured_output:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": schema,
        }
        payload["provider"] = {"require_parameters": True}
        payload["plugins"] = [{"id": "response-healing"}]
    return payload


def call_openrouter(*, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openrouter.ai",
        "X-Title": "libero-target-trace-augmentation",
    }
    request = Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def query_openrouter_json(
    *,
    api_key: str,
    model: str,
    prompt: str,
    media_data_url: str,
    media_type: str,
    schema: dict[str, Any],
    normalizer: Callable[[Any], dict[str, Any]],
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_sleep_seconds: float,
    structured_output: bool,
    store_full_response: bool,
    query_name: str,
    extra_image_data_urls: list[str] | None = None,
    extra_video_data_url: str | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    attempted_fallback = False
    for attempt in range(1, max_retries + 1):
        request_prompt = prompt
        if last_error is not None:
            request_prompt = (
                prompt
                + "\n\nThe previous output failed validation. Regenerate from scratch and fix this issue:\n"
                + f"- {last_error}\n"
                + "Return valid JSON only."
            )

        payload = build_request_payload(
            model=model,
            prompt=request_prompt,
            media_data_url=media_data_url,
            media_type=media_type,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            structured_output=structured_output and not attempted_fallback,
            extra_image_data_urls=extra_image_data_urls,
            extra_video_data_url=extra_video_data_url,
        )
        try:
            request_t0 = time.time()
            response_json = call_openrouter(api_key=api_key, payload=payload)
            print(f"    {query_name}: Gemini response in {time.time() - request_t0:.1f}s", flush=True)
            parsed = parse_response_json(extract_message_content(response_json))
            normalized = normalizer(parsed)
            result = {
                "parsed_output": parsed,
                "normalized": normalized,
            }
            if store_full_response:
                result["openrouter_response"] = response_json
            return result
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if (
                structured_output
                and not attempted_fallback
                and exc.code in (400, 422)
                and "response_format" in body
            ):
                attempted_fallback = True
                print(f"    {query_name}: structured output rejected; retrying without json_schema", flush=True)
                last_error = ValueError(body)
                continue
            last_error = RuntimeError(f"HTTP {exc.code}: {body}")
        except URLError as exc:
            last_error = RuntimeError(f"Network error: {exc}")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            last_error = exc

        if attempt < max_retries:
            sleep_seconds = retry_sleep_seconds * attempt
            print(f"    retry {attempt}/{max_retries - 1} for {query_name} after error: {last_error}", flush=True)
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise last_error


def ensure_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    skill_annotations_path: Path,
    dataset_root: Path,
    start_episode: int,
    end_episode: int,
    selected_count: int,
) -> None:
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_version": TARGET_TRACE_PROMPT_VERSION,
        "repo_id": args.repo_id,
        "dataset_root": str(dataset_root),
        "skill_annotations": str(skill_annotations_path),
        "model": args.model,
        "image_key": args.image_key,
        "query_image_width": args.query_image_width,
        "query_image_height": args.query_image_height,
        "model_coordinate_resolution": args.model_coordinate_resolution,
        "trace_frame_count": args.trace_frame_count,
        "trace_video_fps": args.trace_video_fps,
        "contact_prediction_enabled": not args.no_contact_prediction,
        "contact_extraction_enabled": not args.no_contact_extraction,
        "contact_prediction_only_enabled": bool(args.predict_contact_only),
        "use_place_video": bool(args.use_place_video),
        "end_effector_trace_enabled": not args.no_ee_trace,
        "semantic_target_enabled": True,
        "ee_trace_kind": END_EFFECTOR_TRACE_KIND,
        "ee_max_step_delta_pixels": args.ee_max_step_delta_pixels,
        "ee_max_out_of_bounds_fraction": args.ee_max_out_of_bounds_fraction,
        "ee_max_reprojection_error_meters": args.ee_max_reprojection_error_meters,
        "ee_max_rounded_reprojection_error_meters": args.ee_max_rounded_reprojection_error_meters,
        "libero_bddl_root": str(args.libero_bddl_root.expanduser().resolve()) if args.libero_bddl_root else None,
        "start_episode": start_episode,
        "end_episode": end_episode,
        "selected_episodes": selected_count,
        "structured_output": not args.disable_structured_output,
        "save_trace_scenes": not args.disable_saving_trace_scenes,
    }
    save_json_atomic(path, manifest)


def get_episode_ee_camera_calibration(
    *,
    args: argparse.Namespace,
    skill_episode: dict[str, Any],
    image_width: int,
    image_height: int,
    bddl_index: dict[str, list[Path]],
    calibration_cache: dict[tuple[str, str, int, int], dict[str, Any]],
) -> dict[str, Any] | None:
    if args.no_ee_trace:
        return None
    if args.image_key != "image":
        raise ValueError(
            "Dense projected EE traces currently support --image-key image / LIBERO agentview only. "
            "The wrist camera is frame-dependent and the LeRobot dataset does not store enough simulator state "
            "to recover it exactly."
        )

    camera_name = camera_name_for_image_key(args.image_key)
    bddl_path = resolve_bddl_path_for_instruction(
        str(skill_episode["instruction"]),
        bddl_index=bddl_index,
        bddl_root=args.libero_bddl_root,
    )
    cache_key = (str(bddl_path), camera_name, int(image_width), int(image_height))
    if cache_key not in calibration_cache:
        print(f"  computing EE projection camera calibration: {camera_name} from {bddl_path.name}", flush=True)
        calibration_cache[cache_key] = compute_libero_camera_calibration(
            bddl_path=bddl_path,
            camera_name=camera_name,
            image_width=int(image_width),
            image_height=int(image_height),
        )
    return calibration_cache[cache_key]


def annotate_skill_segment(
    *,
    api_key: str,
    args: argparse.Namespace,
    dataset: Any,
    record: Any,
    episode_bounds: dict[int, tuple[int, int]],
    skill_episode: dict[str, Any],
    segment: dict[str, Any],
    skill_index: int,
    image_width: int,
    image_height: int,
    output_dir: Path,
    video_dir: Path,
    ee_camera_calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    skill = str(segment["skill"])
    start_step = int(segment["start_step"])
    end_step = int(segment["end_step"])
    sent_image_width = int(args.query_image_width) if args.query_image_width is not None else int(image_width)
    sent_image_height = int(args.query_image_height) if args.query_image_height is not None else int(image_height)
    model_coordinate_width = int(args.model_coordinate_resolution)
    model_coordinate_height = int(args.model_coordinate_resolution)

    semantic_target_enabled = True
    contact_prediction_query_enabled = (not args.no_contact_prediction) or args.predict_contact_only
    contact_query_enabled = (
        not args.no_contact_prediction
        or not args.no_contact_extraction
        or args.predict_contact_only
    )
    needs_first_frame_query = semantic_target_enabled or contact_query_enabled

    semantic_object_id: str | None = None
    contact_object_id: str | None = None
    image_data_url: str | None = None
    if needs_first_frame_query:
        descriptions = parse_skill_object_descriptions(skill)
        semantic_object_id = descriptions["semantic_target_object_id"]
        if contact_query_enabled:
            contact_object_id = descriptions["contact_object_id"]
        start_frame = load_episode_frame(
            dataset,
            record=record,
            episode_bounds=episode_bounds,
            local_step=start_step,
            image_key=args.image_key,
        )
        image_data_url = build_image_data_url(
            start_frame,
            width=args.query_image_width,
            height=args.query_image_height,
        )

    raw_model_responses: dict[str, Any] = {}
    full_openrouter_responses: dict[str, Any] = {}

    end_effector_trace = None
    if ee_camera_calibration is not None:
        end_effector_trace = build_projected_end_effector_trace(
            dataset,
            record=record,
            episode_bounds=episode_bounds,
            start_step=start_step,
            end_step=end_step,
            camera_calibration=ee_camera_calibration,
            image_width=image_width,
            image_height=image_height,
            max_step_delta_pixels=float(args.ee_max_step_delta_pixels),
            max_out_of_bounds_fraction=float(args.ee_max_out_of_bounds_fraction),
            max_reprojection_error_meters=float(args.ee_max_reprojection_error_meters),
            max_rounded_reprojection_error_meters=float(args.ee_max_rounded_reprojection_error_meters),
        )

    semantic_target = None
    hint_video_path: Path | None = None
    if semantic_target_enabled:
        assert semantic_object_id is not None and image_data_url is not None
        video_hint_step: int | None = None
        hint_video_data_url: str | None = None
        skill_name = parse_skill_name(skill)
        needs_video_hint = (
            skill_name == "PICKUP_FROM"
            or (args.use_place_video and skill_name in {"PLACE_ON", "PLACE_IN"})
        )
        if needs_video_hint and end_step > start_step:
            video_hint_step = int(end_step) - 1
            hint_frame_indices = select_evenly_spaced_frame_indices(
                start_step=start_step,
                end_step=end_step,
                max_frame_count=args.trace_frame_count,
            )
            if args.keep_videos:
                hint_video_path = (
                    video_dir
                    / f"episode_{record.episode_index:06d}_skill_{skill_index:03d}_semantic_hint.mp4"
                )
            else:
                hint_video_path = (
                    output_dir
                    / f".episode_{record.episode_index:06d}_skill_{skill_index:03d}_semantic_hint.mp4"
                )
            render_sampled_segment_video(
                dataset,
                record=record,
                episode_bounds=episode_bounds,
                frame_indices=hint_frame_indices,
                output_path=hint_video_path,
                image_key=args.image_key,
                width=args.query_image_width,
                height=args.query_image_height,
                fps=args.trace_video_fps,
                overlay_text=True,
            )
            hint_video_data_url = build_video_data_url(hint_video_path)
            print(
                f"    skill={skill_index} {skill_name.lower()} hint: rendered "
                f"{len(hint_frame_indices)}-frame segment video for steps [{start_step}, {end_step})",
                flush=True,
            )
        semantic_prompt = build_semantic_target_prompt(
            instruction=str(skill_episode["instruction"]),
            plan=str(skill_episode["plan"]),
            skill=skill,
            skill_index=skill_index,
            start_step=start_step,
            image_width=sent_image_width,
            image_height=sent_image_height,
            target_object_id=semantic_object_id,
            coordinate_width=model_coordinate_width,
            coordinate_height=model_coordinate_height,
            video_hint_step=video_hint_step,
        )
        semantic_query = query_openrouter_json(
            api_key=api_key,
            model=args.model,
            prompt=semantic_prompt,
            media_data_url=image_data_url,
            media_type="image",
            schema=build_point_schema("libero_semantic_target_point"),
            normalizer=lambda raw: normalize_point_response(
                raw,
                expected_object_id=semantic_object_id,
                expected_label=SEMANTIC_TARGET_LABEL,
                coordinate_width=model_coordinate_width,
                coordinate_height=model_coordinate_height,
                image_width=image_width,
                image_height=image_height,
            ),
            extra_video_data_url=hint_video_data_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
            structured_output=not args.disable_structured_output,
            store_full_response=args.store_full_openrouter_response,
            query_name=f"skill={skill_index} semantic_target",
        )
        semantic_target = semantic_query["normalized"]
        raw_model_responses["semantic_target"] = semantic_query["parsed_output"]
        if args.store_full_openrouter_response:
            full_openrouter_responses["semantic_target"] = semantic_query["openrouter_response"]

    contact_prediction = None
    prediction_trace = None
    extraction_trace = None

    if contact_prediction_query_enabled:
        assert contact_object_id is not None and image_data_url is not None
        contact_prompt = build_contact_point_prompt(
            instruction=str(skill_episode["instruction"]),
            plan=str(skill_episode["plan"]),
            skill=skill,
            skill_index=skill_index,
            start_step=start_step,
            image_width=sent_image_width,
            image_height=sent_image_height,
            contact_object_id=contact_object_id,
            coordinate_width=model_coordinate_width,
            coordinate_height=model_coordinate_height,
        )
        contact_query = query_openrouter_json(
            api_key=api_key,
            model=args.model,
            prompt=contact_prompt,
            media_data_url=image_data_url,
            media_type="image",
            schema=build_point_schema("libero_contact_point"),
            normalizer=lambda raw: normalize_point_response(
                raw,
                expected_object_id=contact_object_id,
                expected_label=CONTACT_POINT_LABEL,
                coordinate_width=model_coordinate_width,
                coordinate_height=model_coordinate_height,
                image_width=image_width,
                image_height=image_height,
            ),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
            structured_output=not args.disable_structured_output,
            store_full_response=args.store_full_openrouter_response,
            query_name=f"skill={skill_index} contact_point",
        )
        contact_prediction = contact_query["normalized"]
        raw_model_responses["contact_prediction"] = contact_query["parsed_output"]
        if args.store_full_openrouter_response:
            full_openrouter_responses["contact_prediction"] = contact_query["openrouter_response"]

    sampled_frame_indices = select_evenly_spaced_frame_indices(
        start_step=start_step,
        end_step=end_step,
        max_frame_count=args.trace_frame_count,
    )
    video_data_url = None
    video_path = None
    if not args.no_contact_prediction or not args.no_contact_extraction:
        if args.keep_videos:
            video_path = video_dir / f"episode_{record.episode_index:06d}_skill_{skill_index:03d}.mp4"
        else:
            video_path = output_dir / f".episode_{record.episode_index:06d}_skill_{skill_index:03d}.mp4"
        render_sampled_segment_video(
            dataset,
            record=record,
            episode_bounds=episode_bounds,
            frame_indices=sampled_frame_indices,
            output_path=video_path,
            image_key=args.image_key,
            width=args.query_image_width,
            height=args.query_image_height,
            fps=args.trace_video_fps,
            overlay_text=True,
        )
        video_data_url = build_video_data_url(video_path)

    if not args.no_contact_prediction:
        if contact_prediction is None:
            raise ValueError("contact_prediction is required for prediction_trace.")
        assert video_data_url is not None
        assert contact_object_id is not None
        prediction_prompt = build_prediction_trace_prompt(
            instruction=str(skill_episode["instruction"]),
            plan=str(skill_episode["plan"]),
            skill=skill,
            skill_index=skill_index,
            start_step=start_step,
            end_step=end_step,
            sampled_frame_indices=sampled_frame_indices,
            contact_object_id=contact_object_id,
            contact_point=contact_prediction,
            image_width=sent_image_width,
            image_height=sent_image_height,
            coordinate_width=model_coordinate_width,
            coordinate_height=model_coordinate_height,
        )
        prediction_query = query_openrouter_json(
            api_key=api_key,
            model=args.model,
            prompt=prediction_prompt,
            media_data_url=video_data_url,
            media_type="video",
            schema=build_trace_schema("libero_prediction_contact_trace"),
            normalizer=lambda raw: normalize_trace_response(
                raw,
                expected_label=PREDICTION_TRACE_LABEL,
                sampled_frame_indices=sampled_frame_indices,
                start_step=start_step,
                end_step=end_step,
                coordinate_width=model_coordinate_width,
                coordinate_height=model_coordinate_height,
                image_width=image_width,
                image_height=image_height,
            ),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
            structured_output=not args.disable_structured_output,
            store_full_response=args.store_full_openrouter_response,
            query_name=f"skill={skill_index} prediction_trace",
        )
        prediction_trace = prediction_query["normalized"]
        prediction_trace["trace_kind"] = "prediction_tracking"
        raw_model_responses["prediction_trace"] = prediction_query["parsed_output"]
        if args.store_full_openrouter_response:
            full_openrouter_responses["prediction_trace"] = prediction_query["openrouter_response"]

    if not args.no_contact_extraction:
        assert video_data_url is not None
        assert contact_object_id is not None
        extraction_prompt = build_extraction_trace_prompt(
            instruction=str(skill_episode["instruction"]),
            plan=str(skill_episode["plan"]),
            skill=skill,
            skill_index=skill_index,
            start_step=start_step,
            end_step=end_step,
            sampled_frame_indices=sampled_frame_indices,
            contact_object_id=contact_object_id,
            image_width=sent_image_width,
            image_height=sent_image_height,
            coordinate_width=model_coordinate_width,
            coordinate_height=model_coordinate_height,
        )
        extraction_query = query_openrouter_json(
            api_key=api_key,
            model=args.model,
            prompt=extraction_prompt,
            media_data_url=video_data_url,
            media_type="video",
            schema=build_trace_schema("libero_extraction_contact_trace"),
            normalizer=lambda raw: normalize_trace_response(
                raw,
                expected_label=EXTRACTION_TRACE_LABEL,
                sampled_frame_indices=sampled_frame_indices,
                start_step=start_step,
                end_step=end_step,
                coordinate_width=model_coordinate_width,
                coordinate_height=model_coordinate_height,
                image_width=image_width,
                image_height=image_height,
            ),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
            structured_output=not args.disable_structured_output,
            store_full_response=args.store_full_openrouter_response,
            query_name=f"skill={skill_index} extraction_trace",
        )
        extraction_trace = extraction_query["normalized"]
        extraction_trace["trace_kind"] = "direct_extraction"
        raw_model_responses["extraction_trace"] = extraction_query["parsed_output"]
        if args.store_full_openrouter_response:
            full_openrouter_responses["extraction_trace"] = extraction_query["openrouter_response"]

    if video_path is not None and not args.keep_videos and video_path.exists():
        video_path.unlink()
    if (
        hint_video_path is not None
        and not args.keep_videos
        and hint_video_path.exists()
    ):
        hint_video_path.unlink()

    entry = {
        "skill_index": int(skill_index),
        "skill": skill,
        "start_step": start_step,
        "end_step": end_step,
        "semantic_target": semantic_target,
        "end_effector_trace": end_effector_trace,
        "sampled_frame_indices": sampled_frame_indices,
        "raw_model_responses": raw_model_responses,
    }
    if contact_prediction_query_enabled:
        entry["contact_prediction"] = contact_prediction
    if not args.no_contact_prediction:
        entry["prediction_trace"] = prediction_trace
    if not args.no_contact_extraction:
        entry["extraction_trace"] = extraction_trace
    if args.store_full_openrouter_response:
        entry["openrouter_responses"] = full_openrouter_responses

    return entry


def main() -> int:
    args = parse_args()
    if args.skip_existing and args.overwrite_existing:
        raise ValueError("Use either --skip-existing or --overwrite-existing, not both.")
    if (args.query_image_width is None) != (args.query_image_height is None):
        raise ValueError("Use both --query-image-width and --query-image-height together, or omit both for no resize.")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be positive.")
    if args.ee_max_step_delta_pixels <= 0:
        raise ValueError("--ee-max-step-delta-pixels must be positive.")
    if not (0.0 <= args.ee_max_out_of_bounds_fraction <= 1.0):
        raise ValueError("--ee-max-out-of-bounds-fraction must be in [0, 1].")
    if args.ee_max_reprojection_error_meters <= 0:
        raise ValueError("--ee-max-reprojection-error-meters must be positive.")
    if args.ee_max_rounded_reprojection_error_meters <= 0:
        raise ValueError("--ee-max-rounded-reprojection-error-meters must be positive.")
    if not args.no_ee_trace and args.image_key != "image":
        raise ValueError(
            "Dense projected EE traces currently support --image-key image only because wrist-image calibration "
            "is frame-dependent and not recoverable exactly from the LeRobot state vector alone."
        )
    semantic_target_enabled = True

    print("Default annotation model: ", DEFAULT_OPENROUTER_MODEL)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    annotation_dir = args.annotation_dir.expanduser().resolve()
    skill_annotations_path = (
        args.skill_annotations.expanduser().resolve()
        if args.skill_annotations is not None
        else annotation_dir / "skill_annotations.json"
    )
    skill_data, skill_episodes = load_skill_annotation_episodes(skill_annotations_path)
    repo_id = str(skill_data.get("source_repo_id", args.repo_id))
    args.repo_id = repo_id
    dataset_root = resolve_dataset_root_from_skill_data(args, skill_data)

    max_annotated_episode = max(skill_episodes)
    start_episode = max(0, int(args.start_episode))
    end_episode = max_annotated_episode + 1 if args.end_episode is None else int(args.end_episode)
    if not (0 <= start_episode < end_episode):
        raise ValueError(f"Invalid episode range [{start_episode}, {end_episode}).")

    requested_indices = list(range(start_episode, end_episode))
    selected_indices = [idx for idx in requested_indices if idx in skill_episodes]
    if not selected_indices:
        raise ValueError(f"No skill-annotated episodes found in requested range [{start_episode}, {end_episode}).")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else annotation_dir / f"target_trace_run_{start_episode}_{end_episode}"
    )
    shard_dir = output_dir / "episode_shards"
    error_dir = output_dir / "errors"
    video_dir = output_dir / "videos"
    trace_scene_dir = output_dir / "target_trace_scenes"
    shard_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)
    if args.keep_videos:
        video_dir.mkdir(parents=True, exist_ok=True)
    if not args.disable_saving_trace_scenes:
        trace_scene_dir.mkdir(parents=True, exist_ok=True)

    ensure_manifest(
        output_dir / "run_manifest.json",
        args=args,
        skill_annotations_path=skill_annotations_path,
        dataset_root=dataset_root,
        start_episode=start_episode,
        end_episode=end_episode,
        selected_count=len(selected_indices),
    )

    print(f"dataset_root={dataset_root}", flush=True)
    print(f"skill_annotations={skill_annotations_path}", flush=True)
    print(f"annotated_episodes={len(skill_episodes)}, range=[{start_episode}, {end_episode})", flush=True)
    print(f"selected_episodes={len(selected_indices)}", flush=True)
    print(f"trace_frame_count={args.trace_frame_count}", flush=True)
    print(f"semantic_target_enabled={semantic_target_enabled}", flush=True)
    print(f"contact_prediction_enabled={not args.no_contact_prediction}", flush=True)
    print(f"contact_extraction_enabled={not args.no_contact_extraction}", flush=True)
    print(f"contact_prediction_only_enabled={bool(args.predict_contact_only)}", flush=True)
    print(f"use_place_video={bool(args.use_place_video)}", flush=True)
    print(f"end_effector_trace_enabled={not args.no_ee_trace}", flush=True)
    if args.query_image_width is None:
        print("query_image_size=original (no resize)", flush=True)
    else:
        print(f"query_image_size={args.query_image_width}x{args.query_image_height}", flush=True)
    print(
        f"model_coordinate_grid={args.model_coordinate_resolution}x{args.model_coordinate_resolution}",
        flush=True,
    )
    print(f"output_dir={output_dir}", flush=True)

    records = load_episode_records(dataset_root)
    episode_bounds = cumulative_episode_bounds(records)
    print("loading lerobot dataset...", flush=True)
    dataset = load_lerobot_dataset(repo_id, dataset_root)
    print("dataset loaded", flush=True)

    bddl_index: dict[str, list[Path]] = {}
    ee_calibration_cache: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    if not args.no_ee_trace:
        bddl_index = build_bddl_language_index(args.libero_bddl_root)
        if not bddl_index:
            raise ValueError("Could not find any LIBERO BDDL files for EE camera calibration.")
        print(f"indexed_bddl_tasks={len(bddl_index)} for EE projection", flush=True)

    processed = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    for counter, episode_index in enumerate(selected_indices, start=1):
        skill_episode = skill_episodes[episode_index]
        validate_skill_episode_shape(skill_episode)
        record = records[episode_index]
        shard_path = episode_shard_path(shard_dir, episode_index)
        error_path = error_dir / f"episode_{episode_index:06d}.error.json"
        episode_trace_scene_dir = (
            target_trace_scene_episode_dir(trace_scene_dir, episode_index)
            if not args.disable_saving_trace_scenes
            else None
        )

        if shard_path.exists():
            if episode_trace_scene_dir is not None and args.skip_existing and not args.overwrite_existing:
                try:
                    existing_episode = load_json(shard_path)
                    expected_paths = target_trace_scene_paths(
                        trace_scene_dir,
                        episode_index=episode_index,
                        target_trace_entries=existing_episode["target_traces"],
                    )
                    if not all(path.exists() for path in expected_paths):
                        save_target_trace_scene_images(
                            dataset,
                            record=record,
                            episode_bounds=episode_bounds,
                            target_trace_entries=existing_episode["target_traces"],
                            output_dir=trace_scene_dir,
                            image_key=args.image_key,
                        )
                except Exception as exc:
                    print(f"  warning: could not backfill trace scenes for episode {episode_index}: {exc}", flush=True)
            if args.skip_existing and not args.overwrite_existing:
                skipped += 1
                print(f"[skip] episode={episode_index} target-trace shard already exists", flush=True)
                continue
            if args.overwrite_existing:
                shard_path.unlink()

        if episode_trace_scene_dir is not None and episode_trace_scene_dir.exists():
            shutil.rmtree(episode_trace_scene_dir)

        episode_t0 = time.time()
        print(
            f"[{counter}/{len(selected_indices)}] episode={episode_index} "
            f"task_index={skill_episode['task_index']} num_steps={skill_episode['num_steps']} "
            f"skills={len(skill_episode['segments'])}",
            flush=True,
        )

        try:
            first_start_step = int(skill_episode["segments"][0]["start_step"])
            first_frame = load_episode_frame(
                dataset,
                record=record,
                episode_bounds=episode_bounds,
                local_step=first_start_step,
                image_key=args.image_key,
            )
            image_height, image_width = first_frame.shape[:2]
            ee_camera_calibration = get_episode_ee_camera_calibration(
                args=args,
                skill_episode=skill_episode,
                image_width=int(image_width),
                image_height=int(image_height),
                bddl_index=bddl_index,
                calibration_cache=ee_calibration_cache,
            )
            target_trace_entries: list[dict[str, Any]] = []
            for skill_index, segment in enumerate(skill_episode["segments"]):
                print(
                    f"  skill={skill_index} interval=[{segment['start_step']}, {segment['end_step']}) "
                    f"{segment['skill']}",
                    flush=True,
                )
                entry = annotate_skill_segment(
                    api_key=api_key,
                    args=args,
                    dataset=dataset,
                    record=record,
                    episode_bounds=episode_bounds,
                    skill_episode=skill_episode,
                    segment=segment,
                    skill_index=skill_index,
                    image_width=int(image_width),
                    image_height=int(image_height),
                    output_dir=output_dir,
                    video_dir=video_dir,
                    ee_camera_calibration=ee_camera_calibration,
                )
                target_trace_entries.append(entry)

            episode_target_trace = build_episode_target_trace_annotation(
                skill_episode=skill_episode,
                target_trace_entries=target_trace_entries,
                image_key=args.image_key,
                image_width=int(image_width),
                image_height=int(image_height),
                query_image_width=args.query_image_width,
                query_image_height=args.query_image_height,
                model_coordinate_width=args.model_coordinate_resolution,
                model_coordinate_height=args.model_coordinate_resolution,
                trace_frame_count=args.trace_frame_count,
                model=args.model,
                source_repo_id=repo_id,
                dataset_root=dataset_root,
                skill_annotation_source=skill_annotations_path,
                semantic_target_enabled=semantic_target_enabled,
                contact_prediction_enabled=not args.no_contact_prediction,
                contact_extraction_enabled=not args.no_contact_extraction,
                contact_prediction_only_enabled=bool(args.predict_contact_only),
                end_effector_trace_enabled=not args.no_ee_trace,
                ee_projection_camera=ee_camera_calibration,
                prompt_version=TARGET_TRACE_PROMPT_VERSION,
            )
            if episode_trace_scene_dir is not None:
                save_target_trace_scene_images(
                    dataset,
                    record=record,
                    episode_bounds=episode_bounds,
                    target_trace_entries=target_trace_entries,
                    output_dir=trace_scene_dir,
                    image_key=args.image_key,
                )
            save_json_atomic(shard_path, episode_target_trace)
            if error_path.exists():
                error_path.unlink()

            processed += 1
            elapsed = time.time() - episode_t0
            trace_point_count = 0
            for entry in target_trace_entries:
                for key in ("prediction_trace", "extraction_trace", "end_effector_trace"):
                    trace_obj = entry.get(key)
                    if isinstance(trace_obj, dict):
                        trace_point_count += len(trace_obj.get("trace", []))
            print(
                f"  saved {shard_path.name} with {len(target_trace_entries)} skill target(s) "
                f"and {trace_point_count} trace point(s) in {elapsed:.1f}s",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            error_payload = {
                "episode_index": episode_index,
                "task_index": skill_episode.get("task_index"),
                "instruction": skill_episode.get("instruction"),
                "num_steps": skill_episode.get("num_steps"),
                "skill_annotations": str(skill_annotations_path),
                "error": f"{type(exc).__name__}: {exc}",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            save_json_atomic(error_path, error_payload)
            print(f"  failed episode={episode_index}: {error_payload['error']}", flush=True)

    total_elapsed = time.time() - start_time
    print(
        f"done: processed={processed} skipped={skipped} failed={failed} elapsed={total_elapsed/60.0:.1f}m",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
