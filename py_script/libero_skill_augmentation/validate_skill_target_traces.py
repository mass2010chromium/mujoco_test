#!/usr/bin/env python3
"""
Structural validator for LIBERO semantic target and contact trace annotations.

This checks that skill_target_traces.json remains aligned with skill_annotations.json
and that every saved target point / trace point is inside the side-view image.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import DEFAULT_REPO_ID, list_episode_shards, load_json, save_json_atomic  # noqa: E402
from common_trace import (  # noqa: E402
    DEFAULT_EE_TRACE_MAX_OUT_OF_BOUNDS_FRACTION,
    DEFAULT_EE_REPROJECTION_MAX_ERROR_METERS,
    DEFAULT_EE_ROUNDED_REPROJECTION_MAX_ERROR_METERS,
    DEFAULT_EE_TRACE_MAX_STEP_DELTA_PIXELS,
    load_skill_annotation_episodes,
    reproject_lerobot_image_points_to_world,
)


@dataclass
class ValidationFailure:
    source: str
    episode_index: int | None
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate LIBERO skill target/trace annotation files against skill_annotations.json. "
            "Accepts combined skill_target_traces.json files or directories of target-trace episode shards."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Target-trace file(s) or directory/directories containing episode shard files.",
    )
    parser.add_argument(
        "--skill-annotations",
        type=Path,
        default=None,
        help=(
            "Skill annotation JSON to validate against. If omitted, the validator tries the "
            "skill_annotation_file metadata in skill_target_traces.json, then nearby skill_annotations.json files."
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional validation result JSON path.")
    parser.add_argument("--stop-on-first-error", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Unused; retained for tooling consistency.")
    parser.add_argument(
        "--max-ee-step-delta-pixels",
        type=float,
        default=DEFAULT_EE_TRACE_MAX_STEP_DELTA_PIXELS,
        help="Maximum allowed adjacent-frame pixel jump for dense end-effector traces.",
    )
    parser.add_argument(
        "--max-ee-out-of-bounds-fraction",
        type=float,
        default=DEFAULT_EE_TRACE_MAX_OUT_OF_BOUNDS_FRACTION,
        help="Maximum allowed raw projected out-of-bounds fraction when this metadata is present.",
    )
    parser.add_argument(
        "--max-ee-reprojection-error-meters",
        type=float,
        default=DEFAULT_EE_REPROJECTION_MAX_ERROR_METERS,
        help="Maximum allowed exact EE pixel-to-3D reprojection error when metadata is present.",
    )
    parser.add_argument(
        "--max-ee-rounded-reprojection-error-meters",
        type=float,
        default=DEFAULT_EE_ROUNDED_REPROJECTION_MAX_ERROR_METERS,
        help="Maximum allowed rounded in-bounds EE pixel-to-3D reprojection error when metadata is present.",
    )
    return parser.parse_args()


def expand_inputs(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_path in inputs:
        path = input_path.expanduser().resolve()
        if path.is_dir():
            files.extend(list_episode_shards(path))
        else:
            files.append(path)
    return files


def infer_skill_annotations_path(inputs: list[Path]) -> Path | None:
    for input_path in inputs:
        path = input_path.expanduser().resolve()
        candidate_paths: list[Path] = []
        if path.is_file():
            try:
                data = load_json(path)
                if isinstance(data, dict) and data.get("skill_annotation_file"):
                    candidate_paths.append(Path(str(data["skill_annotation_file"])).expanduser())
            except Exception:
                pass
            candidate_paths.append(path.parent / "skill_annotations.json")
        else:
            candidate_paths.append(path.parent / "skill_annotations.json")
            candidate_paths.append(path.parent.parent / "skill_annotations.json")

        for candidate in candidate_paths:
            if candidate.exists():
                return candidate.resolve()
    return None


def target_episodes_from_file(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = load_json(path)
    if isinstance(data, dict) and "episode_index" in data and "target_traces" in data:
        return {}, [data]

    if isinstance(data, dict):
        episodes: list[dict[str, Any]] = []
        for key in sorted((item for item in data if isinstance(item, str) and item.isdigit()), key=lambda item: int(item)):
            value = data[key]
            if isinstance(value, dict):
                episode = dict(value)
                episode.setdefault("episode_index", int(key))
                episodes.append(episode)
        if episodes:
            return data, episodes

    raise ValueError(f"Unsupported target-trace JSON structure in {path}")


def coerce_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric, got boolean {value!r}.")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric, got {type(value).__name__}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}.")
    return number


def coerce_point(point: Any, *, field_name: str) -> tuple[float, float]:
    if not isinstance(point, list) or len(point) != 2:
        raise ValueError(f"{field_name} must be [x, y], got {point!r}.")
    x = coerce_number(point[0], field_name=f"{field_name}[0]")
    y = coerce_number(point[1], field_name=f"{field_name}[1]")
    return x, y


def resolve_image_bounds(target_episode: dict[str, Any], top_meta: dict[str, Any]) -> tuple[int, int]:
    width = target_episode.get("image_width", top_meta.get("image_width"))
    height = target_episode.get("image_height", top_meta.get("image_height"))
    try:
        width_int = int(width)
        height_int = int(height)
    except Exception as exc:
        raise ValueError(f"Could not resolve image bounds: width={width!r}, height={height!r}") from exc
    if width_int <= 0 or height_int <= 0:
        raise ValueError(f"Image bounds must be positive, got width={width_int}, height={height_int}.")
    return width_int, height_int


def add_failure(failures: list[ValidationFailure], source: Path, episode_index: int | None, message: str) -> None:
    failures.append(ValidationFailure(str(source), episode_index, message))


def validate_point_object(
    *,
    point_obj: Any,
    image_width: int,
    image_height: int,
    required: bool,
    field_name: str,
) -> list[str]:
    errors: list[str] = []
    if point_obj is None:
        if required:
            errors.append(f"{field_name} is missing.")
        return errors
    if not isinstance(point_obj, dict):
        errors.append(f"{field_name} must be an object or null.")
        return errors
    try:
        x, y = coerce_point(point_obj.get("point"), field_name=f"{field_name}.point")
        if not (0 <= x < image_width and 0 <= y < image_height):
            errors.append(
                f"{field_name}.point={point_obj.get('point')!r} is outside image bounds "
                f"width={image_width}, height={image_height}."
            )
    except Exception as exc:
        errors.append(str(exc))
    return errors


def validate_trace_object(
    *,
    trace_obj: Any,
    image_width: int,
    image_height: int,
    start_step: int,
    end_step: int,
    sampled_frame_indices: list[int],
    field_name: str,
) -> list[str]:
    errors: list[str] = []
    if trace_obj is None:
        return errors
    if not isinstance(trace_obj, dict):
        return [f"{field_name} must be an object or null."]

    trace = trace_obj.get("trace")
    frame_indices = trace_obj.get("frame_indices")
    contact_start_step = trace_obj.get("contact_start_step")
    if not isinstance(trace, list) or not trace:
        errors.append(f"{field_name}.trace must be a non-empty list.")
        return errors
    if not isinstance(frame_indices, list) or len(frame_indices) != len(trace):
        errors.append(f"{field_name}.frame_indices must be a list with the same length as trace.")
        return errors
    try:
        contact_start = int(contact_start_step)
    except Exception:
        errors.append(f"{field_name}.contact_start_step must be an integer.")
        contact_start = start_step
    sampled_set = {int(idx) for idx in sampled_frame_indices}
    if contact_start not in sampled_set:
        errors.append(f"{field_name}.contact_start_step={contact_start} is not in sampled_frame_indices.")
    if not (start_step <= contact_start < end_step):
        errors.append(f"{field_name}.contact_start_step={contact_start} is outside [{start_step}, {end_step}).")

    coerced_frames: list[int] = []
    for idx, frame_index in enumerate(frame_indices):
        try:
            frame = int(frame_index)
        except Exception:
            errors.append(f"{field_name}.frame_indices[{idx}]={frame_index!r} is not an integer.")
            continue
        coerced_frames.append(frame)
        if frame not in sampled_set:
            errors.append(f"{field_name}.frame_indices[{idx}]={frame} is not in sampled_frame_indices.")
        if not (contact_start <= frame < end_step):
            errors.append(f"{field_name}.frame_indices[{idx}]={frame} is outside contact interval.")

    if coerced_frames and coerced_frames != sorted(coerced_frames):
        errors.append(f"{field_name}.frame_indices must be sorted.")
    if len(set(coerced_frames)) != len(coerced_frames):
        errors.append(f"{field_name}.frame_indices contain duplicates.")
    if coerced_frames and coerced_frames[0] != contact_start:
        errors.append(f"{field_name}.frame_indices[0] must equal contact_start_step.")

    model_trace = trace_obj.get("model_trace")
    if model_trace is not None and (not isinstance(model_trace, list) or len(model_trace) != len(trace)):
        errors.append(f"{field_name}.model_trace length must match trace when present.")

    for point_idx, point in enumerate(trace):
        try:
            x, y = coerce_point(point, field_name=f"{field_name}.trace[{point_idx}]")
        except Exception as exc:
            errors.append(str(exc))
            continue
        if not (0 <= x < image_width and 0 <= y < image_height):
            errors.append(
                f"{field_name}.trace[{point_idx}]={point!r} is outside image bounds "
                f"width={image_width}, height={image_height}."
            )
    return errors


def validate_dense_ee_trace_object(
    *,
    trace_obj: Any,
    image_width: int,
    image_height: int,
    start_step: int,
    end_step: int,
    max_step_delta_pixels: float,
    max_out_of_bounds_fraction: float,
    max_reprojection_error_meters: float,
    max_rounded_reprojection_error_meters: float,
    camera_calibration: dict[str, Any] | None,
    field_name: str,
) -> list[str]:
    errors: list[str] = []
    if trace_obj is None:
        return [f"{field_name} is missing."]
    if not isinstance(trace_obj, dict):
        return [f"{field_name} must be an object."]

    trace = trace_obj.get("trace")
    frame_indices = trace_obj.get("frame_indices")
    expected_frames = list(range(start_step, end_step))
    expected_len = end_step - start_step
    if not isinstance(trace, list) or len(trace) != expected_len:
        errors.append(f"{field_name}.trace must contain exactly {expected_len} dense points.")
        return errors
    if frame_indices is not None and frame_indices != expected_frames:
        # Only enforce when the field is present (shard files retain it; the
        # combined skill_target_traces.json strips it because it's recoverable
        # from start_step/end_step).
        errors.append(f"{field_name}.frame_indices must equal every frame in [{start_step}, {end_step}).")

    raw_trace = trace_obj.get("raw_trace")
    if raw_trace is not None and (not isinstance(raw_trace, list) or len(raw_trace) != len(trace)):
        errors.append(f"{field_name}.raw_trace length must match trace when present.")
    in_bounds = trace_obj.get("in_bounds")
    if in_bounds is not None and (not isinstance(in_bounds, list) or len(in_bounds) != len(trace)):
        errors.append(f"{field_name}.in_bounds length must match trace when present.")
    projection_depth = trace_obj.get("projection_depth")
    if projection_depth is not None and (not isinstance(projection_depth, list) or len(projection_depth) != len(trace)):
        errors.append(f"{field_name}.projection_depth length must match trace when present.")
    source_world_positions = trace_obj.get("source_world_positions")
    if source_world_positions is not None and (
        not isinstance(source_world_positions, list) or len(source_world_positions) != len(trace)
    ):
        errors.append(f"{field_name}.source_world_positions length must match trace when present.")

    try:
        out_of_bounds_fraction = float(trace_obj.get("out_of_bounds_fraction", 0.0))
        if out_of_bounds_fraction > max_out_of_bounds_fraction:
            errors.append(
                f"{field_name}.out_of_bounds_fraction={out_of_bounds_fraction:.3f} exceeds "
                f"{max_out_of_bounds_fraction:.3f}."
            )
    except Exception:
        errors.append(f"{field_name}.out_of_bounds_fraction must be numeric when present.")

    previous_point: tuple[float, float] | None = None
    for point_idx, point in enumerate(trace):
        try:
            x, y = coerce_point(point, field_name=f"{field_name}.trace[{point_idx}]")
        except Exception as exc:
            errors.append(str(exc))
            continue
        if not (0 <= x < image_width and 0 <= y < image_height):
            errors.append(
                f"{field_name}.trace[{point_idx}]={point!r} is outside image bounds "
                f"width={image_width}, height={image_height}."
            )
        if previous_point is not None:
            delta = math.hypot(x - previous_point[0], y - previous_point[1])
            if delta > max_step_delta_pixels:
                errors.append(
                    f"{field_name}.trace[{point_idx}] jumps {delta:.2f}px from previous point, "
                    f"above {max_step_delta_pixels:.2f}px."
                )
        previous_point = (x, y)

    reprojection_checks = trace_obj.get("reprojection_checks")
    if isinstance(reprojection_checks, dict):
        try:
            exact_error = float(reprojection_checks.get("exact_max_error_meters", 0.0))
            if exact_error > max_reprojection_error_meters:
                errors.append(
                    f"{field_name}.reprojection_checks.exact_max_error_meters={exact_error:.3e} "
                    f"exceeds {max_reprojection_error_meters:.3e}."
                )
        except Exception:
            errors.append(f"{field_name}.reprojection_checks.exact_max_error_meters must be numeric.")
        try:
            rounded_error = float(reprojection_checks.get("rounded_in_bounds_max_error_meters", 0.0))
            if rounded_error > max_rounded_reprojection_error_meters:
                errors.append(
                    f"{field_name}.reprojection_checks.rounded_in_bounds_max_error_meters={rounded_error:.3e} "
                    f"exceeds {max_rounded_reprojection_error_meters:.3e}."
                )
        except Exception:
            errors.append(f"{field_name}.reprojection_checks.rounded_in_bounds_max_error_meters must be numeric.")

    if (
        camera_calibration is not None
        and isinstance(raw_trace, list)
        and isinstance(projection_depth, list)
        and isinstance(source_world_positions, list)
        and len(raw_trace) == len(projection_depth) == len(source_world_positions) == len(trace)
    ):
        try:
            import numpy as np

            raw_xy = np.asarray(raw_trace, dtype=float)
            depth = np.asarray(projection_depth, dtype=float)
            source_world = np.asarray(source_world_positions, dtype=float)
            reprojected_world = reproject_lerobot_image_points_to_world(
                raw_xy,
                depth=depth,
                world_to_camera_transform=camera_calibration["world_to_camera_transform"],
                image_width=image_width,
                image_height=image_height,
                image_convention=camera_calibration.get("image_convention"),
            )
            exact_errors = np.linalg.norm(reprojected_world - source_world, axis=1)
            exact_max_error = float(np.max(exact_errors)) if len(exact_errors) else 0.0
            if exact_max_error > max_reprojection_error_meters:
                errors.append(
                    f"{field_name} exact reprojected world error {exact_max_error:.3e}m exceeds "
                    f"{max_reprojection_error_meters:.3e}m."
                )

            if isinstance(in_bounds, list):
                mask = np.asarray([bool(value) for value in in_bounds], dtype=bool)
                if np.any(mask):
                    rounded_xy = np.asarray(trace, dtype=float)[mask]
                    rounded_reprojected = reproject_lerobot_image_points_to_world(
                        rounded_xy,
                        depth=depth[mask],
                        world_to_camera_transform=camera_calibration["world_to_camera_transform"],
                        image_width=image_width,
                        image_height=image_height,
                        image_convention=camera_calibration.get("image_convention"),
                    )
                    rounded_errors = np.linalg.norm(rounded_reprojected - source_world[mask], axis=1)
                    rounded_max_error = float(np.max(rounded_errors)) if len(rounded_errors) else 0.0
                    if rounded_max_error > max_rounded_reprojection_error_meters:
                        errors.append(
                            f"{field_name} rounded in-bounds reprojected world error {rounded_max_error:.3e}m "
                            f"exceeds {max_rounded_reprojection_error_meters:.3e}m."
                        )
        except Exception as exc:
            errors.append(f"{field_name} reprojection validation failed: {exc}")
    return errors


def validate_episode(
    *,
    target_episode: dict[str, Any],
    skill_episode: dict[str, Any] | None,
    top_meta: dict[str, Any],
    source: Path,
    max_ee_step_delta_pixels: float,
    max_ee_out_of_bounds_fraction: float,
    max_ee_reprojection_error_meters: float,
    max_ee_rounded_reprojection_error_meters: float,
) -> list[ValidationFailure]:
    failures: list[ValidationFailure] = []
    episode_index = int(target_episode.get("episode_index", -1)) if "episode_index" in target_episode else None

    if skill_episode is None:
        add_failure(failures, source, episode_index, "No matching episode in skill_annotations.json.")
        return failures

    try:
        image_width, image_height = resolve_image_bounds(target_episode, top_meta)
    except Exception as exc:
        add_failure(failures, source, episode_index, str(exc))
        return failures

    skill_segments = skill_episode.get("segments")
    target_segments = target_episode.get("segments")
    target_traces = target_episode.get("target_traces")
    if not isinstance(skill_segments, list) or not skill_segments:
        add_failure(failures, source, episode_index, "Skill episode has missing or empty segments.")
        return failures
    if not isinstance(target_segments, list) or not isinstance(target_traces, list):
        add_failure(failures, source, episode_index, "Target episode must contain segments and target_traces lists.")
        return failures

    if int(target_episode.get("num_steps", -1)) != int(skill_episode.get("num_steps", -2)):
        add_failure(
            failures,
            source,
            episode_index,
            f"Target num_steps={target_episode.get('num_steps')} does not match skill num_steps={skill_episode.get('num_steps')}.",
        )
    if target_segments != skill_segments:
        add_failure(failures, source, episode_index, "Target segments do not exactly match skill annotation segments.")
    if len(target_traces) != len(skill_segments):
        add_failure(
            failures,
            source,
            episode_index,
            f"Target trace count {len(target_traces)} does not equal skill count {len(skill_segments)}.",
        )

    semantic_enabled = bool(target_episode.get("semantic_target_enabled", top_meta.get("semantic_target_enabled", True)))
    prediction_enabled = bool(target_episode.get("contact_prediction_enabled", top_meta.get("contact_prediction_enabled", True)))
    extraction_enabled = bool(target_episode.get("contact_extraction_enabled", top_meta.get("contact_extraction_enabled", True)))
    contact_prediction_only_enabled = bool(
        target_episode.get(
            "contact_prediction_only_enabled",
            top_meta.get("contact_prediction_only_enabled", False),
        )
    )
    contact_prediction_required = prediction_enabled or contact_prediction_only_enabled
    ee_trace_enabled = bool(target_episode.get("end_effector_trace_enabled", top_meta.get("end_effector_trace_enabled", False)))
    ee_projection_camera = target_episode.get("ee_projection_camera", top_meta.get("ee_projection_camera"))
    if ee_projection_camera is not None and not isinstance(ee_projection_camera, dict):
        add_failure(failures, source, episode_index, "ee_projection_camera must be an object when present.")
        ee_projection_camera = None

    total_semantic_points = 0
    total_trace_points = 0
    total_ee_trace_points = 0
    for idx, segment in enumerate(skill_segments):
        if idx >= len(target_traces):
            continue
        entry = target_traces[idx]
        if not isinstance(entry, dict):
            add_failure(failures, source, episode_index, f"Target trace {idx} is not an object.")
            continue

        start_step = int(segment["start_step"])
        end_step = int(segment["end_step"])
        if int(entry.get("skill_index", -1)) != idx:
            add_failure(failures, source, episode_index, f"Entry {idx} has wrong skill_index={entry.get('skill_index')}.")
        if str(entry.get("skill")) != str(segment["skill"]):
            add_failure(failures, source, episode_index, f"Entry {idx} skill does not match skill segment.")
        if int(entry.get("start_step", -1)) != start_step or int(entry.get("end_step", -1)) != end_step:
            add_failure(failures, source, episode_index, f"Entry {idx} interval does not match skill segment.")

        sampled_frame_indices = entry.get("sampled_frame_indices")
        # The combined skill_target_traces.json strips sampled_frame_indices on
        # entries with no contact-point trace, so only require it when at least
        # one contact-point trace is actually saved on this entry.
        entry_has_contact_trace = (
            isinstance(entry.get("prediction_trace"), dict)
            or isinstance(entry.get("extraction_trace"), dict)
        )
        if entry_has_contact_trace and (
            not isinstance(sampled_frame_indices, list) or not sampled_frame_indices
        ):
            add_failure(failures, source, episode_index, f"Entry {idx} has missing sampled_frame_indices.")
            sampled_frame_indices = []
        elif not isinstance(sampled_frame_indices, list):
            sampled_frame_indices = []

        for message in validate_point_object(
            point_obj=entry.get("semantic_target"),
            image_width=image_width,
            image_height=image_height,
            required=semantic_enabled,
            field_name=f"entry {idx}.semantic_target",
        ):
            add_failure(failures, source, episode_index, message)
        if isinstance(entry.get("semantic_target"), dict):
            total_semantic_points += 1

        for message in validate_point_object(
            point_obj=entry.get("contact_prediction"),
            image_width=image_width,
            image_height=image_height,
            required=contact_prediction_required,
            field_name=f"entry {idx}.contact_prediction",
        ):
            add_failure(failures, source, episode_index, message)

        if prediction_enabled and entry.get("prediction_trace") is None:
            add_failure(failures, source, episode_index, f"Entry {idx} is missing prediction_trace.")
        if extraction_enabled and entry.get("extraction_trace") is None:
            add_failure(failures, source, episode_index, f"Entry {idx} is missing extraction_trace.")
        if ee_trace_enabled and entry.get("end_effector_trace") is None:
            add_failure(failures, source, episode_index, f"Entry {idx} is missing end_effector_trace.")

        for trace_field in ("prediction_trace", "extraction_trace"):
            trace_obj = entry.get(trace_field)
            if isinstance(trace_obj, dict):
                total_trace_points += len(trace_obj.get("trace", []))
            for message in validate_trace_object(
                trace_obj=trace_obj,
                image_width=image_width,
                image_height=image_height,
                start_step=start_step,
                end_step=end_step,
                sampled_frame_indices=sampled_frame_indices,
                field_name=f"entry {idx}.{trace_field}",
            ):
                add_failure(failures, source, episode_index, message)

        if ee_trace_enabled:
            ee_trace = entry.get("end_effector_trace")
            if isinstance(ee_trace, dict):
                total_ee_trace_points += len(ee_trace.get("trace", []))
            for message in validate_dense_ee_trace_object(
                trace_obj=ee_trace,
                image_width=image_width,
                image_height=image_height,
                start_step=start_step,
                end_step=end_step,
                max_step_delta_pixels=max_ee_step_delta_pixels,
                max_out_of_bounds_fraction=max_ee_out_of_bounds_fraction,
                max_reprojection_error_meters=max_ee_reprojection_error_meters,
                max_rounded_reprojection_error_meters=max_ee_rounded_reprojection_error_meters,
                camera_calibration=ee_projection_camera,
                field_name=f"entry {idx}.end_effector_trace",
            ):
                add_failure(failures, source, episode_index, message)

    if semantic_enabled and total_semantic_points != len(skill_segments):
        add_failure(
            failures,
            source,
            episode_index,
            f"Episode has {total_semantic_points} semantic targets for {len(skill_segments)} skill segments.",
        )
    if (prediction_enabled or extraction_enabled) and total_trace_points == 0:
        add_failure(failures, source, episode_index, "Episode has no contact trace points.")
    if ee_trace_enabled and total_ee_trace_points != int(skill_episode.get("num_steps", 0)):
        add_failure(
            failures,
            source,
            episode_index,
            f"Episode has {total_ee_trace_points} dense EE trace points for {skill_episode.get('num_steps')} steps.",
        )

    return failures


def default_output_path(raw_inputs: list[Path]) -> Path:
    if len(raw_inputs) == 1:
        resolved = raw_inputs[0].expanduser().resolve()
        return resolved.parent / "target_trace_validation_results.json"
    return Path("target_trace_validation_results.json").resolve()


def main() -> int:
    args = parse_args()
    if args.max_ee_step_delta_pixels <= 0:
        raise ValueError("--max-ee-step-delta-pixels must be positive.")
    if not (0.0 <= args.max_ee_out_of_bounds_fraction <= 1.0):
        raise ValueError("--max-ee-out-of-bounds-fraction must be in [0, 1].")
    if args.max_ee_reprojection_error_meters <= 0:
        raise ValueError("--max-ee-reprojection-error-meters must be positive.")
    if args.max_ee_rounded_reprojection_error_meters <= 0:
        raise ValueError("--max-ee-rounded-reprojection-error-meters must be positive.")
    skill_annotations_path = (
        args.skill_annotations.expanduser().resolve()
        if args.skill_annotations is not None
        else infer_skill_annotations_path(args.inputs)
    )
    if skill_annotations_path is None:
        raise ValueError("Could not infer skill_annotations.json. Pass --skill-annotations explicitly.")
    _, skill_episodes = load_skill_annotation_episodes(skill_annotations_path)

    files = expand_inputs(args.inputs)
    if not files:
        raise ValueError("No target-trace files found in inputs.")

    failures: list[ValidationFailure] = []
    checked_episodes = 0
    for path in files:
        top_meta, target_episodes = target_episodes_from_file(path)
        for target_episode in target_episodes:
            episode_index = int(target_episode["episode_index"])
            episode_failures = validate_episode(
                target_episode=target_episode,
                skill_episode=skill_episodes.get(episode_index),
                top_meta=top_meta,
                source=path,
                max_ee_step_delta_pixels=args.max_ee_step_delta_pixels,
                max_ee_out_of_bounds_fraction=args.max_ee_out_of_bounds_fraction,
                max_ee_reprojection_error_meters=args.max_ee_reprojection_error_meters,
                max_ee_rounded_reprojection_error_meters=args.max_ee_rounded_reprojection_error_meters,
            )
            checked_episodes += 1
            failures.extend(episode_failures)
            if episode_failures and not args.quiet:
                print(f"[fail] episode={episode_index} source={path}", flush=True)
                for failure in episode_failures[:10]:
                    print(f"  - {failure.message}", flush=True)
            if episode_failures and args.stop_on_first_error:
                break
        if failures and args.stop_on_first_error:
            break

    result = {
        "skill_annotations": str(skill_annotations_path),
        "checked_files": [str(path) for path in files],
        "checked_episodes": checked_episodes,
        "ok": not failures,
        "failure_count": len(failures),
        "failures": [asdict(failure) for failure in failures],
    }
    output_path = args.output.expanduser().resolve() if args.output is not None else default_output_path(args.inputs)
    save_json_atomic(output_path, result)

    if not args.quiet or failures:
        print(
            f"checked_episodes={checked_episodes} failures={len(failures)} output={output_path}",
            flush=True,
        )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
