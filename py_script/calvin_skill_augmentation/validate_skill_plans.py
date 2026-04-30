#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
PY_SCRIPT_DIR = THIS_DIR.parent
for path in (THIS_DIR, PY_SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (  # noqa: E402
    DEFAULT_REPO_ID,
    list_episode_shards,
    load_json,
    parse_plan_skills,
    save_json_atomic,
    transition_boundary_steps_from_segments,
    transition_scene_paths,
    utc_now,
    validate_skill_expr,
)
from vla_verify_calvin.pddl_parsing import setup_pddl_simulation  # noqa: E402
from vla_verify_calvin.scene_graph import TaskSceneGraph  # noqa: E402
from vla_verify_calvin.verifier import SubtaskVerificationResult, VLAVerifier  # noqa: E402
from vlm_interfaces import (  # noqa: E402
    get_ollama_interfaces,
    get_openrouter_interfaces,
    get_r4b_interfaces,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate CALVIN skill annotations with the symbolic scene-graph verifier. Supports "
            "single-episode shard JSONs, combined annotation JSONs, and directories of shards."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Annotation file(s) or directory/directories containing episode shard files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the validation summary JSON. Defaults beside the input when a single input is given.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Optional repo id override. Defaults to the annotation metadata, then to the standard CALVIN repo id.",
    )
    parser.add_argument(
        "--transition-scene-root",
        dest="transition_scene_roots",
        type=Path,
        action="append",
        default=None,
        help=(
            "Optional annotation run root or transition_scenes directory. Pass multiple times if a combined "
            "annotation file draws episodes from multiple runs. If omitted, roots are auto-discovered near each input."
        ),
    )
    parser.add_argument(
        "--pddl-path",
        type=Path,
        default=PY_SCRIPT_DIR / "pddl" / "calvin_domain.pddl",
        help="PDDL domain used by the symbolic CALVIN verifier.",
    )
    parser.add_argument(
        "--backend",
        choices=["openrouter", "ollama", "r4b"],
        default="openrouter",
        help="Backend used for the verifier's LLM/VLM interfaces.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only run symbolic plan verification. Skip all skill-transition verification.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Inclusive lower episode index bound to validate.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Exclusive upper episode index bound to validate.",
    )
    parser.add_argument(
        "--stop-on-first-error",
        action="store_true",
        help="Exit immediately after the first failed episode and still save the partial JSON report.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final summary unless a failure is found.",
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


def default_output_path(raw_inputs: list[Path]) -> Path:
    if len(raw_inputs) == 1:
        resolved = raw_inputs[0].expanduser().resolve()
        return resolved.parent / "validation_results.json"
    return Path.cwd() / "validation_results.json"


def load_input_file(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict) and "episode_index" in data and "segments" in data:
        metadata = {
            "source_repo_id": data.get("source_repo_id"),
            "dataset_root": data.get("dataset_root"),
            "schema_version": data.get("schema_version"),
        }
        return [data], metadata

    if isinstance(data, dict):
        metadata = {
            "source_repo_id": data.get("source_repo_id"),
            "dataset_root": data.get("dataset_root"),
            "schema_version": data.get("schema_version"),
            "prompt_version": data.get("prompt_version"),
            "fps": data.get("fps"),
            "boundary_window": data.get("boundary_window"),
        }
        episodes: list[dict[str, Any]] = []
        for key in sorted((k for k in data if isinstance(k, str) and k.isdigit()), key=lambda item: int(item)):
            value = data[key]
            if isinstance(value, dict):
                episode = dict(value)
                episode.setdefault("episode_index", int(key))
                episodes.append(episode)
        if episodes:
            return episodes, metadata

    raise ValueError(f"Unsupported annotation JSON structure in {path}")


def filter_episodes_by_range(
    episodes: list[dict[str, Any]],
    *,
    start: int | None,
    end: int | None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        if start is not None and episode_index < start:
            continue
        if end is not None and episode_index >= end:
            continue
        filtered.append(episode)
    return filtered


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
            raise ValueError(f"Segment plan strings disagree within the same trajectory at index {idx}.")
    return first_plan


def extract_instruction(episode: dict[str, Any]) -> str | None:
    instruction = episode.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()

    raw_instruction = episode.get("raw_instruction")
    if isinstance(raw_instruction, str) and raw_instruction.strip():
        return raw_instruction.strip()

    segments = episode.get("segments")
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            candidate = segment.get("instruction")
            if isinstance(candidate, str) and candidate.strip():
                text = candidate.strip()
                if text.startswith("Instruction: "):
                    return text[len("Instruction: ") :].strip()
                return text
    return None


def canonical_segments_from_episode(episode: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = episode.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("Episode has no segments.")

    canonical: list[dict[str, Any]] = []
    for idx, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            raise ValueError(f"Segment {idx} is not a JSON object.")

        try:
            start_step = int(raw_segment["start_step"])
            end_step = int(raw_segment["end_step"])
        except Exception as exc:
            raise ValueError(f"Segment {idx} is missing integer start/end steps: {raw_segment!r}") from exc
        skill = validate_skill_expr(str(raw_segment["skill"]))
        segment = {"start_step": start_step, "end_step": end_step, "skill": skill}

        if end_step <= start_step:
            raise ValueError(f"Segment {idx} has non-positive length: {segment}")

        if not canonical:
            if start_step != 0:
                raise ValueError(f"First segment must start at 0, got {start_step}.")
            canonical.append(segment)
            continue

        if start_step != canonical[-1]["end_step"]:
            raise ValueError(
                f"Segments must be contiguous. Segment {idx} starts at {start_step}, "
                f"expected {canonical[-1]['end_step']}."
            )

        if skill == canonical[-1]["skill"]:
            canonical[-1]["end_step"] = end_step
        else:
            canonical.append(segment)

    return canonical


def unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def normalize_transition_scene_root(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    if resolved.name == "transition_scenes" and resolved.is_dir():
        return resolved

    candidate = resolved / "transition_scenes"
    if candidate.is_dir():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve a transition_scenes directory from {root}. "
        f"Tried {resolved} and {candidate}."
    )


def discover_transition_scene_roots(source: Path) -> list[Path]:
    candidates: list[Path] = []

    direct_candidates = [
        source.parent / "transition_scenes",
    ]
    if source.parent.name == "episode_shards":
        direct_candidates.append(source.parent.parent / "transition_scenes")

    for candidate in direct_candidates:
        if candidate.is_dir():
            candidates.append(candidate.resolve())

    search_roots = [source.parent]
    if source.parent.name == "episode_shards":
        search_roots.append(source.parent.parent)

    for search_root in unique_paths(search_roots):
        if not search_root.exists():
            continue
        for candidate in search_root.rglob("transition_scenes"):
            if candidate.is_dir():
                candidates.append(candidate.resolve())

    return unique_paths(candidates)


def load_rgb_image(path: Path) -> Any:
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def resolve_transition_scene_files(
    *,
    source: Path,
    episode_index: int,
    canonical_segments: list[dict[str, Any]],
    transition_scene_roots: list[Path],
) -> tuple[Path, list[Path]]:
    candidate_roots = unique_paths(transition_scene_roots)
    if not candidate_roots:
        candidate_roots = discover_transition_scene_roots(source)
    if not candidate_roots:
        raise FileNotFoundError(
            f"Could not find any transition_scenes directories near {source}. "
            "Pass --transition-scene-root explicitly if the saved boundary frames live elsewhere."
        )

    matching_roots: list[tuple[Path, list[Path]]] = []
    for root in candidate_roots:
        paths = transition_scene_paths(
            root,
            episode_index=episode_index,
            segments=canonical_segments,
        )
        if all(path.exists() for path in paths):
            matching_roots.append((root, paths))

    if not matching_roots:
        expected_paths = transition_scene_paths(
            candidate_roots[0],
            episode_index=episode_index,
            segments=canonical_segments,
        )
        sample = ", ".join(str(path) for path in expected_paths[:3])
        raise FileNotFoundError(
            f"Could not find saved transition-scene images for episode {episode_index} from {source}. "
            f"Expected files like: {sample}"
        )

    if len(matching_roots) > 1:
        roots_text = ", ".join(str(root) for root, _ in matching_roots)
        raise ValueError(
            f"Found multiple matching transition_scenes roots for episode {episode_index} from {source}: {roots_text}. "
            "Pass --transition-scene-root to disambiguate."
        )

    return matching_roots[0]


def get_interfaces(backend: str):
    if backend == "openrouter":
        return get_openrouter_interfaces()
    if backend == "ollama":
        return get_ollama_interfaces()
    if backend == "r4b":
        return get_r4b_interfaces()
    raise ValueError(f"Unsupported backend: {backend}")


def serialize_subtask_result(result: SubtaskVerificationResult) -> dict[str, Any]:
    return {
        "subtask": result.subtask,
        "accepted": result.accepted,
        "reason": result.reasoning,
        "grounded_action": result.grounded_action,
    }


def serialize_plan_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "feasible": bool(result["feasible"]),
        "failed_step": result["failed_step"],
        "failure_reason": result["failure_reason"],
        "steps": list(result["steps"]),
        "applied_actions": list(result["applied_actions"]),
        "step_results": [serialize_subtask_result(step_result) for step_result in result["step_results"]],
    }


def build_transition_check(
    *,
    canonical_segments: list[dict[str, Any]],
    segment_idx: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    current_segment = canonical_segments[segment_idx]
    previous_skill = canonical_segments[segment_idx - 1]["skill"] if segment_idx > 0 else None
    return {
        "plan_skill_index": segment_idx + 1,
        "transition_index": segment_idx,
        "transition_kind": "initial_skill_activation" if segment_idx == 0 else "skill_boundary",
        "boundary_step": int(current_segment["start_step"]),
        "previous_skill": previous_skill,
        "next_skill": current_segment["skill"],
        "feasible": bool(result["feasible"]),
        "failure_reason": result["failure_reason"],
    }


def sanity_check_pddl(pddl_path: Path) -> str:
    pddl_domain_text = pddl_path.read_text(encoding="utf-8")
    sanity_problem = (
        "(define (problem sanity_check)\n"
        " (:domain tabletop)\n"
        " (:objects\n"
        "  robot_0 - robot\n"
        "  table_0 - scene_object\n"
        " )\n"
        " (:init\n"
        "  (free robot_0)\n"
        " )\n"
        ")\n"
    )
    setup_pddl_simulation(sanity_problem, pddl_domain_text)
    return pddl_domain_text


def validate_episode_symbolically(
    *,
    source: Path,
    episode: dict[str, Any],
    pddl_domain_text: str,
    llm_interface: Any,
    vlm_interface: Any,
    repo_id: str,
    dataset_root: str | None,
    transition_scene_roots: list[Path],
    plan_only: bool,
    max_retries: int = 5,
) -> dict[str, Any] | None:
    episode_index = int(episode["episode_index"])
    task_index = episode.get("task_index")
    instruction = extract_instruction(episode)

    try:
        plan = resolve_episode_plan(episode)
        plan_skills = parse_plan_skills(plan)
        canonical_segments = canonical_segments_from_episode(episode)
    except Exception as exc:
        return {
            "source": str(source),
            "episode_index": episode_index,
            "task_index": int(task_index) if isinstance(task_index, int) else task_index,
            "instruction": instruction,
            "failure_stage": "annotation_structure",
            "error": f"{type(exc).__name__}: {exc}",
            "repo_id": repo_id,
            "dataset_root": dataset_root,
        }

    canonical_skills = [segment["skill"] for segment in canonical_segments]
    if canonical_skills != plan_skills:
        return {
            "source": str(source),
            "episode_index": episode_index,
            "task_index": int(task_index) if isinstance(task_index, int) else task_index,
            "instruction": instruction,
            "plan": plan,
            "failure_stage": "annotation_structure",
            "error": (
                "Canonical segment skill sequence does not match the resolved plan. "
                f"plan={plan_skills}, segments={canonical_skills}"
            ),
            "canonical_segments": canonical_segments,
            "repo_id": repo_id,
            "dataset_root": dataset_root,
        }

    try:
        episode_num_steps = episode.get("num_steps")
        canonical_num_steps = int(canonical_segments[-1]["end_step"])
        if episode_num_steps is not None and int(episode_num_steps) != canonical_num_steps:
            raise ValueError(
                f"Episode metadata num_steps={episode_num_steps} does not match canonical segment length {canonical_num_steps}."
            )
    except Exception as exc:
        return {
            "source": str(source),
            "episode_index": episode_index,
            "task_index": int(task_index) if isinstance(task_index, int) else task_index,
            "instruction": instruction,
            "plan": plan,
            "failure_stage": "annotation_structure",
            "error": f"{type(exc).__name__}: {exc}",
            "canonical_segments": canonical_segments,
            "repo_id": repo_id,
            "dataset_root": dataset_root,
        }

    plan_result = None
    last_error: Exception | None = None
    last_stage = None
    for attempt in range(1, max_retries + 1):
        try:
            transition_scene_root, transition_files = resolve_transition_scene_files(
                source=source,
                episode_index=episode_index,
                canonical_segments=canonical_segments,
                transition_scene_roots=transition_scene_roots,
            )
            initial_frame = load_rgb_image(transition_files[0])
            scene_graph_hint = [f"Task Instruction: {instruction}"]
            scene_graph = TaskSceneGraph(pddl_domain_text, vlm_interface)
            scene_graph.read_image(initial_frame, hint=scene_graph_hint, ground=False)
            if scene_graph.simulator is None:
                raise RuntimeError("Scene graph initialization produced no simulator.")
            verifier = VLAVerifier(scene_graph, llm_interface, vlm_interface)
            scene_graph_initial_copy = copy.deepcopy(scene_graph)
        except Exception as exc:
            last_error = exc
            last_stage = "scene_graph_initialization"
            if attempt < max_retries:
                print(
                    f"plan verification retry {attempt}/{max_retries - 1} for episode {episode_index} after error: {exc}",
                    flush=True,
                )
                continue
            break

        try:
            plan_result = verifier.verify_skill_plan(plan)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            last_stage = "plan_validation_runtime"
            if attempt < max_retries:
                print(
                    f"plan verification retry {attempt}/{max_retries - 1} for episode {episode_index} after error: {exc}",
                    flush=True,
                )
                continue
            break

    if last_error is not None:
        return {
            "source": str(source),
            "episode_index": episode_index,
            "task_index": int(task_index) if isinstance(task_index, int) else task_index,
            "instruction": instruction,
            "plan": plan,
            "failure_stage": last_stage,
            "error": f"{type(last_error).__name__}: {last_error}",
            "canonical_segments": canonical_segments,
            "repo_id": repo_id,
            "dataset_root": dataset_root,
        }

    assert plan_result is not None

    serialized_plan_result = serialize_plan_result(plan_result)
    if not plan_result["feasible"]:
        return {
            "source": str(source),
            "episode_index": episode_index,
            "task_index": int(task_index) if isinstance(task_index, int) else task_index,
            "instruction": instruction,
            "plan": plan,
            "failure_stage": "plan_validation",
            "plan_validation": serialized_plan_result,
            "canonical_segments": canonical_segments,
            "transition_boundary_steps": transition_boundary_steps_from_segments(canonical_segments),
            "transition_scene_root": str(transition_scene_root),
            "repo_id": repo_id,
            "dataset_root": dataset_root,
        }

    if plan_only:
        return None

    for attempt in range(1, max_retries + 1):
        try:
            scene_graph = copy.deepcopy(scene_graph_initial_copy)
            verifier = VLAVerifier(scene_graph, llm_interface, vlm_interface)

            verifier.set_skill_plan(plan)
            checked_transitions: list[dict[str, Any]] = []

            first_transition_result = verifier.verify_skill_transition(canonical_segments[0]["skill"])
            first_check = build_transition_check(
                canonical_segments=canonical_segments,
                segment_idx=0,
                result=first_transition_result,
            )
            checked_transitions.append(first_check)
            if not first_transition_result["feasible"]:
                return {
                    "source": str(source),
                    "episode_index": episode_index,
                    "task_index": int(task_index) if isinstance(task_index, int) else task_index,
                    "instruction": instruction,
                    "plan": plan,
                    "failure_stage": "transition_validation",
                    "plan_validation": serialized_plan_result,
                    "transition_validation": {
                        "failed_transition_index": first_check["transition_index"],
                        "failed_plan_skill_index": first_check["plan_skill_index"],
                        "failed_boundary_step": first_check["boundary_step"],
                        "previous_skill": first_check["previous_skill"],
                        "next_skill": first_check["next_skill"],
                        "failure_reason": first_check["failure_reason"],
                        "checked_transitions": checked_transitions,
                    },
                    "canonical_segments": canonical_segments,
                    "transition_scene_root": str(transition_scene_root),
                    "repo_id": repo_id,
                    "dataset_root": dataset_root,
                }

            for segment_idx in range(1, len(canonical_segments)):
                next_segment = canonical_segments[segment_idx]
                image_rgb = load_rgb_image(transition_files[segment_idx])
                transition_result = verifier.verify_skill_transition(
                    next_skill=next_segment["skill"],
                    image_rgb=image_rgb,
                )
                transition_check = build_transition_check(
                    canonical_segments=canonical_segments,
                    segment_idx=segment_idx,
                    result=transition_result,
                )
                checked_transitions.append(transition_check)

                if not transition_result["feasible"]:
                    return {
                        "source": str(source),
                        "episode_index": episode_index,
                        "task_index": int(task_index) if isinstance(task_index, int) else task_index,
                        "instruction": instruction,
                        "plan": plan,
                        "failure_stage": "transition_validation",
                        "plan_validation": serialized_plan_result,
                        "transition_validation": {
                            "failed_transition_index": transition_check["transition_index"],
                            "failed_plan_skill_index": transition_check["plan_skill_index"],
                            "failed_boundary_step": transition_check["boundary_step"],
                            "previous_skill": transition_check["previous_skill"],
                            "next_skill": transition_check["next_skill"],
                            "failure_reason": transition_check["failure_reason"],
                            "checked_transitions": checked_transitions,
                        },
                        "canonical_segments": canonical_segments,
                        "transition_scene_root": str(transition_scene_root),
                        "repo_id": repo_id,
                        "dataset_root": dataset_root,
                    }

        except Exception as exc:
            if attempt < max_retries:
                print(
                    f"transition validation retry {attempt}/{max_retries - 1} for episode {episode_index} after error: {exc}",
                    flush=True,
                )
                continue
            break

    return None


def main() -> int:
    args = parse_args()
    files = expand_inputs(args.inputs)
    if not files:
        raise FileNotFoundError("No annotation files found in the provided inputs.")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(args.inputs).expanduser().resolve()
    )
    pddl_path = args.pddl_path.expanduser().resolve()
    pddl_domain_text = sanity_check_pddl(pddl_path)
    llm_interface, vlm_interface = get_interfaces(args.backend)
    explicit_transition_scene_roots = (
        [normalize_transition_scene_root(path) for path in args.transition_scene_roots]
        if args.transition_scene_roots
        else []
    )

    loaded_inputs: list[tuple[Path, list[dict[str, Any]], dict[str, Any], list[Path]]] = []
    total_expected_episodes = 0
    for path in files:
        episodes, file_metadata = load_input_file(path)
        episodes = filter_episodes_by_range(
            episodes,
            start=args.start,
            end=args.end,
        )
        auto_roots = discover_transition_scene_roots(path)
        loaded_inputs.append((path, episodes, file_metadata, unique_paths(explicit_transition_scene_roots + auto_roots)))
        total_expected_episodes += len(episodes)

    failures: list[dict[str, Any]] = []
    total_files = 0
    total_episodes = 0

    for path, episodes, file_metadata, transition_scene_roots in loaded_inputs:
        total_files += 1
        if not args.quiet:
            print(f"checking {path} ({len(episodes)} episode(s))", flush=True)

        repo_id = args.repo_id or str(file_metadata.get("source_repo_id") or DEFAULT_REPO_ID)
        dataset_root = file_metadata.get("dataset_root")

        for episode in episodes:
            total_episodes += 1
            episode_index = episode.get("episode_index")
            print(
                f"[{total_episodes}/{total_expected_episodes}] validating {path} episode={episode_index}",
                flush=True,
            )
            failure = validate_episode_symbolically(
                source=path,
                episode=episode,
                pddl_domain_text=pddl_domain_text,
                llm_interface=llm_interface,
                vlm_interface=vlm_interface,
                repo_id=repo_id,
                dataset_root=dataset_root,
                transition_scene_roots=transition_scene_roots,
                plan_only=args.plan_only,
            )
            if failure is None:
                continue

            failures.append(failure)
            print(
                f"[fail] source={failure['source']} episode={failure['episode_index']} "
                f"stage={failure['failure_stage']}",
                flush=True,
            )

            report = {
                "created_at": utc_now(),
                "backend": args.backend,
                "pddl_path": str(pddl_path),
                "plan_only": args.plan_only,
                "total_files": total_files,
                "checked_episodes": total_episodes,
                "failed_episodes": len(failures),
                "failures": failures,
            }
            save_json_atomic(output_path, report)
            if args.stop_on_first_error:
                print(f"saved_partial_report={output_path}", flush=True)
                return 1

    report = {
        "created_at": utc_now(),
        "backend": args.backend,
        "pddl_path": str(pddl_path),
        "plan_only": args.plan_only,
        "total_files": total_files,
        "checked_episodes": total_episodes,
        "failed_episodes": len(failures),
        "failures": failures,
    }
    save_json_atomic(output_path, report)

    print(f"checked_files={total_files}", flush=True)
    print(f"checked_episodes={total_episodes}", flush=True)
    print(f"failed_episodes={len(failures)}", flush=True)
    print(f"output={output_path}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
