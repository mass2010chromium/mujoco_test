"""
Note: This policy class requires as input additional fields in its observation:

`ee_position`: [pixel_x, pixel_y, depth]
`agentview_depth`: depth image (for 3d trace only)
"""
import dataclasses
import functools
import textwrap
from typing import Any

import cv2
import numpy as np

from openpi.models import trace_utils as _trace_utils
from openpi.policies import policy_config as _policy_config

from inference_common import quat2axisangle, register_model
from skill_processing import (
    get_skill_name,
    parse_plan
)
from pi05_trace_extra import (
    PLAN_SCHEMA,
    build_plan_prompt,
    _normalize_plan_response,
    SEMANTIC_POINT_SCHEMA,
    build_semantic_point_prompt,
    _normalize_semantic_point_response
)

OPENROUTER_MODEL = "google/gemini-3.1-pro-preview"
REPLAN_EVERY_CHUNKS = 5
ACTIONS_PER_CHUNK = 5
COMPLETION_THRESHOLD = 0.9
CONSECUTIVE_REQUIRED = 2

def pixel_to_normalized_xy(x: int, y: int, image_width: int, image_height: int) -> tuple[float, float]:
    return (float(x) / max(int(image_width) - 1, 1), float(y) / max(int(image_height) - 1, 1))

def run_module():
    import os
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set.")
        print("VLM requiring trace VLA will not be enabled.")
        print("To set, do:")
        print('  export OPENROUTER_API_KEY="sk-or-v1-..."')
        return

    from llm_apis import transformers_api
    from llm_apis.llm_tool import LLMTool, OpenRouterTool, json_output
    from llm_apis.response_parsing import extract_in_backticks, extract_json_from_response

    vlm_interface = LLMTool.make_factory(OpenRouterTool, OPENROUTER_MODEL, api_key, temperature=0.0)
    def vlm_interface_with_schema(schema=None):
        return lambda f: vlm_interface(f, system_prompt=None, schema=schema)


    ########################
    # VLM Interfaces
    ########################
    @vlm_interface_with_schema(PLAN_SCHEMA)
    @json_output
    def get_initial_plan(task_instruction: str, initial_scene_image: np.ndarray) -> dict[str, Any]:
        return [ transformers_api.make_message(
            texts=build_plan_prompt(task_instruction),
            images=[initial_scene_image]
        ) ]

    @vlm_interface_with_schema(SEMANTIC_POINT_SCHEMA)
    def get_semantic_target(llm_response, task_instruction: str,
                             plan_str: str, skill_text: str,
                             current_scene_image: np.ndarray) -> tuple[int, int]:
        """
        Return: [normalized_x, normalized_y, depth]
        """
        H, W, _ = current_scene_image.shape
        yield [transformers_api.make_message(
            texts=build_semantic_point_prompt(
                task_instruction=task_instruction, plan_str=plan_str, skill_text=skill_text,
                image_width=W, image_height=H,
            ),
            images=current_scene_image
        )]
        try:
            raw_response = llm_response['content']
            result = extract_json_from_response(raw_response)
            sem_pixel = _normalize_semantic_point_response(result, image_width=W, image_height=H)['point_pixel']
        except Exception as exc:
            print(f"  [skill={skill_text}] semantic-target query failed: {exc}", flush=True)
            print(f"  -> falling back to image center.", flush=True)
            sem_pixel = (W // 2, H // 2)
        yield sem_pixel


    ########################
    # Inference wrappers
    ########################
    # TODO: Do not double define!
    TRACE_OVERLAY_COLOR = (0, 255, 255)        # cyan, matches data_config.overlay_color
    TRACE_OVERLAY_THICKNESS = 2                  # matches data_config.overlay_thickness
    TRACE_OVERLAY_ENDPOINT_RADIUS = 2.5          # matches data_config.overlay_endpoint_radius
    EE_DOT_COLOR = (0, 255, 0)                   # green
    SEM_DOT_COLOR = (255, 0, 0)                  # red
    @dataclasses.dataclass
    class SkillContext:
        skill_text: str                     # full parameterized form, e.g. "PICKUP_FROM(black bowl, table)"
        skill_name: str                     # bare verb, e.g. "PICKUP_FROM"
        skill_id: int                       # MoE expert id, 0-4
        semantic_target_pixel: tuple[int, int]   # (x, y) pixel
        semantic_target: tuple[float, float] | tuple[float, float, float]    # (x, y, [depth]), normalized
        cached_trace: np.ndarray | None = None  # (N, k) in [0, 1], k=2 or 3

    def _make_base_obs(obs: dict) -> dict[str, Any]:
        """Common obs fields shared across plan-mode and execution-mode invocations."""
        return {
            "observation/image": obs['agentview_image'][::-1, ::-1, :],
            "observation/wrist_image": obs['robot0_eye_in_hand_image'][::-1, ::-1, :],
            "observation/state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ).astype(np.float32)
        }
    def make_planning_obs(libero_obs: dict, skill_ctx: SkillContext, task_description: str,
                          ee_info: tuple[float, float] | tuple[float, float, float]) -> dict[str, Any]:
        """Build the obs dict the trace expert (planning mode) needs."""
        base = _make_base_obs(libero_obs)
        base.update({
            "atomic_token": float(skill_ctx.skill_id),
            "semantic_target_xy": np.asarray(skill_ctx.semantic_target, dtype=np.float32),
            "current_ee_xy": np.asarray(ee_info, dtype=np.float32),
            # Provide both forms so TraceTokenizePrompt picks the parameterized text first.
            "skill_text": skill_ctx.skill_text,
            "skill_name": skill_ctx.skill_name,
            "prompt": task_description,
            # Marker fields to keep TraceObservation typecheck happy when the policy
            # tries to read these (with `.get(...)` they default to None on missing).
            "has_trace": True,
            "has_overlay": False,
            "progress": 0.0,
        })
        return base
    def make_execution_obs(libero_obs: dict, skill_ctx: SkillContext, task_description: str,
                            ee_info: tuple[float, float] | tuple[float, float, float],
                            overlay_image: np.ndarray,
                            trace: np.ndarray | None = None) -> dict[str, Any]:
        """Build the obs dict the action expert + completion head need."""
        base = make_planning_obs(libero_obs, skill_ctx, task_description, ee_info)
        base["observation/overlay_image"] = overlay_image
        base["has_overlay"] = True
        base["future_trace_xy"] = trace
        return base


    def render_video_frame(img: np.ndarray, *, trace: np.ndarray,
                            sem_pixel: tuple[int, int], ee_pixel: tuple[int, int],
                            skill_text: str, progress: float,
                            consecutive_high: int, completion_threshold: float) -> np.ndarray:
        """Compose the per-frame visualization (overlay + dots + text) for the video.

        The "what model sees" overlay (cyan polyline from `render_overlay_image`) is the
        background. We then add green/red dots for the EE/sem keypoints and a top-left
        text block annotating skill + progress, so a single video frame contains
        everything needed to debug the rollout.
        """
        # Creates internal copy
        img = _trace_utils.draw_polyline_overlay(img, trace,
            color=TRACE_OVERLAY_COLOR,
            line_thickness=TRACE_OVERLAY_THICKNESS,
            endpoint_radius=TRACE_OVERLAY_ENDPOINT_RADIUS,
        )
        _trace_utils._filled_disk(img, float(ee_pixel[0]), float(ee_pixel[1]), 4.0, EE_DOT_COLOR)
        _trace_utils._filled_disk(img, float(sem_pixel[0]), float(sem_pixel[1]), 4.0, SEM_DOT_COLOR)

        # Text block.
        progress_str = f"{progress:.3f}"
        lines = [
            f"Skill: {skill_text}",
            f"Progress: {progress_str}  (>= {completion_threshold:.2f} streak: {consecutive_high})",
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.30
        thickness = 1
        margin = 6
        line_metrics = [cv2.getTextSize(line, font, font_scale, thickness) for line in lines]
        text_w = max(m[0][0] for m in line_metrics)
        text_h = max(m[0][1] for m in line_metrics)
        baseline = max(m[1] for m in line_metrics)
        line_gap = 3
        block_h = len(lines) * text_h + (len(lines) - 1) * line_gap

        x = margin
        y = margin + text_h
        cv2.rectangle(
            img,
            (max(0, x - 4), max(0, y - text_h - 4)),
            (min(img.shape[1] - 1, x + text_w + 4),
             min(img.shape[0] - 1, y - text_h + block_h + baseline + 4)),
            (0, 0, 0), -1,
        )
        for idx, line in enumerate(lines):
            line_y = y + idx * (text_h + line_gap)
            cv2.putText(img, line, (x, line_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return img


    def trace_init(policy, obs, task):
        policy._consecutive_high = 0
        policy._replan_threshold = REPLAN_EVERY_CHUNKS
        policy._chunks_since_replan = REPLAN_EVERY_CHUNKS
        policy._chunks_since_completion_check = 0
        policy._completion_check_interval = 1
        policy._last_progress = None
        policy._completion_threshold = COMPLETION_THRESHOLD
        policy._consecutive_required = CONSECUTIVE_REQUIRED
        # NOTE: this policy requires as input additional fields in the observation:
        plan_raw = get_initial_plan(task, np.array(obs['agentview_image'][::-1, ::-1, :], copy=True))['plan'].strip()
        skills = parse_plan(plan_raw)
        policy._task = task
        policy._plan_raw = plan_raw
        policy._skill_list = skills
        policy._skill_idx = 0
        policy._save_video_frames = []
        trace_skill_init(policy, obs)

    def trace_skill_init(policy, obs):
        scene_image = np.array(obs['agentview_image'][::-1, ::-1, :], copy=True)
        H, W = scene_image.shape[:2]
        ee_world = np.asarray(obs["robot0_eef_pos"], dtype=float)
        # TODO:
        ee_x, ee_y, ee_depth = obs['ee_position']
        ee_info = list(pixel_to_normalized_xy(ee_x, ee_y, W, H))
        if policy._use_3d:
            ee_info += [ee_depth]

        skill_text = policy._skill_list[policy._skill_idx]
        sem_pixel = get_semantic_target(policy._task, policy._plan_raw, skill_text, scene_image)

        sem_target = list(pixel_to_normalized_xy(sem_pixel[0], sem_pixel[1], W, H))
        if policy._use_3d:
            sem_target += [obs['agentview_depth'][::-1, ::-1, :][sem_pixel[1], sem_pixel[0]][0]]

        policy._skill_ctx = SkillContext(
            skill_text=skill_text,
            skill_name=get_skill_name(skill_text),
            skill_id=_trace_utils.skill_to_expert_id(skill_text),
            semantic_target_pixel=sem_pixel,
            semantic_target=sem_target
        )

    def trace_infer(policy, obs):
        # --- Project current EE to image pixel ---
        scene_image = np.array(obs['agentview_image'][::-1, ::-1, :], copy=True)
        H, W = scene_image.shape[:2]
        ee_world = np.asarray(obs["robot0_eef_pos"], dtype=float)
        # TODO:
        ee_x, ee_y, ee_depth = obs['ee_position']
        ee_info = list(pixel_to_normalized_xy(ee_x, ee_y, W, H))
        if policy._use_3d:
            ee_info += [ee_depth]

        # Trigger replan if needed. This should fire on the first invocation
        if policy._chunks_since_replan >= policy._replan_threshold:
            planning_obs = make_planning_obs(obs, policy._skill_ctx, policy._task, ee_info)
            trace = np.array(policy.sample_trace(planning_obs), dtype=np.float32, copy=True)     # (N, 2) in [0, 1]
            # The model inference doesn't clip it right now.
            trace[:, :2] = np.clip(trace[:, :2], 0.0, 1.0)
            policy._skill_ctx.cached_trace = trace
            policy._chunks_since_replan = 0
            print(f"    chunk-replan: trace start={trace[0].tolist()}, end={trace[-1].tolist()}", flush=True)

        # --- Build overlay image for execution-mode forward ---
        policy._overlay_image = _trace_utils.draw_polyline_overlay(
            scene_image,
            policy._skill_ctx.cached_trace,
            color=TRACE_OVERLAY_COLOR,
            line_thickness=TRACE_OVERLAY_THICKNESS,
            endpoint_radius=TRACE_OVERLAY_ENDPOINT_RADIUS,
        )

        if policy._chunks_since_completion_check == policy._completion_check_interval:
            policy._chunks_since_completion_check = 0
            # No trace needed -- only completion prediction.
            exec_obs = make_execution_obs(obs, policy._skill_ctx, policy._task, ee_info, policy._overlay_image)
            progress = float(np.asarray(policy.predict_completion(exec_obs)))
            policy._last_progress = progress
            policy.consecutive_high = policy._consecutive_high + 1 if progress >= policy._completion_threshold else 0
            if policy._consecutive_high >= policy._consecutive_required:
                print(f"    completion advance: progress={progress_only:.3f} streak={consecutive_high} "
                      f"-> next skill", flush=True)

                # Render one final frame so the abort moment is visible in the video.
                policy._save_video_frames.append(render_video_frame(
                    scene_image, trace=policy._skill_ctx.cached_trace,
                    sem_pixel=policy._skill_ctx.semantic_target_pixel, ee_pixel=(ee_x, ee_y),
                    skill_text=policy._skill_ctx.skill_text, progress=policy._last_progress,
                    consecutive_high=policy._consecutive_high, completion_threshold=policy._completion_threshold,
                ))

                # still have available skill, proceed to next skill
                if policy._skill_idx < len(policy._skill_list) - 1:
                    policy._skill_idx += 1
                    trace_skill_init(policy, obs)
                    policy._consecutive_high = 0
                    policy._chunks_since_replan = policy._replan_threshold
                    policy._chunks_since_completion_check = 0
                    policy._last_progress = None    # Maybe not necessary?
                    # jump back to the top, to replan.
                    return trace_infer(policy, obs)

        policy._chunks_since_completion_check += 1
        policy._chunks_since_replan += 1

        # --- (3b) Action generation (also returns a fresh progress estimate) ---
        exec_obs = make_execution_obs(obs, policy._skill_ctx, policy._task, ee_info, policy._overlay_image, policy._skill_ctx.cached_trace)
        infer_result = policy.infer(exec_obs)
        policy._last_progress = float(np.asarray(infer_result["progress"]))
        return infer_result


    def record_video_frame(policy, obs):
        ee_x, ee_y, ee_depth = obs['ee_position']
        frame = render_video_frame(
            obs['agentview_image'][::-1, ::-1, :],
            trace=policy._skill_ctx.cached_trace,
            sem_pixel=policy._skill_ctx.semantic_target_pixel,
            ee_pixel=(ee_x, ee_y),
            skill_text=policy._skill_ctx.skill_text,
            progress=policy._last_progress,
            consecutive_high=policy._consecutive_high,
            completion_threshold=policy._completion_threshold,
        )
        policy._save_video_frames.append(frame)


    def create_trace_vla_policy(model_name, config: "TrainConfig", checkpoint_dir, norm_stats):
        policy = _policy_config.create_trained_trace_vla_policy(
            config,
            checkpoint_dir,
            norm_stats=norm_stats,
        )
        policy._use_3d = config.model.trace_dim == 3
        policy.initialize = functools.partial(trace_init, policy)
        policy.run_vla = functools.partial(trace_infer, policy)
        policy.record_video_frame = functools.partial(record_video_frame, policy)
        return policy


    register_model(create_trace_vla_policy, "trace_vla_3d_lora", 25000)

run_module()
