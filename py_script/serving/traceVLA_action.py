import functools

import numpy as np

from openpi.models import trace_utils as _trace_utils
from openpi.policies import policy_config as _policy_config

from inference_common import register_model
from skill_processing import get_skill_name


def _get(obs, *keys):
    for key in keys:
        if key in obs:
            return obs[key]
    raise KeyError(f"Missing required observation key; expected one of {keys}")


def _xy_from_obs(obs, xy_key, pixel_key):
    if xy_key in obs:
        return np.asarray(obs[xy_key], dtype=np.float32)

    pixel = np.asarray(_get(obs, pixel_key), dtype=np.float32)
    image = np.asarray(_get(obs, "image", "observation/image"))
    h, w = image.shape[:2]
    return np.asarray(
        [pixel[0] / max(w - 1, 1), pixel[1] / max(h - 1, 1)],
        dtype=np.float32,
    )


def trace_prompt(obs, task, *, require_overlay):
    skill_text = str(_get(obs, "skill_text"))
    prompt = {
        "observation/image": _get(obs, "image", "observation/image"),
        "observation/wrist_image": _get(obs, "wrist_image", "observation/wrist_image"),
        "observation/state": np.asarray(_get(obs, "state", "observation/state"), dtype=np.float32),
        "atomic_token": float(obs.get("atomic_token", _trace_utils.skill_to_expert_id(skill_text))),
        "semantic_target_xy": _xy_from_obs(obs, "semantic_target_xy", "semantic_target_pixel"),
        "current_ee_xy": _xy_from_obs(obs, "current_ee_xy", "current_ee_pixel"),
        "skill_text": skill_text,
        "skill_name": str(obs.get("skill_name", get_skill_name(skill_text))),
        "plan_text": str(obs.get("plan_text", "")),
        "skill_step_num": int(obs.get("skill_step_num", 0)),
        "prompt": str(obs.get("prompt", task)),
        "has_trace": True,
        "has_overlay": False,
        "progress": 0.0,
    }
    if require_overlay:
        prompt["observation/overlay_image"] = _get(obs, "overlay_image", "observation/overlay_image")
        prompt["has_overlay"] = True
    return prompt


def trace_init(policy, obs, task):
    del obs
    policy.task = task


def trace_infer(policy, obs):
    mode = str(obs.get("mode", obs.get("query_mode", "action"))).lower()

    if mode in {"trace", "plan", "planning"}:
        prompt = trace_prompt(obs, policy.task, require_overlay=False)
        trace = np.asarray(policy.sample_trace(prompt), dtype=np.float32)
        return {"trace": np.clip(trace, 0.0, 1.0)}

    if mode in {"completion", "progress"}:
        prompt = trace_prompt(obs, policy.task, require_overlay=True)
        return {"progress": np.asarray(policy.predict_completion(prompt), dtype=np.float32)}

    if mode not in {"action", "act", "execution", "execute"}:
        raise ValueError(f"Unsupported TraceVLA query mode: {mode!r}")

    prompt = trace_prompt(obs, policy.task, require_overlay=True)
    return policy.infer(prompt)


def trace_reset(policy):
    policy.task = ""


def create_trace_vla_real(model_name, config, checkpoint_dir, norm_stats):
    del model_name
    policy = _policy_config.create_trained_trace_vla_policy(
        config,
        checkpoint_dir,
        norm_stats=norm_stats,
    )
    policy.initialize = functools.partial(trace_init, policy)
    policy.run_vla = functools.partial(trace_infer, policy)
    policy.reset = functools.partial(trace_reset, policy)
    return policy


register_model(
    create_trace_vla_real,
    "trace_vla_moe_table_tasks",
    30000,
    _checkpoint_dir="/work/hdd/bgtb/zhong2/checkpoints/trace_vla_moe_table_tasks/trace_vla_moe_table_tasks/30000",
)
