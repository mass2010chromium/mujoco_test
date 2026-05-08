# TraceVLA — Implementation Reference

A complete walk-through of the TraceVLA training pipeline and model code, with file
and line-level references for every component. Reading this end-to-end should leave
you knowing **what is where** and **why** each piece exists.

> **Update log (2026-05-08).** Reflects fixes for issues 1, 2, 3 (+ skill-as-prompt),
> 5, 6, and 7 from `TraceVLA_implementation_confirmed_issues.md`:
>
> - **Image augmentation now co-transforms keypoints + overlay** via augmax's native
>   `KEYPOINTS` input type (§9.3). Random crop / rotate now act consistently across
>   the base image, the overlay base image, the semantic-target keypoint, the current
>   EE keypoint, and every waypoint of the supervised trace. Wrist images get a
>   separate color-only chain.
> - **Trace target now starts at `t_now`**, not the receding-horizon anchor (§4.4).
>   The overlay still uses the anchor-aged trace, so the action head still trains on
>   stale plans, but the trace generator's flow-matching target is aligned with the
>   inpainting clamp by construction.
> - **The VLM now sees the parameterized skill text as its prompt** (§5.3). The
>   dataset emits `skill_text="PICKUP_FROM(white mug, table)"` alongside
>   `skill_name="PICKUP_FROM"`, both forwarded by `LiberoTraceInputs`. The original
>   LIBERO task instruction is no longer fed to the VLA — by design, the off-the-shelf
>   high-level VLM produces the skill, and the VLA only needs the skill.
> - **Norm-stats are loaded the standard way** (§8.4). The training script no longer
>   passes `skip_norm_stats=True`; the user must run
>   `compute_norm_stats.py --config-name trace_vla` (and `trace_vla_lora`) once.
>   `create_torch_dataset` has a `LiberoTraceDataConfig` branch so that script works.
> - **Weight remap covers `q_einsum` and `kv_einsum`** (§8.1). The action expert
>   `gemma_300m` uses split Q + KV (because `num_heads != num_kv_heads`), so the
>   stream-1 → stream-2 copy now includes those names — the trace stream's attention
>   is now actually warm-started from `pi05_base`, not silently left at random init.
> - **Public completion-inference path is now in place** (§15). The model exposes
>   `sample_actions_and_completion(rng, obs)` (combined, shared prefix prefill — the
>   recommended deployment endpoint) and `predict_completion(rng, obs)` (standalone,
>   no action denoising). `TraceVLAPolicy` in `policies/policy.py` wraps these and
>   returns `progress` alongside `actions` from `infer()`. A factory
>   `create_trained_trace_vla_policy` is added to `policies/policy_config.py`.
>
> Issue 4 (state not consumed) was deliberately *not* fixed: it matches AtomicVLA's
> working `pi05` Libero configuration. Issue 8 (progress target ceiling < 1) remains
> open.

---

## 0. TL;DR

TraceVLA is a trace-augmented Vision-Language-Action model. It pairs a pi05/AtomicVLA-style
**action expert** with a new **trace expert** (a 5-skill hard-routed Mixture-of-Experts) that
generates pixel-space end-effector traces. The action expert sees the trace as a visual
overlay on the input image. A small per-skill **completion head** also predicts skill
progress in `[0, 1]`. The whole thing is one model, trained end-to-end with three losses
(action, trace, completion).

Two run configs are exposed:

```bash
# Full finetune
python pace/openpi/scripts/train_trace_vla.py trace_vla --exp-name trace_vla

# LoRA finetune (LoRA on PaliGemma 2B + gemma_300m action expert; trace MoE + completion head are full FT)
python pace/openpi/scripts/train_trace_vla.py trace_vla_lora --exp-name trace_vla_lora
```

Both load `yilin-wu/libero-100`. Annotations live at:

- `pace/openpi/data/libero-100/skill_annotations.json`
- `pace/openpi/data/libero-100/skill_target_traces.json`

---

## 1. File map

All new files. Existing pipeline files (`gemmoe.py`, `pi0_atomic.py`, `model.py`,
`data_loader.py`, etc.) are imported but never modified, except for additive entries
in `training/config.py`.

| File | LOC | Role |
|---|---:|---|
| [pace/openpi/src/openpi/models/gemmoe_trace.py](pace/openpi/src/openpi/models/gemmoe_trace.py) | 416 | 3-stream Gemma trunk, `HardMoeBlock`, `TraceBlock`, `TraceModule`. |
| [pace/openpi/src/openpi/models/trace_observation.py](pace/openpi/src/openpi/models/trace_observation.py) | 147 | `TraceObservation` dataclass + `preprocess_trace_observation`. |
| [pace/openpi/src/openpi/models/trace_utils.py](pace/openpi/src/openpi/models/trace_utils.py) | 276 | Skill→expert mapping, trace resampling (arc-length / time-uniform), polyline overlay rendering. |
| [pace/openpi/src/openpi/models/pi0_trace_vla_config.py](pace/openpi/src/openpi/models/pi0_trace_vla_config.py) | 168 | `Pi0TraceVLAConfig` model config + 3-stream LoRA freeze filter. |
| [pace/openpi/src/openpi/models/pi0_trace_vla.py](pace/openpi/src/openpi/models/pi0_trace_vla.py) | ~830 | The model. Two forward passes (planning + execution), three losses, `sample_actions`, `sample_actions_and_completion`, `predict_completion`, `sample_trace`. |
| [pace/openpi/src/openpi/policies/libero_trace_dataset.py](pace/openpi/src/openpi/policies/libero_trace_dataset.py) | ~395 | `LiberoTraceDataset` — annotation join, decoupled trace target / overlay (Issue 2 fix), anchor-age augmentation, scene dropout, action zeroing. |
| [pace/openpi/src/openpi/policies/libero_trace_policy.py](pace/openpi/src/openpi/policies/libero_trace_policy.py) | ~180 | Input/output transforms: `LiberoTraceInputs` (forwards `skill_text`/`skill_name`), `TraceResizeImages`, `TraceTokenizePrompt` (uses skill text as prompt), `LiberoTraceOutputs`. |
| [pace/openpi/scripts/train_trace_vla.py](pace/openpi/scripts/train_trace_vla.py) | ~382 | Training script: weight-load remap (now incl. `q_einsum`/`kv_einsum`), init train state, JIT'd train step, data loader (no longer skips norm stats). |

**Inference-side additions** (added with the Issue 7 fix):

| File | Role |
|---|---|
| [pace/openpi/src/openpi/policies/policy.py](pace/openpi/src/openpi/policies/policy.py) | Adds `TraceVLAPolicy`. `infer()` returns `{actions, progress, ...}` via the combined-forward model endpoint; also exposes `sample_trace()` and `predict_completion()`. |
| [pace/openpi/src/openpi/policies/policy_config.py](pace/openpi/src/openpi/policies/policy_config.py) | Adds `create_trained_trace_vla_policy(...)` factory mirroring `create_trained_policy` but instantiating `TraceVLAPolicy`. |

**Additions** to existing `pace/openpi/src/openpi/training/config.py`:
- `LiberoTraceDataConfig` (line 119): per-dataset hyperparameters (annotation paths, trace shape, anchor-age, scene dropout, overlay style).
- `LeRobotTraceVLADataConfig` (line 591): factory that wires the trace data transforms.
- `trace_vla` TrainConfig (line 1813): full FT.
- `trace_vla_lora` TrainConfig (line 1857): LoRA FT.

---

## 2. Configuration layer

### 2.1 `Pi0TraceVLAConfig`

[pi0_trace_vla_config.py:23-100](pace/openpi/src/openpi/models/pi0_trace_vla_config.py#L23-L100)

The model config. Notable fields:

| Field | Default (in `trace_vla`) | Meaning |
|---|---|---|
| `paligemma_variant` | `"gemma_2b"` (`"gemma_2b_lora"` for LoRA cfg) | Stream 0 backbone |
| `action_expert_variant` | `"gemma_300m"` (`"gemma_300m_lora"` for LoRA) | Stream 1 |
| `trace_expert_variant` | `"trace_moe_gemma_300m"` | Stream 2 (always full FT) |
| `action_horizon` | 10 | per the spec |
| `pi05` | True | adaRMS time pathway (no continuous state token) |
| `discrete_state_input` | False | matches AtomicVLA on Libero |
| `trace_horizon` | 20 | N waypoints |
| `num_trace_experts` | 5 | matches `embed_sigma` mapping |
| `fourier_num_freqs` | 8 | for the semantic-target Fourier encoding (→ 32-dim feature) |
| `completion_shared_dim` | 256 | shared compressor before per-skill heads |
| `completion_per_skill_hidden` | 64 | per-skill MLP hidden |
| `trace_loss_coeff`, `action_loss_coeff`, `completion_loss_coeff` | 1.0, 1.0, 0.1 | loss weights |

`inputs_spec` (lines 109-148) declares the full TraceObservation schema so JAX can `eval_shape` the model during training-state init.

### 2.2 LoRA freeze filter

[pi0_trace_vla_config.py:150-167](pace/openpi/src/openpi/models/pi0_trace_vla_config.py#L150-L167)

Built from path-regex predicates over the param tree (using `nnx_utils.PathRegex`):

- `all_llm = ".*llm.*"` — anything in the trunk.
- `action_expert_subtree = ".*llm.*(_1).*"` — Gemma stream-1 weights (`name_1` suffix).
- `trace_expert_subtree = ".*llm.*(_2).*"` — Gemma stream-2 weights (`name_2` suffix).

**For LoRA + LoRA combination** (`trace_vla_lora`), the freeze filter resolves to:

```
All(all_llm, Not(trace_expert_subtree), Not(".*lora.*"))
```

i.e. **freeze**: PaliGemma + action expert (excluding LoRA params); **trainable**: trace
expert (`_2` subtree), all LoRA adapters, completion head, all I/O projections, time MLPs.
Verified by manual inspection of expected paths during sandbox tests.

### 2.3 Data config + factory

[training/config.py:119-153](pace/openpi/src/openpi/training/config.py#L119-L153) — `LiberoTraceDataConfig` carries:
- Annotation paths (skill/segment + trace).
- `trace_horizon=20`, `trace_resample_method="arc_length"` (default; `"time_uniform"` selectable).
- `h_train_max=15` — max anchor age, see §4.3.
- `scene_dropout_rate=0.15` — see §4.5.
- `overlay_color`, `overlay_thickness`, `overlay_endpoint_radius`.

[training/config.py:591-637](pace/openpi/src/openpi/training/config.py#L591-L637) — `LeRobotTraceVLADataConfig`:
factory. Wires the input/output transforms. Specifically:

```python
data_transforms = Group(
    inputs=[libero_trace_policy.LiberoTraceInputs(model_type=model_config.model_type)],
    outputs=[libero_trace_policy.LiberoTraceOutputs()],
)

model_transforms = Group(
    inputs=[
        libero_trace_policy.TraceResizeImages(224, 224),
        libero_trace_policy.TraceTokenizePrompt(
            tokenizer.PaligemmaTokenizer(model_config.max_token_len),
            discrete_state_input=False,
        ),
        transforms.PadStatesAndActions(model_config.action_dim),
    ],
)
```

### 2.4 TrainConfig entries

[training/config.py:1813-1880](pace/openpi/src/openpi/training/config.py#L1813-L1880)

Both configs:
- `repo_id="yilin-wu/libero-100"`,
- `repo_path=REPO_ROOT/"data/libero-100"`,
- `weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`,
- `assets_base_dir=REPO_ROOT/"assets"`,
- `batch_size=64`.

`trace_vla`: `peak_lr=5e-5`, `decay_steps=200_000`, `num_train_steps=100_000`, `ema_decay=0.999`.

`trace_vla_lora`: `peak_lr=2e-4` (LoRA-friendly), `num_train_steps=50_000`, `ema_decay=None`,
`freeze_filter=...get_freeze_filter()` from a matching `Pi0TraceVLAConfig`.

---

## 3. The 3-stream Gemma trunk (`gemmoe_trace.py`)

The trunk is a fork of `gemmoe.py` that:
- Adds a third stream.
- Replaces the learned softmax router with **hard, externally-supplied** combine weights.
- Drops the always-on shared expert.

Reusable bits from `gemmoe.py` — `Config`, `Embedder`, `RMSNorm`, `_apply_rope`,
`_gated_residual`, `_name`, `GemmoeBlockSparseTop2MLP`, `KVCache`, `PALIGEMMA_VOCAB_SIZE`,
`FeedForward as DenseFeedForward`. See [gemmoe_trace.py:43-58](pace/openpi/src/openpi/models/gemmoe_trace.py#L43-L58).

### 3.1 Variants

[gemmoe_trace.py:67-126](pace/openpi/src/openpi/models/gemmoe_trace.py#L67-L126) — `get_trace_config()`:

| Variant | width | depth | mlp_dim | num_experts | Role |
|---|---|---|---|---|---|
| `trace_moe_dummy` | 64 | 4 | 128 | 5 | Sandbox |
| `trace_moe_gemma_300m` | 1024 | 18 | 4096 | 5 | Production trace expert |
| `trace_moe_gemma_300m_lora` | 1024 | 18 | 4096 | 5 | Optional LoRA variant (unused in v1; trace stream is always full FT) |

**No shared expert**: `num_local_experts` directly = number of skill experts.

### 3.2 `Attention` (3-stream variant)

[gemmoe_trace.py:128-217](pace/openpi/src/openpi/models/gemmoe_trace.py#L128-L217)

Same structure as `gemmoe.Attention` but tolerant of an arbitrary number of streams.
Per-stream Q/K/V projections, joint attention, per-stream output projection. No
parameters are shared across streams; only the attention compute (Q·Kᵀ softmax) is.

### 3.3 `HardMoeBlock`

[gemmoe_trace.py:220-253](pace/openpi/src/openpi/models/gemmoe_trace.py#L220-L253)

K parallel `GemmoeBlockSparseTop2MLP` experts (SwiGLU FFN, no bias). Combine weights
(`(B, T, K)`, **expected to be one-hot** for hard routing) come in pre-computed.
`einsum("btk,btkd->btd", combine_weights, expert_outs)` selects the chosen expert per token.

There is **no `Router` module** here — routing is determined externally by the skill ID
(see §6.2 for how it flows in).

### 3.4 `TraceBlock`

[gemmoe_trace.py:256-322](pace/openpi/src/openpi/models/gemmoe_trace.py#L256-L322)

The per-layer block. Same shape as `gemmoe.Block`:
- Per-stream pre-attention RMSNorm (with adaRMS modulation if `adarms_cond[i]` is provided).
- Joint attention.
- Gated residual using the `gate` returned from RMSNorm.
- Per-stream pre-FFN RMSNorm.
- Per-stream FFN: `HardMoeBlock` if `num_local_experts > 1`, else `lora.FeedForward` (dense).
- Gated residual.

Streams whose token tensor is `None` are skipped at every site (FFN, RMSNorm, residual)
so the unused stream costs nothing per layer.

### 3.5 `TraceModule`

[gemmoe_trace.py:325-415](pace/openpi/src/openpi/models/gemmoe_trace.py#L325-L415)

The top-level transformer. Identical interface contract to `gemmoe.Module` except:
- The `cond` argument is **renamed `hard_combine_weights`** and is now `(B, T_trace, K)`
  one-hot (no learned router).
- The number of streams is arbitrary (we pass 3).
- `init()` exercises all streams as non-None so all per-stream params are created.

The `nn.scan` setup at line 351 mirrors `gemmoe.Module` exactly:
```python
self.layers = nn.scan(
    block_cls,
    variable_axes={"params": 0},
    split_rngs={"params": True, "dropout": True},
    in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
    length=self.configs[0].depth,
)(...)
```

So per-layer params have a leading layer axis (depth=18 typically).

---

## 4. The dataset (`libero_trace_dataset.py`)

A `LeRobotDataset` subclass that joins each frame with skill/trace annotations and emits
a sample dict with everything the rest of the pipeline needs.

### 4.1 Annotation loading

[libero_trace_dataset.py:139-152](pace/openpi/src/openpi/policies/libero_trace_dataset.py#L139-L152)

Reads the two JSON files into `self.skills_by_episode` and `self.traces_by_episode`,
keyed by integer episode index. The trace coordinate space (default 256×256) is read from
the first episode's `image_width`/`image_height`.

### 4.2 Per-frame skill segment lookup

[libero_trace_dataset.py:62-66](pace/openpi/src/openpi/policies/libero_trace_dataset.py#L62-L66) — `_segment_index_for_step`: linear scan finds which segment contains the current step.

[libero_trace_dataset.py:177-194](pace/openpi/src/openpi/policies/libero_trace_dataset.py#L177-L194) — looks up `seg_idx`, reads the `skill` string, computes `skill_id = trace_utils.skill_to_expert_id(skill)`. Finds the matching `trace_seg` by `skill_index` field.

The skill→expert mapping is the canonical one specified in the design:

| Skill | Expert id |
|---|---|
| `PICKUP_FROM` | 0 |
| `PLACE_ON`, `PLACE_IN` | 1 |
| `OPEN` | 2 |
| `CLOSE` | 3 |
| `TURN_ON`, `TURN_OFF` | 4 |

Defined in [trace_utils.py:30-49](pace/openpi/src/openpi/models/trace_utils.py#L30-L49).

### 4.3 Anchor-age augmentation

[libero_trace_dataset.py:209-211](pace/openpi/src/openpi/policies/libero_trace_dataset.py#L209-L211)

```python
a = int(self.rdm.randint(0, self.h_train_max)) if self.h_train_max > 1 else 0
t_anchor = max(seg_start, episode_step - a)
```

This implements the receding-horizon training trick: the model sees overlays anchored
anywhere from "just re-planned" (a=0) to "stalest" (a=H_train_max-1). Both the
**ground-truth trace target** and the **rendered overlay** are computed from the same
`t_anchor`, so the trace expert and action expert are trained on the same age distribution.

`H_train_max` is the **max-backward training window**, sized slightly larger than the
deployment re-plan period `F` (the constraint is `F ≤ H_train_max`).

### 4.4 Trace target construction

[libero_trace_dataset.py](pace/openpi/src/openpi/policies/libero_trace_dataset.py)

```python
ee_full = np.asarray(trace_seg["end_effector_trace"]["trace"], dtype=np.float32)  # (T_seg, 2)
sem_pt_pixel = np.asarray(sem.get("point", [0, 0]), dtype=np.float32)              # [x, y]

# Normalize semantic target into [0, 1].
sem_target_xy_norm = [sem_pt_pixel[0] / (W-1), sem_pt_pixel[1] / (H-1)]

# Current EE pixel at this frame.
cur_ee_pixel = ee_full[t_now_in_seg]
cur_ee_xy_norm = [.../W, .../H]

# Trace generator supervision target: ALWAYS from t_now to skill end.
# This makes future_trace_xy[0] equal current_ee_xy by construction, matching
# the model's inpainting clamp at x_t[:, 0, :].
seg_residual_target = ee_full[t_now_in_seg:]
future_trace_xy_norm = trace_utils.resample_trace(
    seg_residual_target / [W, H], n_out=trace_horizon, method=trace_resample
)

# Action-head overlay trace: from t_anchor (which can be older than t_now) to skill end.
# When `a > 0` this represents a "stale plan" — what the action head will see at
# inference under receding-horizon planning. When `a == 0` it equals the trace target.
seg_residual_overlay = ee_full[t_anchor_in_seg:]
overlay_trace_xy_norm = trace_utils.resample_trace(
    seg_residual_overlay / [W, H], n_out=trace_horizon, method=trace_resample
)

# Progress in [0, 1].
progress = (episode_step - seg_start) / (seg_end - seg_start)
has_trace = True
```

The resampling itself lives in [trace_utils.py:81-103](pace/openpi/src/openpi/models/trace_utils.py#L81-L103) — it uses cumulative arc length and `np.interp` to produce N points evenly spaced along the polyline length.

**Two distinct trace polylines:**

| Variable | Slice used | Consumer |
|---|---|---|
| `future_trace_xy_norm` | `ee_full[t_now_in_seg:]` | trace generator flow-matching target (and the row-0 inpainting anchor matches current_ee by construction) |
| `overlay_trace_xy_norm` | `ee_full[t_anchor_in_seg:]` | rendered into the action-head overlay image (so the action head learns to consume stale plans) |

**Invariant** (now exact): `future_trace_xy_norm[0] == cur_ee_xy_norm` always, so the inpainting clamp at row 0 does not contradict the supervision target.

### 4.5 Scene dropout (training tricks 12 and 13)

[libero_trace_dataset.py:268-282](pace/openpi/src/openpi/policies/libero_trace_dataset.py#L268-L282)

```python
if self.scene_dropout_rate > 0.0 and not self.is_computing_norm_stats:
    if self.rdm.rand() < self.scene_dropout_rate:
        # Planning-mode dropout: zero out the clean base image.
        base_image_np = np.zeros_like(base_image_np)
    if has_trace and self.rdm.rand() < self.scene_dropout_rate:
        # Execution-mode dropout: replace overlay-on-image with overlay-on-zeros.
        overlay_image_np = trace_utils.draw_polyline_overlay(
            np.zeros_like(base_image_np), future_trace_xy_norm, ...,
        )
```

Two independent dropout draws per sample:
- The **clean image** (used by planning forward) gets zeroed with prob `0.15`.
- The **overlay image** (used by execution forward) is replaced with overlay-on-zeros
  with prob `0.15`, keeping the trace visible but stripping scene cues.

### 4.6 Skill-end action zeroing

[libero_trace_dataset.py:296-310](pace/openpi/src/openpi/policies/libero_trace_dataset.py#L296-L310)

The action chunk is sliced to `[idx, seg_end_idx_global)`:
```python
seg_end_idx_global = start_idx + seg_end
slice_end = min(seg_end_idx_global, idx + (action_horizon - 1) * down_sample + 1)
actions_chunk = self.actions[idx:slice_end:self.action_down_sample_steps]
final_actions = pad_skill_horizon_actions(actions_chunk, self.action_horizon)
```

`pad_skill_horizon_actions` is **reused unmodified** from `libero_reason_dataset` — it
zero-fills the first 6 dims (pose deltas) and replicates the last gripper command for
the remaining steps in the action horizon. Confirmed correct against the Libero schema
(7-dim raw actions: 6 delta-pose + 1 gripper).

### 4.7 Per-sample output dict

[libero_trace_dataset.py:330-365](pace/openpi/src/openpi/policies/libero_trace_dataset.py#L330-L365)

```python
return_dict = {
    "observation/image":         <H, W, 3 uint8>,        # clean (post-dropout)
    "observation/wrist_image":   <H, W, 3 uint8>,
    "observation/overlay_image": <H, W, 3 uint8>,        # base + GT trace overlay (post-dropout)
    "observation/state":         <8,> float32,            # 3 EEF pos + 3 EEF rot + 2 gripper
    "actions":                   <ah, 7> float32,         # zero-padded at skill end
    "action_is_pad":             <ah,> bool,
    "atomic_token":              float (skill id),
    "skill_name":                str (e.g. "PICKUP_FROM"),
    "semantic_target_xy":        <2,> float32 in [0, 1]^2,
    "current_ee_xy":             <2,> float32 in [0, 1]^2,
    "future_trace_xy":           <N, 2> float32 in [0, 1]^2,
    "has_trace":                 bool,
    "has_overlay":               bool,
    "progress":                  float in [0, 1],
    "diffusion_loss_mask":       bool (always True),
    "prompt":                    str (LIBERO task instruction),
    ...
}
```

---

## 5. Data transforms (`libero_trace_policy.py`)

### 5.1 `LiberoTraceInputs`

[libero_trace_policy.py:31-92](pace/openpi/src/openpi/policies/libero_trace_policy.py#L31-L92)

Repacks the dataset dict into the model's expected schema. Most importantly it
constructs both image dicts:
```python
inputs["image"] = {
    "base_0_rgb":         base_image,
    "left_wrist_0_rgb":   wrist_image,
    "right_wrist_0_rgb":  zeros_like(base_image),  # Libero has no right wrist
}
inputs["image_mask"] = {
    "base_0_rgb": True_, "left_wrist_0_rgb": True_, "right_wrist_0_rgb": False_,
}

if "observation/overlay_image" in data:
    inputs["overlay_image"]      = {"base_0_rgb": overlay_image}
    inputs["overlay_image_mask"] = {"base_0_rgb": True_}
```

### 5.2 `TraceResizeImages`

[libero_trace_policy.py:96-108](pace/openpi/src/openpi/policies/libero_trace_policy.py#L96-L108)

Resizes both `image` and `overlay_image` dicts to model resolution (224×224), via
`openpi_client.image_tools.resize_with_pad`. We do **not** reuse the stock
`transforms.ResizeImages` because it only handles `image` — overlay images would
otherwise stay at 256×256 and get resized later inside the model.

### 5.3 `TraceTokenizePrompt`

[libero_trace_policy.py](pace/openpi/src/openpi/policies/libero_trace_policy.py)

Tokenizes the prompt using the standard `PaligemmaTokenizer` (no atomic-VLA-style
suffix logic). Per the design, the VLM consumes the *externally-selected skill* as
its language input, not the raw LIBERO task instruction. The transform prefers the
parameterized skill expression and falls back gracefully:

```python
if skill_text:
    full_prompt = skill_text             # e.g. "PICKUP_FROM(white mug, table)"
elif skill_name:
    full_prompt = skill_name             # e.g. "PICKUP_FROM"
else:
    full_prompt = prompt                 # only used if skill annotations are missing
tokens, token_mask = self.tokenizer.tokenize(full_prompt, state=state_or_None)
```

`PaligemmaTokenizer.tokenize` normalizes underscores to spaces, so the VLM sees
`"PICKUP FROM(white mug, table)"` after detokenize. The original task instruction is
*intentionally* dropped here — at deployment, an off-the-shelf high-level VLM produces the
skill, and the VLA only needs to see that skill.

We zero out `token_ar_mask` (no autoregressive prediction needed — TraceVLA has no language
head) and `token_loss_mask` (no text loss). The skill is also routed to the trace MoE via
the `atomic_token` field in parallel (§6.2).

`skill_text` and `skill_name` are popped from `data` after use because the torch default
collator cannot stack variable-length strings into a batch tensor.

### 5.4 `LiberoTraceOutputs`

[libero_trace_policy.py:111-118](pace/openpi/src/openpi/policies/libero_trace_policy.py#L111-L118)

Trims the action vector back to its original 7 dims at inference time (training-time
actions are zero-padded to `action_dim=32` to match the pretrained pi05 checkpoint).

---

## 6. The model (`pi0_trace_vla.py`)

### 6.1 Constructor

[pi0_trace_vla.py:110-198](pace/openpi/src/openpi/models/pi0_trace_vla.py#L110-L198)

Three big things happen:

1. **Build the 3-stream trunk.** Lines 130-148:
   ```python
   llm = nnx_bridge.ToNNX(
       _gemma_trace.TraceModule(
           configs=[paligemma_config, action_expert_config, trace_expert_config],
           embed_dtype=config.dtype,
       )
   )
   llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True, True])
   ```
   adaRMS is enabled on streams 1 (action) and 2 (trace).

2. **Per-stream I/O projections** (lines 152-167):
   - `action_in_proj : R^32 → R^1024`, `action_out_proj : R^1024 → R^32`.
   - `trace_in_proj : R^2 → R^1024`, `trace_out_proj : R^1024 → R^2`.
   - `action_time_mlp_in/out`, `trace_time_mlp_in/out` (separate time MLPs per expert).

3. **Conditioning + completion head** (lines 170-198):
   - `target_mlp_in/out`: 32-dim Fourier feature → 1024-dim adaRMS contribution.
   - `completion_shared_in/out`: 2048 → 256 → 256 (shared compressor).
   - `cmp_w1`, `cmp_b1`, `cmp_w2`, `cmp_b2`: stacked weights of shape
     `(K, 256, 64)`, `(K, 64)`, `(K, 64, 1)`, `(K, 1)` for K=5 per-skill heads.

### 6.2 Skill → expert routing

[pi0_trace_vla.py:382-389](pace/openpi/src/openpi/models/pi0_trace_vla.py#L382-L389) (inside `_forward_planning`):

```python
skill_id = obs.atomic_token.astype(jnp.int32)              # (B,)
skill_one_hot = jax.nn.one_hot(skill_id, num_trace_experts) # (B, K)
combine_weights = jnp.broadcast_to(
    skill_one_hot[:, None, :],
    (skill_one_hot.shape[0], self.trace_horizon, self.num_trace_experts),
)
```

`atomic_token` ∈ {0, 1, 2, 3, 4} is set by the dataset (§4.2). Hard one-hot, broadcast
across all N trace tokens, then handed to `TraceModule` as `hard_combine_weights`. Inside
each `TraceBlock`, every layer routes every trace token to the chosen expert FFN. **No
learned router parameters anywhere in the trace stack.**

### 6.3 Embedding the prefix

[pi0_trace_vla.py:201-236](pace/openpi/src/openpi/models/pi0_trace_vla.py#L201-L236) — `_embed_prefix_with_images`:

Same logic as `pi0_atomic.embed_prefix` but takes the images dict explicitly so we can
call it twice per training step (once with clean images, once with the overlay image
swapped in).

```python
for name in images:
    image_tokens, _ = self.PaliGemma.img(images[name], train=False)  # SigLIP encoder
    tokens.append(image_tokens)
    input_mask.append(repeat(image_masks[name], "b -> b s", s=image_tokens.shape[1]))
    ar_mask.append(0 * input_mask[-1])

txt_emb = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
tokens.append(txt_emb)
input_mask.append(obs.tokenized_prompt_mask)
ar_mask.append(obs.token_ar_mask)
```

The prefix is non-causal among image+language tokens (`ar_mask = 0` for image, language has
its own ar_mask).

### 6.4 Embedding the action suffix

[pi0_trace_vla.py:238-261](pace/openpi/src/openpi/models/pi0_trace_vla.py#L238-L261) — `_embed_action_suffix`:

```python
action_tokens = self.action_in_proj(noisy_actions)                     # (B, ah, 1024)
time_emb = posemb_sincos(timestep, action_width, 4e-3, 4.0)            # (B, 1024)
time_emb = swish(self.action_time_mlp_in(time_emb))
time_emb = swish(self.action_time_mlp_out(time_emb))
adarms_cond = time_emb                                                  # (B, 1024)
```

`adarms_cond` is the per-block modulation vector for stream 1.

### 6.5 Embedding the trace suffix (with target conditioning)

[pi0_trace_vla.py:263-296](pace/openpi/src/openpi/models/pi0_trace_vla.py#L263-L296) — `_embed_trace_suffix`:

```python
trace_tokens = self.trace_in_proj(noisy_trace)                          # (B, N, 1024)

# Time MLP (same shape as the action expert's, but separate weights).
time_emb = posemb_sincos(timestep, trace_width, 4e-3, 4.0)
time_emb = swish(self.trace_time_mlp_in(time_emb))
time_emb = swish(self.trace_time_mlp_out(time_emb))

# Fourier-encoded semantic target → MLP → same dim.
tgt_feat = fourier_encode_2d(target_xy, num_freqs=8)                    # (B, 32)
tgt_emb = swish(self.target_mlp_in(tgt_feat))
tgt_emb = swish(self.target_mlp_out(tgt_emb))

adarms_cond = time_emb + tgt_emb                                        # (B, 1024)
```

This is the **only** place the semantic target enters the model: as a global modulation
vector, summed alongside time, at every block of stream 2. The target is **not** added
as a VLM prefix token. (See §11.4 of `TRACE_VLA_DESIGN.md`.)

`fourier_encode_2d` is in [pi0_trace_vla.py:84-99](pace/openpi/src/openpi/models/pi0_trace_vla.py#L84-L99): for each of the 2 normalized coordinates, evaluates sin/cos at 8 geometric frequencies, giving a 32-dim feature.

### 6.6 Inpainting clamp on the trace's first row

[pi0_trace_vla.py:354-360](pace/openpi/src/openpi/models/pi0_trace_vla.py#L354-L360) (in `_forward_planning`):

```python
ee = obs.current_ee_xy  # (B, 2) normalized
# Forward-process clamp: same noise level as the rest of x_t but mean centered at p_ee.
x_t_row0 = (1.0 - time[:, None]) * ee + time[:, None] * noise[:, 0, :]
x_t = x_t.at[:, 0, :].set(x_t_row0)
```

This is the **only** place the EE pixel coordinate enters the model. Critically:
- The variance of row 0 is `t²`, matching the rest of `x_t` (no SNR mismatch).
- The mean is `(1-t)·p_ee` instead of `(1-t)·future_trace[:, 0]`; under the dataset's
  contract these are nearly equal.
- Loss on row 0 is masked out (§7.1) since the model is just reconstructing a value it
  already knows.

The same clamp is applied at every Euler step inside `sample_trace` ([pi0_trace_vla.py:649-653](pace/openpi/src/openpi/models/pi0_trace_vla.py#L649-L653)), with a fresh `eps_row0` re-sampled per step.

### 6.7 Planning forward pass

[pi0_trace_vla.py:332-407](pace/openpi/src/openpi/models/pi0_trace_vla.py#L332-L407) — `_forward_planning`:

1. Sample noise + flow-matching time `t ~ Beta(1.5, 1)` (line 343).
2. Form `x_t = t·noise + (1-t)·future_trace` (line 348).
3. Compute target velocity `u_t = noise - future_trace` (line 349).
4. Apply inpainting clamp to row 0 (lines 354-360).
5. Build prefix using **clean images** (`obs.images`, `obs.image_masks`) (line 365).
6. Build trace suffix with target conditioning (line 374).
7. Build hard combine weights from `atomic_token` (lines 382-389).
8. Forward through `TraceModule` with streams `[prefix, None, trace_suffix]` (lines 396-402).
9. `v_t = trace_out_proj(trace_out[:, -N:])` and return `(v_t, u_t, loss_mask)` where
   `loss_mask` zeros out row 0.

### 6.8 Execution forward pass

[pi0_trace_vla.py:409-477](pace/openpi/src/openpi/models/pi0_trace_vla.py#L409-L477) — `_forward_execution`:

Same overall structure but for actions:

1. Sample noise + time, form `x_a_t`, `u_a_t` (lines 421-426).
2. **Build images for execution** by overlaying `obs.overlay_images['base_0_rgb']` onto
   the wrist images (lines 428-435):
   ```python
   exec_images = dict(obs.images)
   if obs.overlay_images is not None:
       for k, v in obs.overlay_images.items():
           exec_images[k] = v
   ```
3. Build prefix using these overlay-augmented images (line 437).
4. Build action suffix (line 442).
5. Forward through `TraceModule` with streams `[prefix, action_suffix, None]` and a dummy
   placeholder for `hard_combine_weights` (lines 444-460).
6. `v_a_t = action_out_proj(action_out[:, -ah:])`.
7. **Completion head**: `progress_pred = self._completion_predict(prefix_out, prefix_mask, skill_id)` (line 469).

The placeholder `dummy_weights` (line 456) is needed to satisfy `nn.scan`'s shape contract;
the `HardMoeBlock` is never invoked because the trace stream is `None`.

### 6.9 Completion head

[pi0_trace_vla.py:299-330](pace/openpi/src/openpi/models/pi0_trace_vla.py#L299-L330):

```python
# 1. Mean-pool the VLM prefix output, weighted by validity mask.
m = prefix_mask.astype(prefix_out.dtype)
denom = jnp.maximum(jnp.sum(m, axis=-1, keepdims=True), 1.0)
h_pool = jnp.sum(prefix_out * m[..., None], axis=1) / denom              # (B, 2048)

# 2. Shared compression.
h = swish(self.completion_shared_in(h_pool))
h = self.completion_shared_out(h)                                         # (B, 256)

# 3. Per-skill stacked MLPs. Compute outputs for ALL K experts then gather.
h1 = einsum("bs,ksh->bkh", h, cmp_w1) + cmp_b1[None, :, :]
h1 = swish(h1)
out = einsum("bkh,kho->bko", h1, cmp_w2) + cmp_b2[None, :, :]
out = out[..., 0]                                                          # (B, K)

# 4. Hard route by skill_id, sigmoid → progress.
skill_one_hot = jax.nn.one_hot(skill_id, K)
logit = einsum("bk,bk->b", skill_one_hot, out)
return jax.nn.sigmoid(logit)                                               # (B,) in [0, 1]
```

Total head parameters: **~670K** (525K shared + 5×16K per-skill). Negligible vs. the 2.5B
trunk.

---

## 7. Loss computation

[pi0_trace_vla.py:479-528](pace/openpi/src/openpi/models/pi0_trace_vla.py#L479-L528) — `compute_loss`:

```python
preprocess_rng, plan_rng, exec_rng = jax.random.split(rng, 3)
observation = preprocess_trace_observation(preprocess_rng, observation, train=train, ...)

# Trace planning
v_t, u_t, trace_loss_mask = self._forward_planning(plan_rng, observation)

# Action + completion (single execution forward pass)
v_a, u_a, progress_pred = self._forward_execution(exec_rng, observation, actions)
```

The three losses live in lines 495-516.

### 7.1 Trace flow-matching loss

```python
per_pt_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)      # (B, N)
has_trace   = obs.has_trace.astype(per_pt_loss.dtype)        # (B,)
mask        = trace_loss_mask                                # (B, N), zero at row 0
denom       = jnp.maximum(jnp.sum(mask, axis=-1), 1.0)
trace_loss  = jnp.sum(per_pt_loss * mask, axis=-1) / denom * has_trace
```

Two layers of masking:
- **Row mask**: row 0 (the inpainted EE anchor) is excluded.
- **Sample mask**: `has_trace=False` samples (no annotation found) contribute zero loss.

### 7.2 Action flow-matching loss

```python
action_loss = jnp.mean(jnp.mean(jnp.square(v_a - u_a), axis=-1), axis=1)  # (B,)
```

No row masking. The action chunk is already correctly truncated by `pad_skill_horizon_actions`
in the dataset (§4.6) — the "stationary" tail is a legitimate target (zero pose deltas +
held gripper).

### 7.3 Completion loss

```python
progress_target = obs.progress
completion_loss = jnp.square(progress_pred - progress_target) * has_trace
```

MSE on the scalar progress, only for samples with valid trace annotations.

### 7.4 Total loss

```python
total_loss = (action_loss_coeff * action_loss
            + trace_loss_coeff  * trace_loss
            + completion_loss_coeff * completion_loss)

info = {"action_loss": ..., "trace_loss": ..., "completion_loss": ..., "progress_pred_mean": ...}
return total_loss, info
```

Default coefficients: `action_loss_coeff=1.0`, `trace_loss_coeff=1.0`, `completion_loss_coeff=0.1`.
All three loss components are returned in `info` and logged separately to wandb.

---

## 8. Training script (`train_trace_vla.py`)

### 8.1 Weight loading remap

[train_trace_vla.py](pace/openpi/scripts/train_trace_vla.py) — `_load_and_filter_weights`:

The pi05_base checkpoint contains weights for streams 0 (paligemma) and 1 (action expert).
We need to populate stream 2 (trace expert) too.

**Step 1 — Replicate stream-1 attention/norm weights to stream-2:**

The action expert variant `gemma_300m` (and `gemma_300m_lora`) has `num_heads=8`,
`num_kv_heads=1`, so attention uses *split* `q_einsum_1` + `kv_einsum_1` projections
rather than the fused `qkv_einsum_1`. The remap therefore explicitly covers all three
spellings (the fused one is harmless when the variant doesn't use it):

```python
suffix_pairs = [
    ("q_einsum_2", "q_einsum_1"),
    ("kv_einsum_2", "kv_einsum_1"),
    ("qkv_einsum_2", "qkv_einsum_1"),
    ("attn_vec_einsum_2", "attn_vec_einsum_1"),
    ("pre_attention_norm_2", "pre_attention_norm_1"),
    ("pre_ffw_norm_2", "pre_ffw_norm_1"),
    ("final_norm_2", "final_norm_1"),
]
for trg_suffix, src_suffix in suffix_pairs:
    for k in flat_loaded:
        if src_suffix in k:
            trg_key = tuple(seg if seg != src_suffix else trg_suffix for seg in k)
            flat_loaded[trg_key] = flat_loaded[k]
```

**Step 2 — Fan out the dense FFN `mlp_1` into the K hard MoE experts** (lines 105-127):

```python
gating_keys = [k for k in flat_loaded if k[-2:] == ("mlp_1", "gating_einsum")]
linear_keys = [k for k in flat_loaded if k[-2:] == ("mlp_1", "linear")]

for k in gating_keys:
    gating = flat_loaded[k]                       # (L, 2, in, hidden) — dense GeGLU weights
    w1 = gating[..., 0, :, :]                     # → SwiGLU expert's w1 (gate path)
    w3 = gating[..., 1, :, :]                     # → expert's w3 (linear path)
    prefix = k[:-1]                                # path up to "mlp_1"
    for e in range(num_trace_experts):
        flat_loaded[(*prefix[:-1], "moe_2", f"expert_{e}", "w1", "kernel")] = w1
        flat_loaded[(*prefix[:-1], "moe_2", f"expert_{e}", "w3", "kernel")] = w3

for k in linear_keys:
    linear = flat_loaded[k]                        # (L, hidden, in)
    prefix = k[:-1]
    for e in range(num_trace_experts):
        flat_loaded[(*prefix[:-1], "moe_2", f"expert_{e}", "w2", "kernel")] = linear
```

So all 5 trace experts start as **identical copies** of the pi05_base action expert FFN.
Hard routing during training will diverge them.

Note: pi05_base's dense FFN uses **GELU**, whereas `GemmoeBlockSparseTop2MLP` uses **SiLU**.
This is the same activation mismatch AtomicVLA accepts; finetuning bridges it quickly.

### 8.2 `init_train_state`

[train_trace_vla.py:145-197](pace/openpi/scripts/train_trace_vla.py#L145-L197)

Standard pattern (matches `train_atomic.py`):

1. Build optimizer.
2. Inner `init` function: create model, optionally apply `partial_params` from
   `_load_and_filter_weights`, cast frozen params to bf16, init optimizer state on
   `params.filter(trainable_filter)`.
3. `jax.eval_shape` to get train state shape, then `fsdp_sharding`.
4. Resume path returns `(train_state_shape, state_sharding)`.
5. Fresh-init path JITs the inner init with proper sharding and returns
   `(train_state, state_sharding)`.

The `trainable_filter` is computed as `nnx.All(nnx.Param, nnx.Not(config.freeze_filter))`
(implicitly via `config.trainable_filter` property). For `trace_vla` (full FT), this
matches everything; for `trace_vla_lora`, this matches only LoRA + trace expert + non-LLM.

### 8.3 `train_step`

[train_trace_vla.py:199-256](pace/openpi/scripts/train_trace_vla.py#L199-L256)

```python
def loss_fn(model, rng, observation, actions):
    per_sample, info = model.compute_loss(rng, observation, actions, train=True)
    return jnp.mean(per_sample), info

(loss, train_info), grads = nnx.value_and_grad(
    loss_fn, argnums=nnx.DiffState(0, config.trainable_filter), has_aux=True
)(model, train_rng, observation, actions)

params  = state.params.filter(config.trainable_filter)
updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
new_params = optax.apply_updates(params, updates)
nnx.update(model, new_params)
new_params = nnx.state(model)
new_state = dataclasses.replace(state, step=state.step+1, params=new_params, opt_state=new_opt_state)
```

The `nnx.DiffState(0, trainable_filter)` ensures gradient is computed only on trainable
params (LoRA + trace expert + completion head + I/O projections in the LoRA case).

EMA is applied if `ema_decay` is not None (full FT only; LoRA disables it).

The `info` dict (line 254) includes `loss`, `grad_norm`, `param_norm`, plus all
per-component losses (`action_loss`, `trace_loss`, `completion_loss`, `progress_pred_mean`).

### 8.4 Data loader

[train_trace_vla.py](pace/openpi/scripts/train_trace_vla.py) — `_create_trace_data_loader`:

```python
data_config = config.data.create(config.assets_dirs, config.model)
dataset     = LiberoTraceDataset(data_config, action_horizon=...)
transformed = transform_dataset(dataset, data_config, skip_norm_stats=False)

torch_loader = TorchDataLoader(transformed, local_batch_size=..., sharding=..., num_workers=...)

class _Wrapper:
    def __iter__(self):
        for batch in self._loader:
            yield TraceObservation.from_dict(batch), batch["actions"]
```

We **reuse the unmodified `TorchDataLoader` and `transform_dataset`** from
`openpi.training.data_loader`. The thin `_Wrapper` constructs `TraceObservation` instead
of going through `DataLoaderImpl`'s built-in dispatch (which only knows about
`Observation`/`FuseObservation`/`AtomicObservation`).

**Norm stats**: training requires the standard openpi norm-stats workflow. Run the
following once before launching training:

```bash
python pace/openpi/scripts/compute_norm_stats.py --config-name trace_vla
# (and / or)
python pace/openpi/scripts/compute_norm_stats.py --config-name trace_vla_lora
```

This computes mean/std/quantile statistics for `state` and `actions` and writes them
under `assets/yilin-wu/libero-100/`. `_create_trace_data_loader` then sets
`skip_norm_stats=False`, matching the baseline `pi05_libero_100` and
`Atomic_libero` paths so the pretrained `pi05_base` action head sees normalized
targets at the same scale it was pretrained on.

`create_torch_dataset` includes a branch for `LiberoTraceDataConfig` so
`compute_norm_stats.py` can iterate the trace dataset directly.

### 8.5 Main loop

[train_trace_vla.py:301-376](pace/openpi/scripts/train_trace_vla.py#L301-L376)

Identical structure to `train_atomic.py`:
- `init_logging` + `init_wandb`.
- Build mesh, sharding, checkpoint manager.
- Build data iter, prime first batch.
- Init train state (with weight remapping).
- JIT the train step.
- Loop: `train_step`, log every `log_interval`, save every `save_interval`.

---

## 9. TraceObservation (`trace_observation.py`)

### 9.1 Dataclass

[trace_observation.py:23-46](pace/openpi/src/openpi/models/trace_observation.py#L23-L46)

Subclasses `_model.Observation` and adds:

| Field | Shape | Use |
|---|---|---|
| `atomic_token` | `(B,)` float32 | skill ID; cast to int32 in the model for routing |
| `semantic_target_xy` | `(B, 2)` float32 | normalized [0,1] pixel for AdaRMS conditioning |
| `current_ee_xy` | `(B, 2)` float32 | normalized [0,1] pixel for inpainting clamp |
| `has_trace` | `(B,)` bool | mask for trace + completion losses |
| `has_overlay` | `(B,)` bool | informational; not used for masking |
| `progress` | `(B,)` float32 | completion target |
| `diffusion_loss_mask` | `(B,)` bool | inherited from AtomicObservation convention |
| `future_trace_xy` | `(B, N, 2)` float32 | trace flow-matching target |
| `overlay_images` | dict | overlay version of `base_0_rgb` only |
| `overlay_image_masks` | dict | parallel mask |

### 9.2 `from_dict`

[trace_observation.py:48-94](pace/openpi/src/openpi/models/trace_observation.py#L48-L94)

Mirrors `Observation.from_dict`'s uint8→float32 conversion in `[-1, 1]` for both `image`
and `overlay_image` dicts. Other fields pass through.

### 9.3 `preprocess_trace_observation`

[trace_observation.py](pace/openpi/src/openpi/models/trace_observation.py)

Custom image-and-keypoint preprocessor that, when `train=True`, applies the **same**
random geometric transform jointly to:

- the base camera image,
- the overlay version of the base image,
- the semantic-target keypoint,
- the current EE keypoint,
- every waypoint of the supervised future trace.

This relies on augmax's native `InputType.KEYPOINTS` support and a single
`augmax.Chain(..., input_types=[IMAGE, IMAGE, KEYPOINTS, KEYPOINTS, KEYPOINTS])`,
so the same RNG produces deterministically-coherent transforms across all inputs.
After the chain, keypoints are converted from pixel space back to normalized [0, 1]
and clamped to the image bounds (out-of-bounds is rare for LIBERO since the
workspace is well inside the camera frame).

Wrist images go through a *separate* color-only chain (no geometric transform) — the
same wrist policy as `_model.preprocess_observation`. ColorJitter, RandomCrop,
Resize and Rotate parameters are all derived from the same per-batch sub-RNG, so
within a sample the base image, the overlay, and all the keypoints undergo identical
geometry; the wrist images get matched color jitter.

`train=False` skips augmentation entirely and only resizes images to the target
resolution. Keypoints pass through unchanged.

---

## 10. Trace utilities (`trace_utils.py`)

### 10.1 Skill mapping

[trace_utils.py:17-49](pace/openpi/src/openpi/models/trace_utils.py#L17-L49)

```python
SKILL_TO_EXPERT = {
    "PICKUP_FROM": 0,
    "PLACE_ON":    1, "PLACE_IN": 1,
    "OPEN":        2,
    "CLOSE":       3,
    "TURN_ON":     4, "TURN_OFF": 4,
}
NUM_TRACE_EXPERTS = 5
```

`skill_to_expert_id(skill)` strips `PICKUP_FROM(white mug, table)` → `PICKUP_FROM` then
looks up the table (with default 0 for unknowns).

### 10.2 Resampling

[trace_utils.py:66-117](pace/openpi/src/openpi/models/trace_utils.py#L66-L117)

Two algorithms:
- `time_uniform_resample` — `linspace` over indices, linear interp.
- `arc_length_resample` — cumulative arc length, then `linspace` over total length, then `np.interp` per coord.

Both have edge cases for empty traces, single-point traces, and stationary traces (total
arc length zero) → fall back to replicating the start point.

`resample_trace(trace, n_out, method)` is the dispatcher used by the dataset.

### 10.3 Polyline overlay

[trace_utils.py:121-186](pace/openpi/src/openpi/models/trace_utils.py#L121-L186) — Xiaolin Wu's antialiased line algorithm.

[trace_utils.py:189-205](pace/openpi/src/openpi/models/trace_utils.py#L189-L205) — small filled disk for endpoint markers.

[trace_utils.py:207-275](pace/openpi/src/openpi/models/trace_utils.py#L207-L275) — `draw_polyline_overlay`:
- Takes an HxWx3 uint8 image and an `(N, 2)` normalized-coord polyline.
- De-normalizes to pixel space.
- Draws each segment with antialiased lines; for `line_thickness > 1`, draws additional
  parallel offset lines.
- Drops a small disk at the start and end.
- Returns a fresh image (does not mutate the input).

Default rendering style (set by the data config):
- `overlay_color=(0, 255, 255)` (cyan — high contrast, common in web images so SigLIP
  features should respond).
- `overlay_thickness=2`.
- `overlay_endpoint_radius=2.5`.

---

## 11. Where each design point lives

A quick lookup if you want to find a specific design choice:

| Design point | Where |
|---|---|
| Hard-routed MoE (no shared, no learned router) | [gemmoe_trace.py:220-253](pace/openpi/src/openpi/models/gemmoe_trace.py#L220-L253) |
| Skill → expert mapping | [trace_utils.py:30-49](pace/openpi/src/openpi/models/trace_utils.py#L30-L49) |
| One-hot routing weights into the MoE | [pi0_trace_vla.py:382-389](pace/openpi/src/openpi/models/pi0_trace_vla.py#L382-L389) |
| Semantic target via AdaRMS (Fourier + MLP, summed with time) | [pi0_trace_vla.py:281-294](pace/openpi/src/openpi/models/pi0_trace_vla.py#L281-L294) |
| EE inpainting clamp (training) | [pi0_trace_vla.py:354-360](pace/openpi/src/openpi/models/pi0_trace_vla.py#L354-L360) |
| EE inpainting clamp (sampling) | [pi0_trace_vla.py:649-653](pace/openpi/src/openpi/models/pi0_trace_vla.py#L649-L653) |
| Trace loss row mask (skip row 0) | [pi0_trace_vla.py:399-405](pace/openpi/src/openpi/models/pi0_trace_vla.py#L399-L405), applied at [pi0_trace_vla.py:497-505](pace/openpi/src/openpi/models/pi0_trace_vla.py#L497-L505) |
| Anchor-age augmentation | [libero_trace_dataset.py:209-211](pace/openpi/src/openpi/policies/libero_trace_dataset.py#L209-L211) |
| Arc-length resampling | [trace_utils.py:81-103](pace/openpi/src/openpi/models/trace_utils.py#L81-L103) |
| Overlay rendering | [trace_utils.py:207-275](pace/openpi/src/openpi/models/trace_utils.py#L207-L275); used by [libero_trace_dataset.py:259-266](pace/openpi/src/openpi/policies/libero_trace_dataset.py#L259-L266) |
| Scene dropout (planning + execution) | [libero_trace_dataset.py:268-282](pace/openpi/src/openpi/policies/libero_trace_dataset.py#L268-L282) |
| Skill-end action zeroing | [libero_trace_dataset.py:296-310](pace/openpi/src/openpi/policies/libero_trace_dataset.py#L296-L310) (uses `pad_skill_horizon_actions` from `libero_reason_dataset`) |
| Two forward passes per train step | [pi0_trace_vla.py:494-499](pace/openpi/src/openpi/models/pi0_trace_vla.py#L494-L499) (orchestration) |
| Three losses + weighting | [pi0_trace_vla.py:497-522](pace/openpi/src/openpi/models/pi0_trace_vla.py#L497-L522) |
| Mean-pool VLM hidden states + per-skill MLP | [pi0_trace_vla.py — `_completion_predict`](pace/openpi/src/openpi/models/pi0_trace_vla.py) |
| LoRA freeze filter (3-stream aware) | [pi0_trace_vla_config.py:150-167](pace/openpi/src/openpi/models/pi0_trace_vla_config.py#L150-L167) |
| pi05_base → trace expert weight remap (incl. q_einsum / kv_einsum) | [train_trace_vla.py — `_load_and_filter_weights`](pace/openpi/scripts/train_trace_vla.py) |
| Per-loss wandb logging | [train_trace_vla.py — `train_step`/`main`](pace/openpi/scripts/train_trace_vla.py) |
| Inference: action + completion (combined, shared prefill) | [pi0_trace_vla.py — `sample_actions_and_completion`](pace/openpi/src/openpi/models/pi0_trace_vla.py) |
| Inference: completion-only standalone path | [pi0_trace_vla.py — `predict_completion`](pace/openpi/src/openpi/models/pi0_trace_vla.py) |
| Public policy class | [policy.py — `TraceVLAPolicy`](pace/openpi/src/openpi/policies/policy.py) |
| Policy factory | [policy_config.py — `create_trained_trace_vla_policy`](pace/openpi/src/openpi/policies/policy_config.py) |

---

## 12. End-to-end flow diagram

```
                                   train_trace_vla.py
                                          │
       ┌──────────────────────────────────┴──────────────────────────────────┐
       │                                                                     │
       │  config.get_config('trace_vla' or 'trace_vla_lora')                 │
       │       → Pi0TraceVLAConfig (model)                                   │
       │       → LeRobotTraceVLADataConfig → LiberoTraceDataConfig (data)    │
       │                                                                     │
       └──────────────────────────────────┬──────────────────────────────────┘
                                          │
                                          ▼
         ┌──────────────────────────────────────────────────────────────┐
         │   _create_trace_data_loader                                  │
         │     1. LiberoTraceDataset                                    │
         │        - load skill_annotations.json + skill_target_traces.json
         │        - per-frame: anchor-age sample, resample trace,       │
         │          render overlay, scene dropout, action zeroing       │
         │     2. transform_dataset                                     │
         │        - LiberoTraceInputs                                   │
         │        - TraceResizeImages (image + overlay_image)           │
         │        - TraceTokenizePrompt                                 │
         │        - PadStatesAndActions                                 │
         │     3. TorchDataLoader → batch                               │
         │     4. _Wrapper: batch → TraceObservation.from_dict          │
         └──────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
         ┌──────────────────────────────────────────────────────────────┐
         │   init_train_state                                           │
         │     - create Pi0TraceVLA (3-stream Gemma)                    │
         │     - load pi05_base, replicate _1 → _2, fan mlp_1 → moe_2   │
         │     - filter trainable params                                │
         │     - build optimizer state                                  │
         └──────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
         ┌──────────────────────────────────────────────────────────────┐
         │   JIT'd train_step                                           │
         │                                                              │
         │   model.compute_loss(rng, obs, actions, train=True):         │
         │   ┌──────────────────────────────────────────────────────┐   │
         │   │ Planning forward pass (clean image + trace stream)   │   │
         │   │   - sample t, noise; x_t = (1-t)·trace + t·noise     │   │
         │   │   - inpaint x_t[:, 0] = (1-t)·p_ee + t·ε             │   │
         │   │   - prefix: SigLIP(clean image) + tokenized prompt   │   │
         │   │   - trace suffix: trace_in_proj(x_t)                 │   │
         │   │   - adaRMS cond: time + Fourier(p_tgt) → MLP         │   │
         │   │   - hard combine_weights = one_hot(skill_id)         │   │
         │   │   - 3-stream Gemma forward [prefix, None, trace]     │   │
         │   │   - v_t = trace_out_proj(trace_out)                  │   │
         │   │   - loss: MSE(v_t, u_t), mask row 0                  │   │
         │   └──────────────────────────────────────────────────────┘   │
         │   ┌──────────────────────────────────────────────────────┐   │
         │   │ Execution forward pass (overlay image + action stream)│  │
         │   │   - sample t, noise; x_a_t = (1-t)·actions + t·noise │   │
         │   │   - prefix: SigLIP(image w/ overlay) + tokens        │   │
         │   │   - action suffix: action_in_proj(x_a_t)             │   │
         │   │   - adaRMS cond: time only                           │   │
         │   │   - 3-stream Gemma forward [prefix, action, None]    │   │
         │   │   - v_a = action_out_proj(action_out)                │   │
         │   │   - loss: MSE(v_a, u_a)                              │   │
         │   │   - completion: mean-pool prefix → per-skill MLP     │   │
         │   │       → MSE(progress_pred, progress)                  │   │
         │   └──────────────────────────────────────────────────────┘   │
         │                                                              │
         │   total = w_a·action + w_t·trace + w_c·completion            │
         │                                                              │
         │   nnx.value_and_grad → grads → optimizer step → EMA          │
         └──────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                     wandb log: loss, grad_norm, param_norm,
                                action_loss, trace_loss, completion_loss
                                          │
                                          ▼
                                  checkpoint.save_state
```

---

## 13. Sandbox-test results (already verified)

All on CPU with the dummy variants and a small batch. Confirms the pipeline works end-to-end.

| Test | Result |
|---|---|
| `trace_utils.arc_length_resample`, `time_uniform_resample`, `draw_polyline_overlay`, `skill_to_expert_id`, `hard_route_one_hot` | ✓ shapes correct, polyline draws to image |
| 3-stream `TraceModule` instantiation + forward (with one stream None) | ✓ output shapes match `(B, P, D₀)`, `None`, `(B, N, D₂)` |
| `Pi0TraceVLA` instantiation (dummy variants) | ✓ |
| `compute_loss` with synthetic obs → returns 3 separate losses | ✓ |
| `compute_loss` end-to-end with real LIBERO data through transforms | ✓ all losses finite, dataset fetches correct fields |
| Gradient flow through `nnx.value_and_grad` | ✓ trainable leaves get gradients |
| One full optimizer step → loss drops 4.31 → 3.98 | ✓ |
| `LiberoTraceDataset` returns `future_trace_xy[0] == current_ee_xy` (inpainting consistency) | ✓ |
| LoRA freeze filter | ✓ paligemma + action expert frozen; trace expert + LoRA + completion trainable |

---

## 14. What is intentionally NOT in v1 (and where to add later)

These were design discussions that we deferred. Each has a clear extension point.

| Future addition | Where it would slot in |
|---|---|
| Endpoint regression auxiliary loss (`||τ̂[N-1] − p_tgt||²`) | Add to `compute_loss` after `_forward_planning`, before `info` dict construction |
| Anchor-noise augmentation (random pixel jitter / thinning on the overlay) | Modify `draw_polyline_overlay` to accept a `noise_sigma` parameter, called from `LiberoTraceDataset.__getitem__` |
| Classifier-free guidance dropout on `p_tgt` | Add a learned `null_target_emb` Param; in `_embed_trace_suffix`, randomly substitute `target_xy` with the null embedding during training |
| Add `p_tgt` as VLM prefix token (if AdaRMS-only is too weak) | Augment `_embed_prefix_with_images` to optionally append a Fourier-encoded target token |
| Action expert overlay-aware exposure when overlay is missing | Add a `has_overlay`-aware fallback in `_forward_execution` (currently always uses overlay if present) |
| Closed-loop trace generation during training | Roll out `sample_trace` (no-grad), use the rolled-out trace as the overlay |
| Stateful closed-loop policy (cache trace internally; auto-replan on progress threshold) | Subclass `TraceVLAPolicy`; cache `last_trace` and a step counter; call `sample_trace` when stale or when `progress` exceeds threshold |
| Progress target reaching exactly 1.0 at segment end (Issue 8) | Change `progress = (episode_step - seg_start) / max(1, seg_end - seg_start - 1)` in `LiberoTraceDataset.__getitem__` |

---

## 15. Inference: action sampling + completion prediction + trace generation

The model exposes three public endpoints for inference; the deployment loop typically
calls them at three different cadences.

### 15.1 Model endpoints

[pi0_trace_vla.py — `sample_actions_and_completion`](pace/openpi/src/openpi/models/pi0_trace_vla.py),
[`predict_completion`](pace/openpi/src/openpi/models/pi0_trace_vla.py),
[`sample_trace`](pace/openpi/src/openpi/models/pi0_trace_vla.py)

| Method | Purpose | Cost |
|---|---|---|
| `sample_actions(rng, obs)` | Action chunk only (legacy, unchanged). | 1× SigLIP encode + 1× Gemma prefill + `num_steps` denoise steps |
| `sample_actions_and_completion(rng, obs)` | Action chunk **and** completion progress, in one shared prefix prefill. **Recommended for the execution loop.** | 1× SigLIP encode + 1× Gemma prefill + `num_steps` denoise steps + ~670K-param completion head (negligible) |
| `predict_completion(rng, obs)` | Completion progress only (no action denoising). For querying completion at a *lower* cadence than action sampling. | 1× SigLIP encode + 1× Gemma prefill + completion head. **No** denoise steps. |
| `sample_trace(rng, obs)` | Generates an `(N, 2)` normalized image-space trace via the planning forward. | 1× SigLIP encode (clean image) + 1× Gemma prefill + `num_steps` denoise steps over the trace stream |

**Why the combined endpoint exists.** The completion head consumes the same execution-mode
prefix output (`prefix_out`) that the action head's prefill produces. Calling
`sample_actions` and then `predict_completion` would redo the SigLIP image encode and
the full Gemma prefill — by far the most expensive parts of the forward pass — for the
*same* observation. `sample_actions_and_completion` runs the prefill once and feeds
`prefix_out` to both heads. Verified in the sandbox: standalone `predict_completion`
returns identical progress to the combined endpoint (max diff `0.0`), and the
combined endpoint produces identical actions to `sample_actions` given the same
noise (max diff `0.0`).

### 15.2 `TraceVLAPolicy`

[pace/openpi/src/openpi/policies/policy.py — `TraceVLAPolicy`](pace/openpi/src/openpi/policies/policy.py)

A minimal `BasePolicy` wrapper that JITs the three model endpoints and threads the
data transforms (the same `LeRobotTraceVLADataConfig` transforms used at training
time). Its public surface:

```python
out = policy.infer(obs)                     # {"actions", "progress", "state", ...}
trace = policy.sample_trace(obs, num_steps=10)   # (N, 2) numpy in [0, 1]^2
progress = policy.predict_completion(obs)        # scalar in [0, 1]
```

`obs` is a dict shaped exactly like the dataset's per-sample dict:

- `observation/image` (uint8 H×W×3)
- `observation/wrist_image` (uint8 H×W×3)
- `observation/overlay_image` (uint8 H×W×3) — only required for `infer` (execution
  mode); `sample_trace` ignores it. The caller is responsible for rendering the
  overlay from a recent `sample_trace` output via `trace_utils.draw_polyline_overlay`.
- `observation/state` (float32, 8-dim raw LIBERO state)
- `atomic_token` (int in `{0..4}`), `skill_name`, `skill_text`,
  `semantic_target_xy` (float32 `[2]` in `[0, 1]^2`),
  `current_ee_xy` (float32 `[2]` in `[0, 1]^2`),
- `prompt` (the original LIBERO task instruction; only used as a fallback if
  `skill_name` and `skill_text` are missing).

Output dict from `infer()`:

- `actions`: `(action_horizon, 7)` after `LiberoTraceOutputs` trims to the LIBERO
  action dim and norm-stats unnormalize.
- `progress`: scalar in `[0, 1]`.
- `state`: pass-through.
- `policy_timing`: `{"infer_ms": <wall clock ms for the JAX call>}`.

### 15.3 Recommended deployment loop

```
loop:
    obs = read_robot_obs()
    obs["semantic_target_xy"] = high_level_vlm(...)  # external VLM
    obs["atomic_token"], obs["skill_name"], obs["skill_text"] = current_skill
    obs["current_ee_xy"] = forward_project_ee(robot_state)

    # Re-plan trace at a low rate (or when progress crosses a threshold).
    if step % F_plan == 0 or progress > 0.85:
        trace = policy.sample_trace(obs)
        trace_overlay_image = trace_utils.draw_polyline_overlay(obs["observation/image"], trace, ...)
    obs["observation/overlay_image"] = trace_overlay_image

    # Action + completion at the high control-loop rate.
    out = policy.infer(obs)
    execute(out["actions"])
    progress = out["progress"]

    if progress > completion_threshold:
        current_skill = next_skill_from_high_level_vlm(...)
```

`F_plan` and `completion_threshold` are deployment tunables. Caching the rendered
overlay between re-plans is cheap; the action expert is trained (via the anchor-age
augmentation) to handle stale plans up to `h_train_max ≈ 15` control steps old.

### 15.4 Factory

[pace/openpi/src/openpi/policies/policy_config.py — `create_trained_trace_vla_policy`](pace/openpi/src/openpi/policies/policy_config.py)

```python
from openpi.policies.policy_config import create_trained_trace_vla_policy
from openpi.training import config as _config

train_config = _config.get_config("trace_vla")  # or "trace_vla_lora"
policy = create_trained_trace_vla_policy(
    train_config,
    checkpoint_dir="<path/to/checkpoint>",
)
```

The factory:

- loads weights from the checkpoint's `params/` (JAX path; PyTorch checkpoints are
  not supported for TraceVLA),
- loads norm stats from the checkpoint's `assets/<asset_id>` so the action head sees
  the same scale it was trained on,
- builds the same input transform stack as `LeRobotTraceVLADataConfig.create()`
  (`LiberoTraceInputs` → `Normalize` → `TraceResizeImages` → `TraceTokenizePrompt` →
  `PadStatesAndActions`) plus the corresponding output transforms (`LiberoTraceOutputs`
  + `Unnormalize`),
- returns a fully-wired `TraceVLAPolicy` instance.

---

*End of document.*
