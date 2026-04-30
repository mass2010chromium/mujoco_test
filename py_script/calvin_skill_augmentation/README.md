# Calvin Skill Augmentation Pipeline

This folder contains the CALVIN counterpart to the existing LIBERO skill-annotation pipeline.

It is designed for the cached LeRobot-format dataset:

- repo id: `fywang/calvin-task-ABC-D-lerobot`
- default cache root: `~/.cache/huggingface/hub/datasets--fywang--calvin-task-ABC-D-lerobot/...`

The pipeline mirrors the LIBERO workflow:

- annotate episodes independently into per-episode JSON shards
- support disjoint episode ranges so multiple annotation jobs can run in parallel
- keep failures isolated in `errors/` so reruns with `--skip-existing` only retry missing episodes
- save boundary transition frames for later symbolic validation
- combine shards into a canonical annotation JSON plus a training-oriented CoT-style JSON
- validate syntax/rules deterministically
- validate plan feasibility and transition consistency with the scene-graph + PDDL verifier

## Files

- `common.py`
  - shared prompt text, CALVIN skill definitions, plan parsing, per-episode parquet loading, video rendering, transition-scene saving, and JSON aggregation helpers
- `annotate_calvin_skills.py`
  - renders one episode at a time into an MP4 with step overlays
  - queries `google/gemini-3.1-pro-preview` through OpenRouter
  - writes one shard JSON per episode
  - saves transition frames by default
  - stores `state` and `action` on every canonical skill segment
- `combine_skill_annotation.py`
  - merges shard directories into canonical and training-oriented combined JSON files
- `validate_skill_annotations.py`
  - checks syntax, plan/segment consistency, segment timing, deterministic skill-order constraints, and canonical `state` / `action` payloads
- `validate_skill_plans.py`
  - validates annotations with the scene-graph verifier using `py_script/pddl/calvin_domain.pddl`

## Dataset assumptions

Unlike the LIBERO pipeline, this CALVIN pipeline reads each episode directly from its per-episode parquet file instead of constructing a single global LeRobot dataset object.

That keeps annotation runs lighter and makes it easy to pull:

- top camera frames from `observation.images.top`
- wrist camera frames from `observation.images.wrist`
- `observation.state`
- `action`

## Canonical annotation format

Each canonical shard stores one segment per skill execution:

```json
{
  "episode_index": 0,
  "task_index": 0,
  "task_name": "lift_red_block_drawer",
  "instruction": "pick up the red block lying in the drawer",
  "num_steps": 43,
  "fps": 10,
  "plan": "1. OPEN(drawer) 2. PICKUP_FROM(red block, drawer)",
  "segments": [
    {
      "start_step": 0,
      "end_step": 17,
      "skill": "OPEN(drawer)",
      "state": [... 15 floats ...],
      "action": [... 7 floats ...]
    },
    {
      "start_step": 17,
      "end_step": 43,
      "skill": "PICKUP_FROM(red block, drawer)",
      "state": [... 15 floats ...],
      "action": [... 7 floats ...]
    }
  ]
}
```

Important detail:

- segments use half-open intervals `[start_step, end_step)`
- the saved `state` and `action` come from the last dataset row inside that segment, i.e. local step `end_step - 1`

## Training-oriented combined format

The combined training JSON expands each canonical skill into:

- a short boundary segment at the start of the skill
- an action segment for the remainder of the skill

This is controlled by `--boundary-window` in `combine_skill_annotation.py`.

## Environment

Use a Python environment that already has:

- `pyarrow`
- `imageio`
- `Pillow`
- `pddlsim`
- the verifier stack used under `py_script/vla_verify`

On this machine, the existing `mujoco_playground` environment works well:

```bash
/home/gatech/workspace_nz/mujoco_test/mujoco_playground/.venv/bin/python ...
```

You also need:

- `OPENROUTER_API_KEY` for annotation
- `OPENROUTER_API_KEY` again for `validate_skill_plans.py` if you use the `openrouter` backend

## Step 1: annotate episodes

Single range:

```bash
export OPENROUTER_API_KEY=...

python annotate_calvin_skills.py \
  --output-dir calvin-task-ABC-D-skill-runs/run_a \
  --start-episode 0 \
  --end-episode 200 \
  --skip-existing \
  --keep-videos
```

Run disjoint ranges in parallel:

```bash
python annotate_calvin_skills.py \
  --output-dir calvin-task-ABC-D-skill-runs/run_a \
  --start-episode 0 \
  --end-episode 5000 \
  --skip-existing
```

```bash
python annotate_calvin_skills.py \
  --output-dir calvin-task-ABC-D-skill-runs/run_b \
  --start-episode 5000 \
  --end-episode 10000 \
  --skip-existing
```

Useful options:

- `--image-key top`
- `--image-key wrist`
- `--disable-saving-transition-scene`
- `--disable-structured-output`
- `--overwrite-existing`

Resume after a crash by rerunning the same range with `--skip-existing`.

Important output locations inside each run:

- `episode_shards/episode_000123.json`
- `errors/episode_000123.error.json`
- `run_manifest.json`
- `transition_scenes/episode_000123/transition_000_step_000000.png`
- optionally `videos/episode_000123.mp4`

## Step 2: combine shards

Combine one or more run directories:

```bash
python combine_skill_annotation.py \
  calvin-task-ABC-D-skill-runs/run_a/episode_shards \
  calvin-task-ABC-D-skill-runs/run_b/episode_shards \
  --canonical-output calvin-task-ABC-D-skill-runs/skill_annotations.json \
  --training-output calvin-task-ABC-D-skill-runs/cot_skill.json
```

If some ranges are still incomplete, add `--allow-missing`.

## Step 3: deterministic validation

Validate shard outputs or combined JSON:

```bash
python validate_skill_annotations.py calvin-task-ABC-D-skill-runs/run_a/episode_shards
```

## Step 4: symbolic plan validation

Plan-only validation:

```bash
python validate_skill_plans.py calvin-task-ABC-D-skill-runs/skill_annotations.json \
  --plan-only
```

Full plan + transition validation:

```bash
python validate_skill_plans.py calvin-task-ABC-D-skill-runs/skill_annotations.json
```

Useful options:

- `--transition-scene-root <run_root_or_transition_scenes_dir>`
- `--start <episode_index>`
- `--end <episode_index>`
- `--plan-only`
- `--backend openrouter|ollama|r4b`

The validator writes `validation_results.json` by default.

## Notes about CALVIN-specific behavior

- Some CALVIN episodes start with an object already grasped.
  - That means a valid first skill can be `PLACE_ON(...)`, `PLACE_IN(...)`, or `TURN_OBJECT(object, direction)`.
- `TURN_OBJECT(object, direction)` is treated as one atomic skill in this pipeline.
  - Do not annotate `PICKUP_FROM(...)` immediately before it.
  - Do not annotate `PLACE_ON(...)` or `PLACE_IN(...)` immediately after it.
- The plan validator relies on the saved transition frames.
  - Do not delete `transition_scenes/` before running `validate_skill_plans.py`.
- Failed annotation episodes are intentionally omitted from shard outputs.
  - They only appear under `errors/` and can be retried safely with `--skip-existing`.
