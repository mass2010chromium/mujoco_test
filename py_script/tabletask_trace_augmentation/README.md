# Tabletask Skill Augmentation Pipeline

This folder contains a skill-annotation pipeline for the
[`n5zhong/table_tasks`](https://huggingface.co/datasets/n5zhong/table_tasks) LeRobot
dataset. It mirrors the structure of `libero_trace_augmentation/` but is adapted to
the real-robot tabletop dataset where camera frames are stored as `.mp4` files
(LeRobot `dtype: "video"`) rather than inline parquet image bytes.

## Dataset choice

The pipeline operates on the cached snapshot of `n5zhong/table_tasks`.

Recommended storage layout:

- Source dataset: `~/.cache/huggingface/hub/datasets--n5zhong--table_tasks/...`
- Run outputs: `tabletask_trace_augmentation/skill-runs/<run_name>/`

## What the scripts do

- `annotate_tabletask_skills.py`
  - reads the pre-encoded per-episode mp4 from the dataset's `videos/chunk-*/{image,wrist_image}/` folders
  - re-renders each episode into a step-overlaid mp4 (the "annotation video") at 10 FPS
  - queries `google/gemini-3.1-pro-preview` through OpenRouter for one episode at a time
  - writes one JSON shard per episode under `episode_shards/`
  - on success: also writes per-segment transition-scene PNGs under `transition_scenes/episode_XXXXXX/`
    and a per-episode visualization mp4 under `videos/episode_XXXXXX.mp4`
  - on failure: writes a JSON record under `errors/episode_XXXXXX.error.json` and does NOT produce any
    visualization video for that episode, so the combined video naturally skips it
  - supports `--start-episode`, `--end-episode`, and is resumable via `--skip-existing`
  - at the end of the run, rebuilds `videos/combined.mp4` from whatever `videos/episode_*.mp4` files exist
- `combine_skill_annotations.py`
  - merges one or more shard directories
  - writes:
    - a canonical skill annotation JSON
    - a training-oriented cot-style skill annotation JSON
- `validate_skill_annotations.py`
  - validates either partial shard files or final combined JSON files
  - checks that segment skills are valid and consistent with the numbered skill plan, and that the
    first skill is not a forbidden type

## Skill vocabulary

The tabletop dataset only uses pick-and-place skills, so the allowed skill set is:

```
PICKUP_FROM(object1, object2)
PLACE_ON(object1, object2)
PLACE_IN(object1, object2)
```

`PLACE_ON` and `PLACE_IN` are forbidden as the first skill in a plan.

## Canonical annotation format

The canonical combined JSON stores one skill execution segment per chunk:

```json
{
  "0": {
    "episode_index": 0,
    "task_index": 0,
    "instruction": "move the basket onto the blue plate and put the green bell pepper in the basket",
    "num_steps": 371,
    "fps": 10,
    "plan": "1. PICKUP_FROM(basket, table) 2. PLACE_ON(basket, blue plate) 3. PICKUP_FROM(green bell pepper, table) 4. PLACE_IN(green bell pepper, basket)",
    "segments": [
      { "start_step": 0, "end_step": 110, "skill": "PICKUP_FROM(basket, table)" },
      { "start_step": 110, "end_step": 165, "skill": "PLACE_ON(basket, blue plate)" },
      { "start_step": 165, "end_step": 265, "skill": "PICKUP_FROM(green bell pepper, table)" },
      { "start_step": 265, "end_step": 371, "skill": "PLACE_IN(green bell pepper, basket)" }
    ]
  }
}
```

## Training-oriented annotation format

The training JSON expands each skill into:

- a short boundary segment at the start of the skill
- an action segment for the rest of the skill

This is controlled by `--boundary-window` in `combine_skill_annotations.py` (default `10`).

## Environment

Run inside an environment with `pyarrow`, `imageio`, `imageio_ffmpeg`, and `Pillow`. The
`mujoco_playground` uv env at `~/workspace_nz/mujoco_test/mujoco_playground/.venv` works.
The pipeline reads frames directly from the dataset's pre-encoded mp4 files via
`imageio` (which bundles its own ffmpeg) and does NOT depend on the LeRobot
torchcodec/ffmpeg video backend.

The annotation step also needs:

- `OPENROUTER_API_KEY` for the OpenRouter request

## Step 1: annotate episodes

Annotate a single range:

```bash
export OPENROUTER_API_KEY=...

python annotate_tabletask_skills.py \
  --output-dir skill-runs/run_0_100 \
  --start-episode 0 \
  --end-episode 100 \
  --skip-existing
```

Run several disjoint ranges in parallel by varying `--output-dir`, `--start-episode`, and `--end-episode`:

```bash
python annotate_tabletask_skills.py \
  --output-dir skill-runs/run_0_150 \
  --start-episode 0 \
  --end-episode 150 \
  --skip-existing
```

```bash
python annotate_tabletask_skills.py \
  --output-dir skill-runs/run_150_299 \
  --start-episode 150 \
  --end-episode 299 \
  --skip-existing
```

Resume after a crash or retry failures with the same command:

```bash
python annotate_tabletask_skills.py \
  --output-dir skill-runs/run_0_150 \
  --start-episode 0 \
  --end-episode 150 \
  --skip-existing
```

Because each completed episode already has its own shard, re-running the same range with
`--skip-existing` is safe: previously-successful episodes are skipped, and previously-failed
episodes (which never wrote a shard, only an error file) are re-attempted. After the run,
`videos/combined.mp4` is rebuilt to include any newly-successful episodes.

Important output locations inside each run directory:

- `episode_shards/episode_000123.json` — one shard per successful episode
- `errors/episode_000123.error.json` — one error file per failed episode (deleted on later success)
- `transition_scenes/episode_000123/transition_000_step_000000.png` — clean per-boundary frames
- `videos/episode_000123.mp4` — visualization mp4 with skill-name overlays for the human
- `videos/combined.mp4` — rebuilt at the end of every run, concatenating all `videos/episode_*.mp4`
- `annotation_videos/episode_000123.mp4` — only kept with `--keep-annotation-videos`
- `run_manifest.json`

If you only want the shards and not any auxiliary visualizations, pass:

```bash
python annotate_tabletask_skills.py \
  --output-dir skill-runs/run_0_150 \
  --start-episode 0 \
  --end-episode 150 \
  --disable-saving-transition-scene \
  --disable-saving-visualization-video
```

## Step 2: combine shard outputs

Combine one or more run directories:

```bash
python combine_skill_annotations.py \
  skill-runs/run_0_150/episode_shards \
  skill-runs/run_150_299/episode_shards \
  --canonical-output skill-runs/skill_annotations.json \
  --training-output skill-runs/cot_skill.json
```

If you have not finished all episodes yet, you can still combine partial shards with `--allow-missing`.

## Step 3: validate

Validate partial shard outputs:

```bash
python validate_skill_annotations.py \
  skill-runs/run_0_150/episode_shards
```

Validate the final combined files:

```bash
python validate_skill_annotations.py \
  skill-runs/skill_annotations.json \
  skill-runs/cot_skill.json
```

The validator checks (per episode):

- the numbered plan parses and matches the compressed segment skill sequence
- every segment skill is one of `{PICKUP_FROM, PLACE_ON, PLACE_IN}` and has the right arg count
- segments are contiguous starting from step 0 and end at `num_steps` when present
- the first planned skill is not `PLACE_ON` or `PLACE_IN`
- text fields in the training-oriented JSON do not mention skills outside the plan

## Visualization video conventions

For every successful episode, the pipeline writes a per-episode mp4 to
`<run>/videos/episode_XXXXXX.mp4`. Each frame is overlaid with:

- top, line 1: `step XXX / YYY` (identical position and size to what the annotator saw)
- top, line 2: `skill: <skill expression for this frame>` (yellow)
- bottom: `episode XXXXXX / YYYYYY`

After every run (and during the auxiliary backfill on `--skip-existing` re-runs), the pipeline
also rebuilds `<run>/videos/combined.mp4` by concatenating every existing
`<run>/videos/episode_*.mp4` in episode order via ffmpeg's `concat` demuxer (no re-encode). Failed
episodes never produce a visualization mp4, so they are naturally skipped in the combined video.
A later `--skip-existing` re-run that successfully annotates previously-failed episodes will
produce their per-episode videos and refresh the combined video to include them.

## Prompting notes

The annotation prompt is the same shape as the libero pipeline:

- render one episode into a 10 FPS step-overlaid video
- ask Gemini to segment the episode into ordered steps using only the restricted
  pick-and-place skill vocabulary
- recover segment boundaries from the returned step endpoints

The prompt enforces the same segmentation invariants as the libero pipeline (first segment
starts at step 0, last `end_step` equals `num_steps`, contiguous segments, no commas inside
object names, plan matches segment sequence, first skill cannot be `PLACE_ON`/`PLACE_IN`).
