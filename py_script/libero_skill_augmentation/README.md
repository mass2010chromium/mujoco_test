# Libero Skill Augmentation Pipeline

This folder contains a clean pipeline for building a skill-annotated LIBERO-100 dataset for Pi05 / reasoning-VLA training.

## Dataset choice

Use the existing cached `yilin-wu/libero-100` LeRobot dataset as the base dataset.

Why this is the cleanest choice:

- It is already in the LeRobot layout that your `openpi` training code expects.
- Your current dataset loader in `scratch/mujoco_test/pace/openpi/src/openpi/policies/libero_reason_dataset.py` already resolves `yilin-wu/libero-100` from the Hugging Face cache.
- The paper pipeline also starts from a LeRobot-formatted LIBERO dataset and compiles each episode into a 10 FPS video before querying Gemini.
- Re-downloading the original LIBERO HDF5 dataset and converting it back into LeRobot would add extra conversion risk without helping the skill annotation step.

Recommended storage layout:

- Source dataset: keep using the cached snapshot under `~/.cache/huggingface/hub/datasets--yilin-wu--libero-100/...`
- Intermediate run outputs: `scratch/mujoco_test/data/libero-100-skill-runs/<run_name>/`
- Final packaged dataset: `scratch/mujoco_test/data/libero-100-skill/`

## What the scripts do

- `inspect_libero_dataset.py`
  - prints episode counts, task counts, duplicates, length statistics
  - optionally exports an episode video with step overlays
- `annotate_libero_skills.py`
  - renders one episode at a time into an MP4 with step overlays
  - saves clean agent-view frames for the initial scene and every annotated skill-transition boundary by default
  - queries `google/gemini-3.1-pro-preview` through OpenRouter
  - writes one JSON shard per episode
  - supports `--start-episode` and `--end-episode`
  - is resumable because each episode is saved independently and atomically
- `combine_skill_annotations.py`
  - merges one or more shard directories
  - writes:
    - a canonical skill annotation JSON
    - a training-oriented cot-style skill annotation JSON
- `validate_skill_annotations.py`
  - validates either partial shard files or final combined JSON files
  - checks that segment skills are valid and consistent with the numbered skill plan
- `validate_skill_plans.py`
  - validates either partial shard files or final combined JSON files with the symbolic scene-graph verifier
  - reads the saved `transition_scenes/` RGB frames produced by `annotate_libero_skills.py`
  - verifies the full skill plan first, then verifies each skill transition at the annotated boundary frames
  - supports `--plan-only` and `--start/--end` for partial-range validation runs
  - writes `validation_results.json` listing every failed episode and the failure stage/details
- `package_lerobot_skill_dataset.py`
  - reuses the existing LeRobot `data/` and `meta/`
  - writes the new annotation files into a final dataset root
  - supports `symlink` for local work and `copy` for upload
- `upload_to_hub.py`
  - uploads the packaged dataset root to a Hugging Face dataset repo

## Canonical annotation format

The canonical combined JSON stores one skill execution segment per chunk:

```json
{
  "0": {
    "episode_index": 0,
    "task_index": 0,
    "instruction": "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "num_steps": 292,
    "fps": 10,
    "plan": "1. PICKUP_FROM(white mug, table) 2. PLACE_ON(white mug, left plate)",
    "segments": [
      {
        "start_step": 0,
        "end_step": 73,
        "skill": "PICKUP_FROM(white mug, table)"
      },
      {
        "start_step": 73,
        "end_step": 292,
        "skill": "PLACE_ON(white mug, left plate)"
      }
    ]
  }
}
```

This is the simplest representation and is the one you should think of as the ground-truth skill annotation.

## Training-oriented annotation format

The combined training JSON expands each skill into:

- a short boundary segment at the start of the skill
- an action segment for the rest of the skill

This is controlled by `--boundary-window` in `combine_skill_annotations.py`. The default is `10` frames because the existing libero reasoning annotations usually use about a 10-step reasoning window at each update boundary.

This training JSON is what you should point your current `LiberoSkillReasonDataset` at.

## Environment

Run these scripts inside the same Python environment you use for `scratch/mujoco_test/pace/openpi`, because they expect the `lerobot` dependencies used by training.

You also need:

- `OPENROUTER_API_KEY` for annotation
- `HF_TOKEN` for the upload helper if you use it

## Step 1: inspect the dataset

Print dataset stats:

```bash
python scratch/mujoco_test/py_script/libero_skill_augmentation/inspect_libero_dataset.py
```

Export one episode as a sanity-check video:

```bash
python scratch/mujoco_test/py_script/libero_skill_augmentation/inspect_libero_dataset.py \
  --episode-index 439 \
  --export-video scratch/mujoco_test/data/libero_debug_episode_439.mp4
```

What you should expect from the cached `yilin-wu/libero-100` snapshot on this machine:

- `meta/info.json` says `4338` episodes and `676070` frames
- `meta/tasks.jsonl` has `83` task rows
- only `82` task strings are unique because one task text appears twice

## Step 2: annotate episodes

Annotate a single range:

```bash
export OPENROUTER_API_KEY=...

python annotate_libero_skills.py \
  --output-dir libero-100-skill-runs/run_a \
  --start-episode 0 \
  --end-episode 200 \
  --skip-existing \
  --keep-videos
```

Run several disjoint ranges in parallel:

```bash
python annotate_libero_skills.py \
  --output-dir libero-100-skill-runs/run_a \
  --start-episode 0 \
  --end-episode 1000 \
  --skip-existing
```

```bash
python annotate_libero_skills.py \
  --output-dir libero-100-skill-runs/run_b \
  --start-episode 1000 \
  --end-episode 2000 \
  --skip-existing
```

Why episode ranges are better than task ranges:

- one annotation request corresponds to one episode
- resuming is trivial
- shard files are keyed directly by episode index
- task sizes are uneven, so task ranges are less balanced for parallel work

Resume after a crash:

- If a run over `100:200` dies after writing episode shards through `129`, just rerun:

```bash
python annotate_libero_skills.py \
  --output-dir libero-100-skill-runs/run_a \
  --start-episode 100 \
  --end-episode 200 \
  --skip-existing
```

Because each completed episode already has its own shard file, rerunning the same range is safe.

Important output locations inside each run:

- `episode_shards/episode_000123.json`
- `errors/episode_000123.error.json`
- `run_manifest.json`
- `transition_scenes/episode_000123/transition_000_step_000000.png`
- optionally `videos/episode_000123.mp4`

If you do not want to save those clean transition-scene frames, pass:

```bash
python scratch/mujoco_test/py_script/libero_skill_augmentation/annotate_libero_skills.py \
  --output-dir scratch/mujoco_test/data/libero-100-skill-runs/run_a \
  --start-episode 0 \
  --end-episode 200 \
  --disable-saving-transition-scene
```

## Step 3: combine shard outputs

Combine one or more run directories:

```bash
python combine_skill_annotations.py \
  libero-100-skill-runs/run_a/episode_shards \
  libero-100-skill-runs/run_b/episode_shards \
  --canonical-output libero-100-skill-runs/skill_annotations.json \
  --training-output libero-100-skill-runs/cot_skill.json
```

If you have not finished all episodes yet, you can still combine partial shards with `--allow-missing`.

Validate partial shard outputs:

```bash
python validate_skill_annotations.py \
  libero-100-skill-runs/run_a/episode_shards
```

Validate the final combined files:

```bash
python validate_skill_annotations.py \
  libero-100-skill-runs/skill_annotations.json \
  libero-100-skill-runs/cot_skill.json
```

Run symbolic plan + transition validation on shard outputs:

```bash
python validate_skill_plans.py \
  libero-100-skill-runs/run_a/episode_shards
```

Only validate plans for a bounded episode range:

```bash
python validate_skill_plans.py \
  libero-100-skill-runs/run_a/episode_shards \
  --plan-only \
  --start 0 \
  --end 200
```

Run symbolic plan + transition validation on the final combined files:

```bash
python validate_skill_plans.py \
  libero-100-skill-runs/skill_annotations.json
```

If a combined file draws episodes from multiple annotation runs and auto-discovery is ambiguous, pass one or more explicit transition-scene roots:

```bash
python validate_skill_plans.py \
  libero-100-skill-runs/skill_annotations.json \
  --transition-scene-root libero-100-skill-runs/run_a/transition_scenes \
  --transition-scene-root libero-100-skill-runs/run_b/transition_scenes
```

The symbolic validator can also read the training-style `cot_skill.json`; it reconstructs the canonical skill boundaries by merging adjacent same-skill segments before running plan and transition checks.

## Step 4: package the final LeRobot dataset root

For local experimentation, use symlinks:

```bash
python scratch/mujoco_test/py_script/libero_skill_augmentation/package_lerobot_skill_dataset.py \
  --canonical-annotation scratch/mujoco_test/data/libero-100-skill-runs/skill_annotations.json \
  --training-annotation scratch/mujoco_test/data/libero-100-skill-runs/cot_skill.json \
  --output-root scratch/mujoco_test/data/libero-100-skill \
  --copy-mode symlink \
  --output-repo-id yourname/libero-100-skill
```

For Hugging Face upload, use copied files instead of symlinks:

```bash
python scratch/mujoco_test/py_script/libero_skill_augmentation/package_lerobot_skill_dataset.py \
  --canonical-annotation scratch/mujoco_test/data/libero-100-skill-runs/skill_annotations.json \
  --training-annotation scratch/mujoco_test/data/libero-100-skill-runs/cot_skill.json \
  --output-root scratch/mujoco_test/data/libero-100-skill-upload \
  --copy-mode copy \
  --output-repo-id yourname/libero-100-skill \
  --force
```

By default the packaged training annotation is named `cot_simple.json` so the final dataset root looks similar to `libero-100-r`.

## Step 5: use the packaged annotation for Pi05 training

Your current skill-training configs in `scratch/mujoco_test/pace/openpi/src/openpi/training/config.py` already point at:

- `REPO_ROOT/'data/libero-100/cot_skill_fixed.json'`
- `REPO_ROOT/'data/libero-100/cot_skill_fixed2.json'`

The easiest path is:

```bash
mkdir -p scratch/mujoco_test/pace/openpi/data/libero-100
ln -sfn /home/hice1/nzhong34/scratch/mujoco_test/data/libero-100-skill/cot_simple.json \
  /home/hice1/nzhong34/scratch/mujoco_test/pace/openpi/data/libero-100/cot_skill_fixed2.json
```

Then train with your existing config:

```bash
cd scratch/mujoco_test/pace/openpi
python scripts/train.py pi05_libero_skill_reason_lora_v2 --exp-name=pi05_libero_skill_reason_lora_v2
```

If you prefer, you can also point `reasoning_json_path` at the packaged annotation via config override or by editing the config directly.

## Step 6: upload the packaged dataset to Hugging Face

First authenticate:

```bash
huggingface-cli login
export HF_TOKEN=...
```

Then upload:

```bash
python scratch/mujoco_test/py_script/libero_skill_augmentation/upload_to_hub.py \
  --dataset-root scratch/mujoco_test/data/libero-100-skill-upload \
  --repo-id yourname/libero-100-skill
```

For very large uploads, use:

```bash
python scratch/mujoco_test/py_script/libero_skill_augmentation/upload_to_hub.py \
  --dataset-root scratch/mujoco_test/data/libero-100-skill-upload \
  --repo-id yourname/libero-100-skill \
  --use-large-folder-upload
```

## Prompting notes

The annotation prompt follows the same high-level idea as the paper:

- render one episode into a 10 FPS video
- ask Gemini to segment the episode into ordered steps
- recover segment boundaries from the returned step endpoints

The main difference is that this pipeline restricts the output to your exact skill language instead of free-form natural-language subtasks.

The scripts also add step-number overlays to the rendered video to make timestep boundaries easier for the model to report.

## References

- LIBERO datasets: https://libero-project.github.io/datasets
- "Do What You Say" project page: https://yilin-wu98.github.io/steering-reasoning-vla/
- Paper PDF: https://arxiv.org/pdf/2510.16281
- Existing LeRobot dataset: https://huggingface.co/datasets/yilin-wu/libero-100
- NVIDIA libero-r dataset tree: https://huggingface.co/datasets/nvidia/libero-r-datasets/tree/main/libero-100-r
- OpenRouter video inputs docs: https://openrouter.ai/docs/features/multimodal/video-inputs
- OpenRouter structured outputs docs: https://openrouter.ai/docs/features/structured-outputs
- Hugging Face Hub upload guide: https://huggingface.co/docs/huggingface_hub/guides/upload
