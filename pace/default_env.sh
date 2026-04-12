export MUJOCO_GL=egl
export JAX_TRACEBACK_FILTERING=off

# Leave W&B mode configurable per job. If not set elsewhere, keep the previous
# default behavior of running offline.
if [[ -z "${WANDB_MODE:-}" ]]; then
  export WANDB_MODE=offline
fi
