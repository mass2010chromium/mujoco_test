#!/usr/bin/env bash
# provision.sh — Bootstrap mujoco_test on a new Azure H100 VM
#
# Run once after VM provisioning. Clones the repo and venv to the
# local temp SSD (/mnt), where POSIX symlinks work. Datasets stay
# on the Azure share — never cloned locally.
#
# Usage: bash provision.sh [--datasets-path /mnt/mujoco-data/gatech/workspace/mujoco_test/datasets]

set -euo pipefail

WORKSPACE="/mnt/workspace"
REPO_URL="https://github.com/mass2010chromium/mujoco_test"
REPO_DIR="$WORKSPACE/mujoco_test"

echo "==> Provisioning mujoco_test on local SSD ($WORKSPACE)"

# 1. Create workspace on ephemeral local disk (survives session, not reboot)
sudo mkdir -p "$WORKSPACE"
sudo chown "$(whoami):$(whoami)" "$WORKSPACE"

# 2. Clone repo locally — NOT to the Azure share
if [ -d "$REPO_DIR/.git" ]; then
  echo "==> Repo already exists at $REPO_DIR, skipping clone"
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# 3. Link user folder with mount, I guess
# Really the *entire* user folder should be linked with the mount
# but I CBA to fix this now
# Also, the local .cache is living on ssd (living in azureuser actually)
# Is that fine?
cd ~
#ln -sT "$WORKSPACE" workspace
REPO_DIR="$(pwd)/workspace/mujoco_test"
cd "$REPO_DIR"
source mujoco_playground/.venv/bin/activate
cd "$REPO_DIR/pace/openpi"
"$REPO_DIR/sync/sync" checkpoints/
"$REPO_DIR/sync/sync" data/

cd ~/
mkdir -p .cache
cd .cache
"$REPO_DIR/sync/sync" huggingface/
"$REPO_DIR/sync/sync" openpi

cd "$REPO_DIR/pace/openpi/data"
# Likely to fail... but maybe azure being slow will actually save us here
huggingface-cli download yilin-wu/libero-100 --repo-type dataset

echo ""
echo "==> Done. Activate with:"
echo "    source $REPO_DIR/mujoco_playground/.venv/bin/activate"
echo ""
echo "NOTE: /mnt is ephemeral on Azure VMs. Re-run provision.sh after reboot."

