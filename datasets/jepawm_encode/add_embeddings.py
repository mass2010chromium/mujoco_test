import json
import os
from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent

import einops
import numpy as np
import torch

from openpi.policies.libero_trace_dataset import LiberoTraceDataset
from openpi.training import config as _config

from jepawm_encoder import load_jepawm

try:
    from compression import zstd
except:
    from backports import zstd

model = load_jepawm(model_name="jepa_wm_pusht")

data_config = _config.get_config('trace_vla_moe')
dataset = LiberoTraceDataset(data_config.data.base_config, data_config.model.action_horizon)

out_dir = SCRIPT_DIR / "dino_outputs"

def get_episode(episode_idx):
    start_idx = dataset.episode_starts[episode_idx]
    end_idx = dataset.episode_ends[episode_idx]
    return dataset.hf_dataset[start_idx:end_idx]['image']

def process_episode(episode, episode_idx):
    # IMPORTANT: input must be 0-255! Or else you will get absolute garbage out
    video_frames = torch.stack(episode).to('cuda:0') * 255
    with torch.no_grad():
        all_encodings = model.encode(einops.rearrange(video_frames, "t c h w -> 1 t c h w")).cpu().numpy()
    print("Type:", all_encodings.dtype)
    print("Byte size:", len(all_encodings.tobytes(order='C')))
    with zstd.open(f"{out_dir}/{episode_idx}.zstd", "wb", level=20) as zf:
        zf.write(all_encodings.tobytes(order='C'))
    with open(f"{out_dir}/{episode_idx}.meta.json", "w") as jf:
        json.dump([all_encodings.shape, str(all_encodings.dtype)], jf)

N = len(dataset.episode_starts)
start_index = 0
end_index = N
skip_present = False

if len(sys.argv) > 1:
    if sys.argv[1].strip() == "--fix":
        skip_present = True
    else:
        start_index = int(sys.argv[1])
if len(sys.argv) > 2:
    end_index = int(sys.argv[2])
#N = 1

from tqdm import tqdm
for i in tqdm(range(start_index, end_index)):
    # Dependencies:
    # pip install pygame shapely scikit-image nevergrad lpips seaborn clusterscope ruamel.yaml backports.zstd
    # also:
    # decord (but needs source install?)
    if skip_present:
        json_filename = f"{out_dir}/{i}.meta.json"
        skip = os.path.exists(f"{out_dir}/{i}.zstd") and os.path.exists(json_filename)
        try:
            with open(json_filename, "r") as jf:
                data = json.load(jf)
        except:
            print("Failed to open metadata file")
            skip = False
        if skip:
            print(f"Skipping episode: {i}", flush=True)
            continue 
    print(f"Processing episode: {i}", flush=True)
    episode = get_episode(i)
    process_episode(episode, i)
print(f"Done. [{start_index}, {end_index})")
