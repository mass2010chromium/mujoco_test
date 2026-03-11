import json
import os
from pathlib import Path
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".05"

import jax
jax.default_device = jax.devices("cpu")[0]
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform

import libero

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="dataset name (ex. libero_10)")
args = parser.parse_args()

DATASET = args.dataset

file_dir = Path(__file__).parent
data_dir = file_dir / 'outputs_and_transforms' / DATASET
meta_file = data_dir / "meta.json"
meta = json.load(open(meta_file, 'r'))

assert meta['dataset'] == DATASET
REPEATS = meta['repeats']

libero_root = Path(libero.__file__).parent
info_file = libero_root / 'libero' / 'bddl_files' / DATASET / 'tasks_info.txt'
target_files = []
with open(info_file, 'r') as f:
    target_files = [s.strip() for s in f.readlines()]

def get_sample(task_name, instance):
    target = np.load(data_dir / f"{task_name}_{instance}_transform.npy")
    intermediate = jnp.load(data_dir / f"{task_name}_{instance}_intermediate.npy")
    # time, layer, batch, token, dimension
    return intermediate[:, :, 0, -1, :], target

n_samples = len(target_files)*REPEATS
print(f"{n_samples} files to read.")
from tqdm import tqdm
all_samples = []
all_rots = []
with tqdm(total=n_samples) as pbar:
    for target_file in target_files:
        task_name = target_file.rsplit('/', 1)[1].split('.', 1)[0]
        for i in range(REPEATS):
            try:
                last_embedding, out = get_sample(task_name, i)
                all_samples.append(last_embedding)
                all_rots.append(out)
                pbar.update(1)
            except Exception as e:
                #print("Cannot read file, skipping")
                raise e
                pbar.update(REPEATS - i)
                break

print(f"Read {len(all_rots)} samples.")
jnp.save(str(file_dir / f"{DATASET}_intermediates.npy"), jnp.concat(all_samples, axis=0))
np.save(str(file_dir / f"{DATASET}_transforms.npy"), np.concatenate(all_rots, axis=0))
