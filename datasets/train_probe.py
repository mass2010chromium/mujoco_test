import json
import os
from pathlib import Path
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".25"

import jax, jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform

import flax.nnx as nnx
import optax

from probe_network import ProbeNetwork, LinearProbeNetwork

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="dataset name (ex. libero_10)")
args = parser.parse_args()

DATASET = args.dataset

model = ProbeNetwork(nnx.Rngs(0))
model.train()


schedule = optax.warmup_cosine_decay_schedule(
  init_value=0.0,
  peak_value=1e-4,
  warmup_steps=50,
  decay_steps=20000,
  end_value=1e-5,
)
optimizer = nnx.Optimizer(model, optax.adam(learning_rate=schedule), wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, intermediates, targets):
    def compute_loss(model):
        predictions = model(intermediates)
        return jnp.mean((predictions - targets)**2)

    loss, grads = nnx.value_and_grad(compute_loss)(model)
    optimizer.update(grads)  # nnx.jit allows in place updates

    return loss

rngs = nnx.Rngs(0)
file_dir = Path(__file__).parent
intermediates = jnp.load(str(file_dir / f"{DATASET}_intermediates.npy"))
_targets = np.load(str(file_dir / f"{DATASET}_transforms.npy"))
tf = RigidTransform.from_exp_coords(_targets)
targets = jnp.array(tf.translation)
print(intermediates.shape)
print(targets.shape)

key = jax.random.key(0)
batch_size = 200
i = 0
while i < 20000:
    key, subkey = jax.random.split(key)
    indices = jax.random.permutation(subkey, len(intermediates))

    for j in range(0, len(intermediates), batch_size):
        intermediates_batch = intermediates[indices[j:j+batch_size]]
        targets_batch = targets[indices[j:j+batch_size]]
        print(intermediates_batch.shape)
        print(targets_batch.shape)
        loss = train_step(model, optimizer, intermediates_batch, targets_batch)
        i += 1
        if i % 100 == 0:
            print(f'step {i}')
            print(f'{loss = }')


import orbax.checkpoint as ocp
_, state = nnx.split(model)
checkpointer = ocp.StandardCheckpointer()
ckpt_dir = ocp.test_utils.erase_and_create_empty(os.path.abspath(str(file_dir / 'checkpoints')))
checkpointer.save(ckpt_dir / 'state', state)

# I am too lazy to fix the error on exit....
import time
time.sleep(5)
