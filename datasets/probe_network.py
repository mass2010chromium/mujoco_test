import einops
import flax.nnx as nnx
import jax

class ProbeNetwork(nnx.Module):
    def __init__(self, rngs):
        self.in_proj = nnx.Linear(in_features=2048*20, out_features=2048, rngs=rngs)
        self.inner = nnx.Linear(in_features=2048, out_features=2048, rngs=rngs)
        self.out_proj = nnx.Linear(in_features=2048, out_features=3, rngs=rngs)

    def __call__(self, x):
        x = einops.rearrange(x, "b l t d -> b (l t d)")
        x = self.in_proj(x)
        x = jax.nn.relu(x)
        x = self.inner(x)
        x = jax.nn.relu(x)
        x = self.out_proj(x)
        return x

class LinearProbeNetwork(nnx.Module):
    def __init__(self, rngs):
        self.linear = nnx.Linear(in_features=2048*20, out_features=3, rngs=rngs)

    def __call__(self, x):
        x = einops.rearrange(x, "b l t d -> b (l t d)")
        return self.linear(x)
