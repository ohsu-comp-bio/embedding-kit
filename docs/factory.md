

## The Factory System

The factory system handles **model serialization** so that you can save a model and reload it on any machine without re-defining the architecture.

### How it works

Every registered model class implements two methods:

- `to_dict()` — returns a plain Python dict describing the architecture
- `from_dict(desc)` — reconstructs the model from that dict

When you call `factory.save(model, path)`, two things are written into the file:

1. The PyTorch `state_dict` (all weights)
2. A `__model__` key containing the architecture description dict

`factory.load(path)` reads the `__model__` key, calls `factory.build()` to reconstruct the architecture, then loads the weights.

```python
from embkit.factory import save, load

save(vae, "my_model.model")
vae2 = load("my_model.model")
```

### Layer and LayerList

Layers are described as **configuration objects**, not `nn.Module` instances, before training. This separation means the architecture can be serialized cleanly.

```python
from embkit.factory.layers import Layer, LayerList

# Layer(units, activation, op, batch_norm, bias)
encoder_layers = LayerList([
    Layer(512, activation="relu"),
    Layer(256, activation="relu"),
])
```

A `LayerList` is a list of `Layer` configs. When the `VAE` is constructed it calls `LayerList.build(input_dim, latent_dim)` to produce an actual `nn.Sequential`.

You can also pass a comma-separated string of sizes to the CLI; the `train-vae` command parses this into a `LayerList` automatically.

### Supported activations

`"relu"`, `"sigmoid"`, `"tanh"`, `"leaky_relu"`, `"none"`

