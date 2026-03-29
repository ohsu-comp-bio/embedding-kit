# Factory

The `embkit.factory` module provides serialization and reconstruction helpers for model components.

It is used by CLI training commands to:

- define layer stacks from compact specs
- instantiate model modules
- save/load complete model state with architecture metadata

## Key APIs

- `factory.Layer` and `factory.LayerList` for layer configuration
- `factory.build(...)` for rebuilding modules from dict/list specs
- `factory.save(...)` and `factory.load(...)` for model serialization

## Example: train-vae style layer setup

This mirrors how `embkit model train-vae` turns `--encode-layers` / `--decode-layers` into `LayerList` objects.

```python
from embkit.factory import LayerList, save, load
from embkit.models.vae.vae import VAE

features = [f"gene_{i}" for i in range(1000)]
latent = 128

encode_layers = "400,200"
decode_layers = "200,400"
final_activation = "none"

enc_sizes = [int(v) for v in encode_layers.split(",")] + [latent]
dec_sizes = [int(v) for v in decode_layers.split(",")] + [len(features)]

encoder_layers = LayerList(enc_sizes)
decoder_layers = LayerList(dec_sizes, end_activation=final_activation)

vae = VAE(
    features=features,
    latent_dim=latent,
    encoder_layers=encoder_layers,
    decoder_layers=decoder_layers,
)

save(vae, "vae.model")
reloaded = load("vae.model")

print(type(reloaded).__name__)
print(reloaded.latent_dim)
```

## Example: `factory.build` patterns from unit tests

These patterns match the factory tests that verify list-based construction and activation lookup.

```python
from embkit import factory

seq = factory.build([
    factory.Linear(10, 20),
    factory.Linear(20, 1),
])

relu = factory.build("relu")

print(len(seq))
print(type(relu).__name__)
```

## Notes

- `factory.save` stores both `state_dict` and model description (`__model__`).
- `factory.load` reconstructs the model via `factory.build` and then loads weights.
- Unknown class names or unsupported build inputs raise explicit exceptions.
