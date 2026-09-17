# Release Notes

What changed in the latest release, and the state of a few model paths.

## Model status

- **NetVAE** is in **maintenance mode** — the team's next architecture is a mixture-of-experts model. Existing NetVAE models and the `embkit model train-netvae` command continue to work; no new features are planned for this path.
- **`BaseVAE`** is the concrete model recommended for new work (see the [API overview](api/models/index.md)).

## Refactors in this release

The current tree reflects a refactor across model verification, masked-layer constraint enforcement, and training/serialization. Highlights:

- `embkit model verify` supports `--json`, `--ci`, `--fail-on-unhealthy`, and strict identity checks (`--strict`, `--expected-feature-count`, `--expected-latent-dim`, `--expected-features-file`).
- Mask clamping is enforced at three points (forward, post-step, pre-save) to keep serialized pathway-constrained artifacts constraint-consistent.
- `VAEOutput` / `EncoderOutput` named tuples make the `(recon, mu, logvar, z)` and `(mu, logvar, z)` outputs explicit.
- `factory.save(...)` clamps constrained weights before writing; `model.history` is normalized to a dict for stable round-trips.

## API notes

- The top-level `embkit` module exposes `get_device`, `dataframe_loader`, `dataframe_tensor`, `tensor_dataframe`, and `dataframe_dataset`.
- `embkit.optimize` provides `fit`, `fit_vae`, and `fit_net_vae` (see [Training](api/optimize.md)).
- `embkit.estimator.VAEEstimator` provides a scikit-learn style `fit`/`score` wrapper.
