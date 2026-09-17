# Estimator

Scikit-learn style wrappers that expose the core models through the familiar `fit` / `score` interface, handy for quick experiments and for use inside pipelines.

- **`VAEEstimator`** — a thin sklearn wrapper around `BaseVAE` that builds, trains, and stores the latent model so you can `.score()` reconstruction quality or pull `.model` for custom use.

See the [VAE Estimator page](vae_estimator.md) for the full reference.
