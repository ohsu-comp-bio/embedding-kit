# Optimize

Training loop functions for VAE and general models.

- **`fit_vae`** — train a VAE/reconstruction model (Beta-KL annealing, custom loss, schedules). The main entry point.
- **`fit`** — general supervised loop (input `x`, target `y`) for non-VAE models such as `FFNN`.
- **`fit_net_vae`** — retained API for pathway-constrained NetVAE with alternating constrained/unconstrained phases. Maintenance mode — the team's next architecture is a mixture-of-experts model.

## fit_vae

```python
from embkit.optimize import fit_vae
```

::: embkit.optimize.fit_vae
    options:
      show_source: false

## fit

```python
from embkit.optimize import fit
```

::: embkit.optimize.fit
    options:
      show_source: false

## fit_net_vae

```python
from embkit.optimize import fit_net_vae
```

> **Status note** — this path is retained for existing NetVAE users but not extended; see the [NetVAE page](models/net_vae.md).

::: embkit.optimize.fit_net_vae
    options:
      show_source: false
