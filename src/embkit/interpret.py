"""
Feature attribution for trained models.

Once an embedding model or a prediction head on top of it is trained, the next
question is usually *which input features drive it* — which genes push a sample
along a latent axis, or towards a predicted class. Integrated gradients answer
that with per-sample, per-feature attributions that sum (approximately) to the
difference between the model's output at the input and at a baseline.

- :func:`integrated_gradients` — Sundararajan et al., 2017.

Example — attributions for latent dimension 3 of a trained VAE::

    attr = integrated_gradients(lambda x: vae.encoder(x).mu[:, 3], x_batch)

Ported from UniVII (``univii.interpretability``).
"""
from typing import Callable, Optional

import torch


def integrated_gradients(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                         baseline: Optional[torch.Tensor] = None, steps: int = 32) -> torch.Tensor:
    """
    Integrated-gradients attribution of ``f`` with respect to its input.

    Gradients of ``f`` are averaged along the straight path from ``baseline``
    to ``x`` (``steps`` points, endpoints included) and multiplied by
    ``x - baseline``.

    Args:
        f: Function mapping a ``(batch, features)`` tensor to a scalar, or to
            one value per sample (summed internally; samples do not interact).
        x: Inputs to explain, ``(batch, features)``.
        baseline: Reference input of the same shape; defaults to zeros.
        steps: Number of interpolation points.

    Returns:
        Attributions with the same shape as ``x``.
    """
    if baseline is None:
        baseline = torch.zeros_like(x)
    alphas = torch.linspace(0.0, 1.0, steps=int(steps), device=x.device, dtype=x.dtype).view(-1, 1, 1)
    xs = baseline.unsqueeze(0) + alphas * (x.unsqueeze(0) - baseline.unsqueeze(0))  # (S, B, F)
    grads = []
    for s in range(int(steps)):
        xi = xs[s].detach().clone().requires_grad_(True)
        y = f(xi)
        if y.dim() != 0:
            y = y.sum()
        grads.append(torch.autograd.grad(y, xi, retain_graph=False, create_graph=False)[0])
    return (x - baseline) * torch.stack(grads, dim=0).mean(dim=0)
