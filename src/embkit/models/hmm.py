"""
A small differentiable hidden Markov model for segmenting ordered data.

Embeddings are often read along an ordering — samples along pseudotime, bins
along the genome, time points of a perturbation. When you expect runs of a few
discrete states along that ordering (e.g. progenitor -> transitional ->
mature), an HMM turns a noisy continuous signal into a clean segmentation.

- :class:`GaussianHMM` — states with diagonal-Gaussian emissions; exact
  inference in log space (forward algorithm, forward-backward posteriors,
  Viterbi), fitted by gradient ascent.
- :func:`segment_by_order` — sort samples by an ordering, fit an HMM, and
  return a state per sample in the original order.

Ported from UniVII (``univii.hmm``).
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import factory

_LOG_2PI = float(np.log(2.0 * np.pi))


@factory.nn_module
class GaussianHMM(nn.Module):
    """
    HMM with ``n_states`` states and diagonal-Gaussian emissions in ``dim`` dimensions.

    A sequence is ``(T,)`` or ``(T, dim)``; a batch of equal-length sequences
    is ``(B, T, dim)`` and is processed together. Computations run in float64.

    Args:
        n_states: Number of hidden states.
        dim: Emission dimension.
        stickiness: Initial probability of staying in the same state; values
            near 1 favour long runs. All parameters are learned by :meth:`fit`.
    """

    def __init__(self, n_states: int, dim: int, stickiness: float = 0.9):
        super().__init__()
        if n_states < 1 or dim < 1 or not 0.0 < stickiness < 1.0:
            raise ValueError("need n_states >= 1, dim >= 1 and 0 < stickiness < 1")
        self.n_states, self.dim, self.stickiness = int(n_states), int(dim), float(stickiness)
        off = (1.0 - stickiness) / max(n_states - 1, 1)
        trans = torch.full((n_states, n_states), off) + torch.eye(n_states) * (stickiness - off)
        self.start_logits = nn.Parameter(torch.zeros(n_states, dtype=torch.float64))
        self.trans_logits = nn.Parameter(torch.log(trans.double()))
        self.means = nn.Parameter(torch.zeros(n_states, dim, dtype=torch.float64))
        self.logvars = nn.Parameter(torch.zeros(n_states, dim, dtype=torch.float64))

    # -- pieces ---------------------------------------------------------------
    def _x(self, x) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x, dtype=torch.float64)
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if x.ndim not in (2, 3) or x.shape[-1] != self.dim:
            raise ValueError(f"expected a (T, {self.dim}) sequence or a (B, T, {self.dim}) batch, got {tuple(x.shape)}")
        return x

    def emission_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """``log p(x_t | state k)``, shape ``(..., T, K)``."""
        diff = x.unsqueeze(-2) - self.means
        return (-0.5 * (diff ** 2 / self.logvars.exp() + self.logvars + _LOG_2PI)).sum(-1)

    def _forward(self, log_b: torch.Tensor) -> torch.Tensor:
        log_a = F.log_softmax(self.trans_logits, dim=1)
        alpha = [F.log_softmax(self.start_logits, dim=0) + log_b[..., 0, :]]
        for t in range(1, log_b.shape[-2]):
            alpha.append(torch.logsumexp(alpha[-1].unsqueeze(-1) + log_a, dim=-2) + log_b[..., t, :])
        return torch.stack(alpha, dim=-2)

    def _backward(self, log_b: torch.Tensor) -> torch.Tensor:
        log_a = F.log_softmax(self.trans_logits, dim=1)
        beta = [torch.zeros_like(log_b[..., 0, :])]
        for t in range(log_b.shape[-2] - 1, 0, -1):
            beta.append(torch.logsumexp(log_a + (log_b[..., t, :] + beta[-1]).unsqueeze(-2), dim=-1))
        return torch.stack(beta[::-1], dim=-2)

    # -- inference ------------------------------------------------------------
    def log_likelihood(self, x) -> torch.Tensor:
        """``log p(x_1..T)``: a scalar for one sequence, ``(B,)`` for a batch (differentiable)."""
        return torch.logsumexp(self._forward(self.emission_log_prob(self._x(x)))[..., -1, :], dim=-1)

    @torch.no_grad()
    def posterior(self, x) -> torch.Tensor:
        """State probabilities at every position given the whole sequence: ``(T, K)`` or ``(B, T, K)``."""
        log_b = self.emission_log_prob(self._x(x))
        return torch.softmax(self._forward(log_b) + self._backward(log_b), dim=-1)

    @torch.no_grad()
    def viterbi(self, x) -> torch.Tensor:
        """Most probable state path: ``(T,)`` or ``(B, T)`` integers."""
        log_b = self.emission_log_prob(self._x(x))
        log_a = F.log_softmax(self.trans_logits, dim=1)
        delta = F.log_softmax(self.start_logits, dim=0) + log_b[..., 0, :]
        back = []
        for t in range(1, log_b.shape[-2]):
            score, arg = (delta.unsqueeze(-1) + log_a).max(dim=-2)
            back.append(arg)
            delta = score + log_b[..., t, :]
        path = [delta.argmax(dim=-1)]
        for arg in reversed(back):
            path.append(arg.gather(-1, path[-1].unsqueeze(-1)).squeeze(-1))
        return torch.stack(path[::-1], dim=-1)

    # -- fitting --------------------------------------------------------------
    def initialise(self, x, seed: int = 0) -> None:
        """Start means at quantiles along the first principal direction and variances at the overall variance."""
        x = self._x(x).reshape(-1, self.dim)
        g = torch.Generator().manual_seed(int(seed))
        centred = x - x.mean(0)
        direction = (torch.linalg.svd(centred, full_matrices=False)[2][0] if x.shape[0] > 1
                     else torch.ones(self.dim, dtype=x.dtype))
        order = torch.argsort(centred @ direction)
        cuts = torch.linspace(0, x.shape[0], self.n_states + 1).long()
        with torch.no_grad():
            for k in range(self.n_states):
                members = order[cuts[k]:max(int(cuts[k + 1]), int(cuts[k]) + 1)]
                self.means[k] = x[members].mean(0) + 1e-3 * torch.randn(self.dim, generator=g, dtype=x.dtype)
            # a single observation has no variance (var() would be NaN); fall back to the floor
            var = x.var(0) if x.shape[0] > 1 else torch.zeros(self.dim, dtype=x.dtype)
            self.logvars.copy_(torch.log(var.clamp_min(1e-6)).expand_as(self.logvars))

    def fit(self, x, n_iter: int = 200, lr: float = 0.05, seed: int = 0, initialise: bool = True) -> "GaussianHMM":
        """Maximize the log-likelihood of one sequence, or a batch of sequences, with Adam."""
        x = self._x(x)
        if initialise:
            self.initialise(x, seed)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(int(n_iter)):
            loss = -self.log_likelihood(x).sum() / x[..., 0].numel()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    def to_dict(self):
        return {"n_states": self.n_states, "dim": self.dim, "stickiness": self.stickiness}

    @classmethod
    def from_dict(cls, params):
        return GaussianHMM(n_states=params["n_states"], dim=params["dim"],
                           stickiness=params.get("stickiness", 0.9))


def segment_by_order(values: np.ndarray, order: np.ndarray, n_states: int = 3, stickiness: float = 0.95, n_iter: int = 200,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray, GaussianHMM]:
    """
    Segment samples into runs of states along an ordering.

    Args:
        values: ``(samples, dims)`` or ``(samples,)`` — e.g. latent coordinates or a score.
        order: One number per sample — e.g. pseudotime or genomic position.
        n_states: Number of states.
        stickiness: Initial self-transition probability.
        n_iter: Optimization steps.
        seed: Seed for initialization.

    Returns:
        Tuple ``(states, probabilities, model)`` in the original sample order.
        State numbers are arbitrary labels.
    """
    values = np.asarray(values.detach().cpu() if torch.is_tensor(values) else values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    order = np.asarray(order, dtype=float)
    if order.shape != (values.shape[0],):
        raise ValueError("order must have one entry per sample")
    idx = np.argsort(order, kind="stable")
    hmm = GaussianHMM(n_states, values.shape[1], stickiness).fit(values[idx], n_iter=n_iter, seed=seed)
    states = np.empty(len(idx), dtype=int)
    probs = np.empty((len(idx), n_states))
    states[idx] = hmm.viterbi(values[idx]).numpy()
    probs[idx] = hmm.posterior(values[idx]).numpy()
    return states, probs, hmm
