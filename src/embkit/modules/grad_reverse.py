"""
Gradient reversal layer.

Adversarial training is a common way to strip a nuisance signal (batch, cohort,
sequencing site) out of an embedding: attach a classifier that tries to predict
the nuisance label from the latent vector, and train the encoder to defeat it.
The gradient reversal layer makes that a single backward pass — it is the
identity in the forward direction and multiplies the gradient by ``-lambd`` on
the way back, so the encoder receives the *opposite* of the classifier's
gradient (Ganin & Lempitsky, 2015).

Usage::

    adv_head = nn.Sequential(GradientReversal(lambd=1.0), nn.Linear(latent_dim, n_batches))
    loss = recon_loss + kl + F.cross_entropy(adv_head(z), batch_labels)

Ported from UniVII (``_GradReverseFn`` in ``univii.models.univii``).
"""
import torch
from torch import nn

from .. import factory


class _GradReverseFn(torch.autograd.Function):
    """Forward: identity. Backward: ``-lambd * grad``."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """
    Functional gradient reversal: returns ``x`` unchanged, reverses and scales its gradient.

    Args:
        x: Input tensor.
        lambd: Scale applied to the reversed gradient.
    """
    return _GradReverseFn.apply(x, float(lambd))


@factory.nn_module
class GradientReversal(nn.Module):
    """
    Module form of :func:`grad_reverse`, so it can sit inside an ``nn.Sequential``.

    Args:
        lambd: Scale applied to the reversed gradient. Can be changed between
            epochs (e.g. ramped up from 0) by setting ``module.lambd``.
    """

    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_reverse(x, self.lambd)

    def extra_repr(self) -> str:
        return f"lambd={self.lambd}"

    def to_dict(self):
        return {"lambd": self.lambd}

    @classmethod
    def from_dict(cls, params):
        return GradientReversal(lambd=params["lambd"])
