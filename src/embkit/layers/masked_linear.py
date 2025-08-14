from torch import nn
from typing import Optional
import torch
import torch.nn.functional as F

class MaskedLinear(nn.Module):
    """
    Linear layer whose weight is elementwise-multiplied by a mask at forward time.
    PyTorch Linear weight shape: (out_features, in_features). Mask must match this shape.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if mask is None:
            mask = torch.ones(out_features, in_features)
        self.register_buffer("mask", mask)

    def set_mask(self, mask: torch.Tensor):
        assert mask.shape == self.linear.weight.shape
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.linear.weight * self.mask
        return F.linear(x, w, self.linear.bias)


