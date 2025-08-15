from torch import nn
from typing import Optional
import torch
import torch.nn.functional as F


class MaskedLinear(nn.Module):
    """
    Linear layer whose weight is elementwise-multiplied by a mask at forward time.

    - Expects `in_features` and `out_features` at construction time (PyTorch convention).
    - `mask` is registered as a buffer so it is saved/loaded and moves with `.to()/.cuda()`.
    - `set_mask()` updates the existing buffer (no rebind), preserving state_dict compatibility.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Create/validate mask and register as buffer (moves with .to(), saved in state_dict)
        if mask is None:
            mask = torch.ones(
                out_features, in_features,
                dtype=self.linear.weight.dtype,
                device=self.linear.weight.device,
            )
        else:
            if mask.shape != (out_features, in_features):
                raise ValueError(
                    f"Mask shape {tuple(mask.shape)} must be "
                    f"({out_features}, {in_features})."
                )
            mask = mask.to(self.linear.weight.device, self.linear.weight.dtype)

        self.register_buffer("mask", mask, persistent=True)

    @torch.no_grad()
    def set_mask(self, mask: torch.Tensor) -> None:
        """
        Update the mask buffer without rebinding (keeps state_dict key stable).
        """
        if mask.shape != self.linear.weight.shape:
            raise AssertionError(
                f"Mask shape {tuple(mask.shape)} must match weight shape "
                f"{tuple(self.linear.weight.shape)}."
            )
        self.mask.copy_(mask.to(self.mask.device, self.mask.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.linear.weight * self.mask
        return F.linear(x, w, self.linear.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.linear.in_features}, "
                f"out_features={self.linear.out_features}, "
                f"bias={self.linear.bias is not None}")
