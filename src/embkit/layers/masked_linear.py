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

    def __init__(self, in_features: int, out_features: int, bias: bool = True, mask: Optional[torch.Tensor] = None, device=None):
        """
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            bias: If set to False, the layer will not learn an additive bias. Default: True
            mask: Optional binary mask of shape (out_features, in_features). If None, defaults to all ones.

        Raises:
            ValueError: if `mask` is provided and does not match shape (out_features,
                        in_features).


        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias, device=device)

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

        Args:
            mask: binary mask of shape (out_features, in_features)
        Raises:
            AssertionError: if `mask` does not match shape of weight.
        """
        if mask.shape != self.linear.weight.shape:
            raise AssertionError(
                f"Mask shape {tuple(mask.shape)} must match weight shape "
                f"{tuple(self.linear.weight.shape)}."
            )
        self.mask.copy_(mask.to(self.mask.device, self.mask.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying the masked linear transformation.

        Args:
            x: input tensor of shape (batch_size, in_features)
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        w = self.linear.weight * self.mask
        return F.linear(x, w, self.linear.bias)

    def extra_repr(self) -> str:
        """
        String representation of the layer showing key parameters.

        Returns:
            A string with in_features, out_features, and bias information.
        """
        return (f"in_features={self.linear.in_features}, "
                f"out_features={self.linear.out_features}, "
                f"bias={self.linear.bias is not None}")
