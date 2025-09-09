"""
Encoder layer
"""

from typing import Optional, List, Union
from torch import nn
import torch
from ...layers import MaskedLinear, LayerInfo, convert_activation
from ...constraints import NetworkConstraint
import logging

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """
    input -> [optional global BN] -> [LayerInfo...] -> (latent heads optional)
    Behavior:
      - If `layers` is empty/None: we auto-insert a projection to `latent_dim` and attach latent heads.
      - If `layers` is provided: we do NOT project to latent; we only attach latent heads
        when the last layer's width equals `latent_dim`. Otherwise we raise with a clear message.
    Forward:
      - With latent heads: returns (mu, logvar, z)
      - Without latent heads: returns hidden h
    """

    def __init__(self,
                 feature_dim: int,
                 latent_dim: Optional[int] = None,
                 layers: Optional[List[LayerInfo]] = None,
                 constraint: Optional[NetworkConstraint] = None,
                 batch_norm: bool = False,
                 default_activation: Union[str, None] = "relu",
                 make_latent_heads: bool = True):
        super().__init__()
        self._constraint = constraint
        self._default_activation = default_activation
        self._make_latent_heads = make_latent_heads

        self.net = nn.ModuleList()
        in_features = feature_dim

        # Optional global BN on input
        if batch_norm:
            self.net.append(nn.BatchNorm1d(in_features))

        # ---- Build user-defined hidden stack (no implicit latent projection here) ----
        if layers:
            logger.info("Building encoder with %d layers", len(layers))
            for i, li in enumerate(layers):
                out_features = li.units

                if li.op == "masked_linear":
                    init_mask = None
                    if constraint is not None and getattr(constraint, "_mask_np", None) is not None:
                        init_mask = torch.tensor(constraint._mask_np, dtype=torch.float32)
                    layer = MaskedLinear(in_features, out_features, bias=li.bias, mask=init_mask)
                elif li.op == "linear":
                    layer = nn.Linear(in_features, out_features, bias=li.bias)
                else:
                    raise ValueError(f"Unknown LayerInfo.op '{li.op}' at index {i}")

                self.net.append(layer)

                if li.activation is not None:
                    act = convert_activation(li.activation)
                    if act is not None:
                        self.net.append(act)

                if li.batch_norm:
                    self.net.append(nn.BatchNorm1d(out_features))

                in_features = out_features

            # If we want latent heads, the final width must already match latent_dim.
            self.z_mean = None
            self.z_log_var = None
            if self._make_latent_heads:
                if latent_dim is None:
                    raise ValueError("latent_dim is required when make_latent_heads=True.")
                if in_features != latent_dim:
                    raise ValueError(
                        "Final hidden width must equal latent_dim because the encoder "
                        "does not insert a latent projection when layers are provided.\n"
                        f"Final hidden size: {in_features}  vs  latent_dim: {latent_dim}\n"
                        "Fix by setting your last LayerInfo(units=latent_dim)."
                    )
                self.z_mean = nn.Linear(latent_dim, latent_dim)
                self.z_log_var = nn.Linear(latent_dim, latent_dim)

        else:
            if latent_dim is None:
                raise ValueError(
                    "latent_dim is required when no layers are provided (auto-projection)."
                )
            logger.info("No encoder layers provided; inserting auto-projection to latent_dim=%d", latent_dim)

            # Auto projection to latent size
            proj = nn.Linear(in_features, latent_dim, bias=True)
            self.net.append(proj)

            # Optional default activation after the auto-projection
            act = convert_activation(self._default_activation)
            if act is not None:
                self.net.append(act)

            # Optional BN after the auto-projection
            self.net.append(nn.BatchNorm1d(latent_dim))

            in_features = latent_dim

            # Latent heads if requested
            self.z_mean = None
            self.z_log_var = None
            if self._make_latent_heads:
                self.z_mean = nn.Linear(latent_dim, latent_dim)
                self.z_log_var = nn.Linear(latent_dim, latent_dim)

        # Book-keeping
        self._final_width = in_features

    def refresh_mask(self, device: torch.device):
        """Update masks from constraint (if MaskedLinear used)."""
        if self._constraint is None:
            return
        mask_t = self._constraint.as_torch(device)
        for m in self.net:
            if isinstance(m, MaskedLinear):
                m.set_mask(mask_t)

    def forward(self, x: torch.Tensor):
        h = x
        for layer in self.net:
            h = layer(h)

        if self._make_latent_heads and (self.z_mean is not None) and (self.z_log_var is not None):
            mu = self.z_mean(h)
            logvar = self.z_log_var(h)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return mu, logvar, z

        # No latent heads: return hidden representation
        return h
