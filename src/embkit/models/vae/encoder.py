from typing import Optional, List, Union, TYPE_CHECKING
from torch import nn
import torch
from ...layers import MaskedLinear, LayerInfo, convert_activation
import logging

if TYPE_CHECKING:
    from ...constraints import NetworkConstraint

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """
    input -> [optional global BN] -> [LayerInfo...] -> (latent heads optional)

    If `layers` is provided:
      - Final hidden width MUST equal latent_dim when make_latent_heads=True.

    If `layers` is None/empty:
      - Insert a Linear projection to latent_dim (+ optional act + BN) and attach latent heads.
    """

    def __init__(self,
                 feature_dim: int,
                 latent_dim: Optional[int] = None,
                 layers: Optional[List[LayerInfo]] = None,
                 batch_norm: bool = False,
                 default_activation: Union[str, None] = "relu",
                 make_latent_heads: bool = True,
                 sampling : bool = False,
                 constraint: Optional["NetworkConstraint"] = None,
                 device=None):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim) if latent_dim is not None else None  # <- help BaseVAE.save()
        self._default_activation = default_activation
        self._make_latent_heads = make_latent_heads
        self._sampling = sampling
        self.constraint = constraint

        self.net = nn.ModuleList()
        in_features = feature_dim

        # Optional global BN on input
        if batch_norm:
            self.net.append(nn.BatchNorm1d(in_features, device=device))

        if layers:
            logger.info("Building encoder with %d layers", len(layers))
            for li in layers:
                out_features = li.units

                layer = li.gen_layer(in_features, device=device)
                self.net.append(layer)

                if li.activation is not None:
                    act = convert_activation(li.activation)
                    if act is not None:
                        self.net.append(act)

                if li.batch_norm:
                    self.net.append(nn.BatchNorm1d(out_features, device=device))

                in_features = out_features

            # Latent heads requirement
            self.z_mean = None
            self.z_log_var = None
            if self._make_latent_heads:
                if self.latent_dim is None:
                    raise ValueError("latent_dim is required when make_latent_heads=True.")
                if in_features != self.latent_dim:
                    raise ValueError(
                        "Final hidden width must equal latent_dim because the encoder "
                        "does not insert a latent projection when layers are provided.\n"
                        f"Final hidden size: {in_features}  vs  latent_dim: {self.latent_dim}\n"
                        "Fix by setting your last LayerInfo(units=latent_dim)."
                    )
                self.z_mean = nn.Linear(self.latent_dim, self.latent_dim, device=device)
                self.z_log_var = nn.Linear(self.latent_dim, self.latent_dim, device=device)

        else:
            if self.latent_dim is None:
                raise ValueError(
                    "latent_dim is required when no layers are provided (auto-projection)."
                )
            logger.info("No encoder layers provided; inserting auto-projection to latent_dim=%d", self.latent_dim)

            # Auto projection to latent size
            proj = nn.Linear(in_features, self.latent_dim, bias=True, device=device)
            self.net.append(proj)

            # Optional default activation after the auto-projection
            act = convert_activation(self._default_activation)
            if act is not None:
                self.net.append(act)

            # Optional BN after the auto-projection
            self.net.append(nn.BatchNorm1d(self.latent_dim, device=device))

            in_features = self.latent_dim

            # Latent heads
            self.z_mean = None
            self.z_log_var = None
            if self._make_latent_heads:
                self.z_mean = nn.Linear(self.latent_dim, self.latent_dim, device=device)
                self.z_log_var = nn.Linear(self.latent_dim, self.latent_dim, device=device)

        self._final_width = in_features

    def forward(self, x: torch.Tensor):
        h = x
        for layer in self.net:
            h = layer(h)

        if self._make_latent_heads and (self.z_mean is not None) and (self.z_log_var is not None):
            mu = self.z_mean(h)
            logvar = self.z_log_var(h)
            if self._sampling:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                return mu, logvar, z
            return mu, logvar, h

        return h

    def refresh_mask(self, device: torch.device) -> None:
        """
        Update masks in all MaskedLinear layers using the constraint.
        This is a no-op if there's no constraint.

        Args:
            device: The device to move the mask tensor to
        """
        if self.constraint is None:
            return
        
        mask_tensor = self.constraint.as_torch(device)
        
        for module in self.net:
            if isinstance(module, MaskedLinear):
                module.set_mask(mask_tensor)
