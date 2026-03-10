from typing import Optional, List, Union, TYPE_CHECKING
from torch import nn
import torch

from ... import factory
from ...modules import MaskedLinear
from ...factory.layers import Layer, LayerList
from ...factory.mapping import get_activation

import logging

if TYPE_CHECKING:
    from ...constraints import NetworkConstraint

logger = logging.getLogger(__name__)


@factory.nn_module
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
                 latent_dim: int,
                 layers: Optional[LayerList] = None,
                 batch_norm: bool = False,
                 default_activation: Union[str, None] = "relu",
                 make_latent_heads: bool = True,
                 sampling : bool = False,
                 constraint: Optional["NetworkConstraint"] = None,
                 device=None, dtype=None):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim)
        self._default_activation = default_activation
        self._make_latent_heads = make_latent_heads
        self._sampling = sampling
        self.constraint = constraint
        self.batch_norm = batch_norm

        self.net = nn.ModuleList()
        in_features = feature_dim

        # Optional global BN on input
        if batch_norm:
            self.net.append(nn.BatchNorm1d(in_features, device=device, dtype=dtype))

        if layers:
            logger.info("Building encoder with %d layers", len(layers))
            enc_net = layers.build( input_dim=in_features, output_dim=self.latent_dim, device=device, dtype=dtype)
            self.net.extend(enc_net)

            in_features = enc_net[-1].out_features

            # Latent heads requirement
            self.z_mean = None
            self.z_log_var = None
            if self._make_latent_heads:
                if in_features != self.latent_dim:
                    raise ValueError(
                        "Final hidden width must equal latent_dim because the encoder "
                        "does not insert a latent projection when layers are provided.\n"
                        f"Final hidden size: {in_features}  vs  latent_dim: {self.latent_dim}\n"
                        "Fix by setting your last Layer(units=latent_dim)."
                    )
                self.z_mean = nn.Linear(self.latent_dim, self.latent_dim, device=device, dtype=dtype)
                self.z_log_var = nn.Linear(self.latent_dim, self.latent_dim, device=device, dtype=dtype)

        else:
            logger.info("No encoder layers provided; inserting auto-projection to latent_dim=%d", self.latent_dim)

            # Auto projection to latent size (masked when constraint is provided)
            if self.constraint is not None:
                proj = MaskedLinear(in_features, self.latent_dim, bias=True, device=device, dtype=dtype)
                self.net.append(proj)
                proj.set_mask(self.constraint.as_torch(device=proj.mask.device))
            else:
                proj = nn.Linear(in_features, self.latent_dim, bias=True, device=device, dtype=dtype)
                self.net.append(proj)

            # Optional default activation after the auto-projection
            act = get_activation(self._default_activation)
            if act is not None:
                self.net.append(act())

            # Optional BN after the auto-projection
            self.net.append(nn.BatchNorm1d(self.latent_dim, device=device, dtype=dtype))

            in_features = self.latent_dim

            # Latent heads
            self.z_mean = None
            self.z_log_var = None
            if self._make_latent_heads:
                self.z_mean = nn.Linear(self.latent_dim, self.latent_dim, device=device, dtype=dtype)
                self.z_log_var = nn.Linear(self.latent_dim, self.latent_dim, device=device, dtype=dtype)

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
    
    def to_dict(self):
        return {
            "feature_dim": self.feature_dim,
            "latent_dim": self.latent_dim,
            "batch_norm": self.batch_norm,
            "default_activation": self._default_activation,
            "make_latent_heads": self._make_latent_heads,
            "sampling": self._sampling,
            "constraint": self.constraint.to_dict() if self.constraint else None,
        }

    @classmethod
    def from_dict(cls, d):
        from ...constraints import NetworkConstraint
        constraint = NetworkConstraint.from_dict(d["constraint"]) if d.get("constraint") else None
        return Encoder(
            feature_dim=d["feature_dim"],
            latent_dim=d["latent_dim"],
            batch_norm=d.get("batch_norm", False),
            default_activation=d.get("default_activation", "relu"),
            make_latent_heads=d.get("make_latent_heads", True),
            sampling=d.get("sampling", False),
            constraint=constraint
        )

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
