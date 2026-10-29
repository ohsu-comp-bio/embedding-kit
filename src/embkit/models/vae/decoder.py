from typing import List, Optional
import torch
from torch import nn

from ... import factory
from ...modules import MaskedLinear
from ...factory.layers import Layer, LayerList
import logging

logger = logging.getLogger(__name__)


@factory.nn_module
class Decoder(nn.Module):
    """
    z -> [LayerInfo...] -> recon(features)
    """

    def __init__(
        self,
        latent_dim: int,
        feature_dim: int,
        layers: Optional[LayerList] = None,
        batch_norm: bool = False,
        device=None, dtype=None
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)  # <- help BaseVAE.save()
        self.feature_dim = int(feature_dim)
        self._global_bn = batch_norm
        self._layers_cfg = layers
        self.net = nn.ModuleList()

        in_features = latent_dim

        if layers:
            logger.info("Building decoder with %d layers", len(layers))
            dec_net = layers.build( latent_dim, feature_dim, device=device, dtype=dtype )
            self.net.extend(dec_net)
            logger.debug("Decoder net built with %d modules", len(dec_net))
            self.out = [m for m in dec_net if isinstance(m, (nn.Linear, MaskedLinear))][-1] # Last projection. Output activation may follow
        else:
            logger.info("Adding final projection layer to %d units", self.feature_dim)
            self.out = nn.Linear(in_features, self.feature_dim, device=device, dtype=dtype)
            self.net.append(self.out)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for layer in self.net:
            h = layer(h)
        return h

    def to_dict(self):
        return {
            "latent_dim": self.latent_dim,
            "feature_dim": self.feature_dim,
            "batch_norm": self._global_bn,
            "layers": self._layers_cfg.to_dict() if self._layers_cfg else []
        }

    @classmethod
    def from_dict(cls, d):
        raw = d.get("layers")
        if isinstance(raw, dict): # LayerList dict 
            layers = factory.build(raw)
        else: # Legacy Format: list of Layer dicts
            layers = LayerList([Layer.from_dict(ld) for ld in raw]) if raw else None
        return Decoder(
            latent_dim=d["latent_dim"],
            feature_dim=d["feature_dim"],
            batch_norm=d.get("batch_norm", False),
            layers=layers
        )
