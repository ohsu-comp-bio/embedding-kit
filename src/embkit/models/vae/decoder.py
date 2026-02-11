from typing import List, Optional
import torch
from torch import nn
from ...layers import MaskedLinear, LayerInfo, convert_activation
import logging

logger = logging.getLogger(__name__)


class Decoder(nn.Module):
    """
    z -> [LayerInfo...] -> recon(features)
    """
    def __init__(self,
                 latent_dim: int,
                 feature_dim: int,
                 layers: Optional[List[LayerInfo]] = None,
                 batch_norm: bool = False,
                 default_activation: str = "relu",
                 device=None):
        super().__init__()
        self.latent_dim = int(latent_dim)       # <- help BaseVAE.save()
        self.feature_dim = int(feature_dim)
        self._default_activation = default_activation
        self._global_bn = batch_norm
        self.net = nn.ModuleList()

        in_features = latent_dim

        if layers:
            logger.info("Building decoder with %d layers", len(layers))
            for i, li in enumerate(layers):
                out_features = li.units

                layer = li.gen_layer(in_features, device=device)
                self.net.append(layer)

                # Activation (don't fall back to default if explicitly None)
                if li.activation is not None:
                    act = convert_activation(li.activation)
                    if act is not None:
                        self.net.append(act)

                # BatchNorm (Linear -> Activation -> BN)
                use_bn = getattr(li, "batch_norm", False)
                if use_bn or self._global_bn:
                    self.net.append(nn.BatchNorm1d(out_features, device=device))

                in_features = out_features

        # Final projection to feature_dim if not already there
        if in_features != self.feature_dim or not layers:
            logger.info("Adding final projection layer to %d units", self.feature_dim)
            self.out = nn.Linear(in_features, self.feature_dim, device=device)
            self.net.append(self.out)
        else:
            # If the last layer matches feature_dim, identify it as self.out
            # assuming the last layer in self.net that is a Linear/MaskedLinear is 'out'
            # But the tests seem to expect a dedicated .out attribute
            # Let's find the last Linear-like layer if it exists
            linear_layers = [m for m in self.net if isinstance(m, (nn.Linear, MaskedLinear))]
            if linear_layers:
                self.out = linear_layers[-1]
            else:
                # Fallback: create identity or just set it
                self.out = nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for layer in self.net:
            h = layer(h)
        return h