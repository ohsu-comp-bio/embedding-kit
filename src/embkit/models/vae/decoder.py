from typing import List, Optional
import torch
from torch import nn
from ...constraints import NetworkConstraint
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
                 constraint: Optional[NetworkConstraint] = None,
                 batch_norm: bool = False,
                 default_activation: str = "relu"):
        super().__init__()
        self._default_activation = default_activation
        self._global_bn = batch_norm
        self.net = nn.ModuleList()
        in_features = latent_dim

        if layers:
            logger.info("Building decoder with %d layers", len(layers))
            for i, li in enumerate(layers):
                out_features = li.units

                # 1) Linear / MaskedLinear
                if li.op == "masked_linear":
                    layer = MaskedLinear(in_features, out_features, bias=li.bias, mask=None)
                elif li.op == "linear":
                    layer = nn.Linear(in_features, out_features, bias=li.bias)
                else:
                    raise ValueError(f"Unknown LayerInfo.op '{li.op}' at index {i}")
                self.net.append(layer)

                # 2) BatchNorm (Linear -> BN -> Activation)
                use_bn = getattr(li, "batch_norm", False)
                # honor global default if LayerInfo doesn't request BN explicitly
                if use_bn or self._global_bn:
                    self.net.append(nn.BatchNorm1d(out_features))

                # 3) Activation (fallback to default if not provided)
                act_name = li.activation if li.activation is not None else self._default_activation
                act = convert_activation(act_name)
                if act is not None:
                    self.net.append(act)

                in_features = out_features
        else:
            logger.info("Building decoder with no hidden layers")

        # Final projection to feature space (linear head; no activation)
        self.out = nn.Linear(in_features, feature_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for layer in self.net:
            h = layer(h)
        return self.out(h)