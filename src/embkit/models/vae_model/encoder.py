from torch import nn
from typing import Optional, Tuple, List
import torch
from ...layers import MaskedLinear, LayerInfo, convert_activation
from ...constraints import NetworkConstraint
import logging

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """
    input -> [optional global BN] -> [LayerInfo...] -> embed (latent_dim)
         -> BN -> (z_mean, z_log_var) -> reparam -> (mu, logvar, z)
    """

    def __init__(self,
                 feature_dim: int,
                 latent_dim: int,
                 layers: Optional[List[LayerInfo]] = None,
                 constraint: Optional[NetworkConstraint] = None,
                 batch_norm: bool = False,
                 default_activation: str = "relu"):
        super().__init__()
        self._constraint = constraint
        self._default_activation = default_activation

        self.net = nn.ModuleList()
        in_features = feature_dim

        # Optional global BN on input
        if batch_norm:
            self.net.append(nn.BatchNorm1d(in_features))

        # Hidden stack
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

                if li.activation is not None:  # only if explicitly set
                    act = convert_activation(li.activation)
                    if act is not None:
                        self.net.append(act)

                if li.batch_norm:
                    self.net.append(nn.BatchNorm1d(out_features))

                in_features = out_features
        else:
            logger.info("Building encoder with no hidden layers")

        # Latent "embedding" layer before parameter heads
        self.embedding = nn.Linear(in_features, latent_dim)
        self.embedding_act = convert_activation(self._default_activation)
        self.embedding_bn = nn.BatchNorm1d(latent_dim)

        # Latent parameter heads
        self.z_mean = nn.Linear(latent_dim, latent_dim)
        self.z_log_var = nn.Linear(latent_dim, latent_dim)

    def refresh_mask(self, device: torch.device):
        """Update masks from constraint (if MaskedLinear used)."""
        if self._constraint is None:
            return
        mask_t = self._constraint.as_torch(device)
        for m in self.net:
            if isinstance(m, MaskedLinear):
                m.set_mask(mask_t)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = x
        for layer in self.net:
            h = layer(h)

        h = self.embedding(h)
        if self.embedding_act is not None:
            h = self.embedding_act(h)
        h = self.embedding_bn(h)

        mu = self.z_mean(h)
        logvar = self.z_log_var(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return mu, logvar, z
