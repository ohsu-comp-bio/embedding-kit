from torch import nn
from typing import Optional, Tuple, List, Dict
import torch
from src.embkit.layers import MaskedLinear
from src.embkit.constraints import NetworkConstraint
import logging

logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    """
    Dynamically built encoder:
      input -> [optional BN] -> [list of Dense layers] -> latent dense -> BN -> (z_mean, z_log_var) -> sample z
    forward(x) returns (z_mean, z_log_var, z)
    """

    def __init__(self, feature_dim: int, latent_dim: int,
                 layers: List[Dict] = None,
                 constraint: Optional[NetworkConstraint] = None,
                 batch_norm: bool = False,
                 activation: str = "relu"):
        super().__init__()
        self._constraint = constraint

        self.net = nn.ModuleList()
        in_features = feature_dim

        # Optional initial batch norm
        if batch_norm:
            self.net.append(nn.BatchNorm1d(in_features))

        # Build intermediate layers
        if layers:
            logger.info(f"Building encoder with {len(layers)} layers")
            for i, layer_cfg in enumerate(layers):
                units = layer_cfg.get("units")
                use_mask = layer_cfg.get("masked", False)

                if use_mask:
                    init_mask = None
                    if constraint is not None and constraint._mask_np is not None:
                        init_mask = torch.tensor(constraint._mask_np, dtype=torch.float32)
                    self.net.append(MaskedLinear(in_features, units, mask=init_mask))
                else:
                    self.net.append(nn.Linear(in_features, units))

                act = self._get_activation(activation)
                if act is not None:
                    self.net.append(act)

                if layer_cfg.get("batch_norm", False):
                    self.net.append(nn.BatchNorm1d(units))

                in_features = units
        else:
            logger.info(f"Building encoder with no layers")

        # Embedding layer
        self.embedding = nn.Linear(in_features, latent_dim)
        self.embedding_act = self._get_activation(activation)
        self.embedding_bn = nn.BatchNorm1d(latent_dim)

        # Latent outputs
        self.z_mean = nn.Linear(latent_dim, latent_dim)
        self.z_log_var = nn.Linear(latent_dim, latent_dim)

    def _get_activation(self, activation: str):
        act_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            None: None
        }
        return act_map.get(activation.lower()) if activation else None

    def refresh_mask(self, device: torch.device):
        if self._constraint is not None:
            for module in self.net:
                if isinstance(module, MaskedLinear):
                    module.set_mask(self._constraint.as_torch(device))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = x
        for layer in self.net:
            h = layer(h)

        h = self.embedding(h)
        if self.embedding_act:
            h = self.embedding_act(h)
        h = self.embedding_bn(h)

        mu = self.z_mean(h)
        logvar = self.z_log_var(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return mu, logvar, z