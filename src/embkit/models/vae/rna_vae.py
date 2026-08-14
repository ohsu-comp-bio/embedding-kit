"""
RNAVAE - RNA-specific Variational Autoencoder

Integrated version using BaseVAE infrastructure while preserving exact
TensorFlow architecture with BatchNorm and ReLU on latent heads.
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any
import torch
import pandas as pd
from torch.optim import Adam
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from .vae import VAE, Decoder, VAEOutput
from .encoder import Encoder, EncoderOutput
from ...factory.layers import Layer, LayerList
from ... import get_device
from ...losses import BCEKLWeightedVAELoss
from ... import factory

logger = logging.getLogger(__name__)


class RNAEncoder(Encoder):
    """
    Extended Encoder for RNA VAE that adds BatchNorm + ReLU to latent heads.
    
    WHY THIS EXISTS:
    The standard Encoder produces latent heads as: mu = Linear(h), logvar = Linear(h)
    This allows mu and logvar to be any real number (standard VAE practice).
    
    Your TensorFlow RNA VAE uses: mu = ReLU(BatchNorm(Linear(h)))
    This constrains mu and logvar to be non-negative (≥ 0), fundamentally changing
    the latent space behavior. Without this custom encoder, the PyTorch model would
    produce mathematically different embeddings than your TensorFlow model.
    
    Architecture:
    - z_mean: Linear -> BatchNorm -> ReLU (NOT standard VAE)
    - z_log_var: Linear -> BatchNorm -> ReLU (NOT standard VAE)
    """
    
    def __init__(self, feature_dim: int, latent_dim: int, 
                 layers: Optional[List[Layer]] = None,
                 batch_norm: bool = False):
        # Initialize parent without making latent heads
        super().__init__(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=layers,
            batch_norm=batch_norm,
            make_latent_heads=False  # We'll build custom ones
        )
        
        # Build custom latent heads with BatchNorm + ReLU
        # Linear -> BatchNorm -> ReLU (matching TensorFlow)
        self.z_mean_linear = nn.Linear(self._final_width, latent_dim)
        self.z_mean_bn = nn.BatchNorm1d(latent_dim)
        
        self.z_log_var_linear = nn.Linear(self._final_width, latent_dim)
        self.z_log_var_bn = nn.BatchNorm1d(latent_dim)
        
        # Xavier/Glorot uniform initialization (TensorFlow default)
        nn.init.xavier_uniform_(self.z_mean_linear.weight)
        nn.init.zeros_(self.z_mean_linear.bias)
        nn.init.xavier_uniform_(self.z_log_var_linear.weight)
        nn.init.zeros_(self.z_log_var_linear.bias)
    
    def forward(self, x: torch.Tensor):
        # Pass through main network
        h = x
        for layer in self.net:
            h = layer(h)
        
        # z_mean: Linear -> BatchNorm -> ReLU
        mu = self.z_mean_linear(h)
        mu = self.z_mean_bn(mu)
        mu = torch.relu(mu)
        
        # z_log_var: Linear -> BatchNorm -> ReLU  
        logvar = self.z_log_var_linear(h)
        logvar = self.z_log_var_bn(logvar)
        logvar = torch.relu(logvar)
        
        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return EncoderOutput(mu=mu, logvar=logvar, z=z)


@factory.nn_module
class RNAVAE(VAE):
    """    
    Architecture:
    - Encoder: feature_dim -> feature_dim//2 -> feature_dim//3 -> latent_dim
      - Latent heads: Linear -> BatchNorm -> ReLU
    - Decoder: latent_dim -> feature_dim (sigmoid)
    - Loss: feature_dim * BCE + 5 * beta * KL
    - Beta warmup: 0 -> 1 (kappa rate per epoch)
    """

    def __init__(
            self,
            features: List[str],
            latent_dim: int = 768,
    ):
        feature_dim = len(features)

        # Build encoder: feature_dim -> feature_dim//2 -> feature_dim//3
        enc_layers = [
            Layer(units=feature_dim // 2, activation="relu"),
            Layer(units=feature_dim // 3, activation="relu"),
        ]
        
        # Use custom RNAEncoder with BatchNorm + ReLU on latent heads
        encoder = RNAEncoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=LayerList(enc_layers),
            batch_norm=False  # We add BN to latent heads specifically
        )

        # Build decoder: latent_dim -> feature_dim with sigmoid
        dec_layers = [
            Layer(units=feature_dim, activation="sigmoid"),
        ]
        
        decoder = Decoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=LayerList(dec_layers),
        )

        super().__init__(encoder=encoder, decoder=decoder)
        self.features = features
        self.latent_dim = latent_dim

    def _initialize_weights(self):
        """Initialize weights with glorot_uniform like TensorFlow"""
        for m in self.modules():
            if isinstance(m, nn.Linear) and not hasattr(m, '_initialized'):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                m._initialized = True

    def forward(self, x: torch.Tensor):
        """Standard VAE forward pass"""
        mu, logvar, z = self.encoder(x)
        recon = self.decoder(z)
        return VAEOutput(recon=recon, mu=mu, logvar=logvar, z=z)

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Specific check for RNAVAE to ensure the BatchNorm+ReLU latent heads
        are producing strictly non-negative mu and logvar.
        """
        report = super().verify_integrity()

        self.eval()
        device = next(self.parameters()).device
        dummy_input = torch.randn(100, len(self.features), device=device)
        with torch.no_grad():
            mu, logvar, _ = self.encoder(dummy_input)

            min_mu = float(torch.min(mu))
            min_logvar = float(torch.min(logvar))

            # Strict numerical tolerance for integrity mode
            is_non_negative = (min_mu >= -1e-9 and min_logvar >= -1e-9)

            report["rna_diagnostics"] = {
                "min_mu": min_mu,
                "min_logvar": min_logvar,
                "is_non_negative": is_non_negative
            }

            if not is_non_negative:
                report["healthy"] = False
                report["issues"].append(
                    f"Negative values detected in RNAVAE latent heads "
                    f"(mu_min={min_mu:.2e}, logvar_min={min_logvar:.2e}). "
                    f"Architectural constraints violated (ReLU/BatchNorm bypass)."
                )

        return report

    def to_dict(self) -> Dict[str, Any]:
        return {
            "features": self.features,
            "latent_dim": self.latent_dim,
        }

    @classmethod
    def from_dict(cls, d):
        model = RNAVAE(
            features=d["features"],
            latent_dim=d.get("latent_dim", 768)
        )
        return model
