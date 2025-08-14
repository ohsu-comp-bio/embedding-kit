import json
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import pandas as pd
from torch.optim import Adam

from .base_vae import VAE as BaseVAE  # <- alias to avoid name clash
from ...losses import vae_loss  # your existing loss

# If Encoder/Decoder kwargs need constraints etc., they’ll be passed through.

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info("Using CUDA GPU")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        logger.info("Using Metal GPU")
        return torch.device("mps")
    logger.info("Using CPU")
    return torch.device("cpu")


class VAE(BaseVAE):
    """
    Concrete VAE that composes the modular Encoder/Decoder from BaseVAE
    and provides a simple fit() loop.

    BaseVAE.forward(x) returns: recon, mu, logvar, z
    """

    def __init__(
            self,
            features: List[str],
            *,
            latent_dim: int,
            # encoder config
            encoder_layers: Optional[List[Dict]] = None,
            constraint=None,
            batch_norm: bool = False,
            activation: str = "relu",
            # decoder config
            hidden_dim_ignored: Optional[int] = None,  # kept for backward API compat
            # training defaults
            lr: float = 1e-3,
    ):
        """
        Args:
            features: list[str] feature names (len(features) == input_dim)
            latent_dim: size of latent space
            encoder_layers: optional list of layer configs for Encoder
            constraint, batch_norm, activation: forwarded to Encoder builder
            hidden_dim_ignored: kept only to mirror old API (not used)
            lr: default learning rate for fit()
        """
        super().__init__(features=features)
        self.lr = lr

        feature_dim = len(features)

        # Use the base helpers to build fresh modules
        self.encoder = self.build_encoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=encoder_layers,
            constraint=constraint,
            batch_norm=batch_norm,
            activation=activation,
        )
        self.decoder = self.build_decoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
        )

        # A place to record simple history if you want
        self.history: Dict[str, list] = {"loss": []}
        self.latent_index = None
        self.latent_groups = None
        self.normal_stats = None

    # NOTE: forward() is inherited from BaseVAE

    def fit(self, X: pd.DataFrame, y=None, *, epochs: int = 20, lr: Optional[float] = None,
            device: Optional[torch.device] = None, progress: bool = True):
        """
        Full-batch training loop using vae_loss(recon, x, mu, logvar).

        X: pandas.DataFrame with float features, columns must match `self.features`.
        """
        if lr is None:
            lr = self.lr

        if device is None:
            device = get_device()

        # Safety check for feature alignment
        if hasattr(X, "columns") and self.features is not None:
            if list(X.columns) != list(self.features):
                raise ValueError(
                    "Input DataFrame columns do not match model features.\n"
                    f"Data columns: {list(X.columns)[:5]}... (n={len(X.columns)})\n"
                    f"Model features: {self.features[:5]}... (n={len(self.features)})"
                )

        self.to(device)
        self.train()

        # IMPORTANT: assign the result of .to(device)
        x_tensor = torch.from_numpy(X.values.astype(np.float32)).to(device)

        optimizer = Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()

            recon, mu, logvar, _z = super().forward(x_tensor)
            total_loss, recon_loss, kl_loss = vae_loss(recon, x_tensor, mu, logvar)

            total_loss.backward()
            optimizer.step()

            self.history["loss"].append(float(total_loss.detach().cpu()))
            if progress:
                logger.debug(
                    f"Epoch {epoch + 1} | Loss: {total_loss.item():.4f} | "
                    f"Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f}"
                )
