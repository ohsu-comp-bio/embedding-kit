import logging
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import pandas as pd
from tqdm.autonotebook import tqdm
from torch.optim import Adam
from ...factory.layers import Layer, LayerList
from .base_vae import BaseVAE
from collections.abc import Callable
from ... import get_device, dataframe_loader
from torch import nn

logger = logging.getLogger(__name__)


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
            latent_dim: Optional[int] = None,
            encoder_layers: Optional[LayerList] = None,
            decoder_layers: Optional[LayerList] = None,
            batch_norm: bool = False,
            lr: float = 1e-3,
            encoder: Optional[nn.Module] = None,
            decoder: Optional[nn.Module] = None,
            device= None
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
        self._encoder_layers_cfg = list(encoder_layers or [])
        self._decoder_layers_cfg = list(decoder_layers or [])
        self._batch_norm = batch_norm

        feature_dim = len(features)

        if encoder is not None and decoder is not None:
            # Loaded path (from BaseVAE.open_model): modules are already built & weight-loaded
            self.encoder = encoder
            self.decoder = decoder
        else:
            # Fresh build path
            if latent_dim is None:
                raise ValueError("latent_dim is required when encoder/decoder are not provided.")
            self.encoder = self.build_encoder(
                feature_dim=feature_dim,
                latent_dim=latent_dim,
                layers=encoder_layers,
                batch_norm=batch_norm,
                device=device,
            )
            self.decoder = self.build_decoder(
                feature_dim=feature_dim,
                latent_dim=latent_dim,
                layers=decoder_layers,
                device=device,
            )

        # A place to record simple history if you want
        self.history: Dict[str, list] = {"loss": [], "recon": [], "kl": []}
        self.latent_index = None
        self.latent_groups = None
        self.normal_stats = None



if __name__ == "__main__":
    # Example usage
    N = 100
    df = pd.DataFrame({
        "feat1": np.random.rand(N),
        "feat2": np.random.rand(N),
    })

    vae: VAE = VAE(features=list(df.columns), latent_dim=2)
    vae.fit(df, epochs=10, lr=0.01)

    # Save the model if needed
    vae.save("vae_model")

    vae: VAE = VAE.open_model(path="vae_model", model_cls=VAE, device="cpu")
    print("Model loaded with features:", vae.features)
