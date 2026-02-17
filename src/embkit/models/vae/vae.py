
import numpy as np
import pandas as pd

import logging
from typing import Dict, List, Optional, Union
from collections.abc import Callable

from torch import nn

from .base_vae import BaseVAE
from ...factory.mapping import nn_module, get_class_name
from ...factory.layers import Layer, LayerList
from ... import get_device, dataframe_loader

logger = logging.getLogger(__name__)


@nn_module
class VAE(BaseVAE):
    """
    Concrete VAE that composes the modular Encoder/Decoder from BaseVAE
    and provides a simple fit() loop.

    BaseVAE.forward(x) returns: recon, mu, logvar, z
    """

    def __init__(
            self,
            features: List[str],
            latent_dim: Optional[int] = None,
            encoder_layers: Optional[LayerList] = None,
            decoder_layers: Optional[LayerList] = None,
            batch_norm: bool = False,
            device= None, dtype=None
    ):
        """
        Args:
            features: list[str] feature names (len(features) == input_dim)
            latent_dim: size of latent space
            encoder_layers: list of layer configs for Encoder
            constraint, batch_norm, activation: forwarded to Encoder builder
        """
        super().__init__(features=features)

        if encoder_layers is None:
            encoder_layers = LayerList()
        if decoder_layers is None:
            decoder_layers = LayerList()

        self._encoder_layers_cfg = encoder_layers
        self._decoder_layers_cfg = decoder_layers
        self._batch_norm = batch_norm
        self.latent_dim = latent_dim

        feature_dim = len(features)

        if latent_dim is None:
            raise ValueError("latent_dim is required when encoder/decoder are not provided.")
        self.encoder = self.build_encoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=encoder_layers,
            batch_norm=batch_norm,
            device=device, dtype=dtype
        )
        self.decoder = self.build_decoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=decoder_layers,
            device=device, dtype=dtype
        )

    def to_dict(self):
        return {
            "__class__" : get_class_name(VAE),
            "features": self.features,
            "latent_dim": self.latent_dim,
            "encoder_layers": [li.to_dict() for li in self._encoder_layers_cfg],
            "decoder_layers": [li.to_dict() for li in self._decoder_layers_cfg],
            "batch_norm": self._batch_norm
        }

    @classmethod
    def from_dict(cls, desc):
        return VAE(
            features=desc["features"],
            latent_dim=desc["latent_dim"],
            encoder_layers=LayerList( [Layer.from_dict(li) for li in desc["encoder_layers"]] ),
            decoder_layers=LayerList( [Layer.from_dict(li) for li in desc["decoder_layers"]] ),
            batch_norm=desc.get("batch_norm", False)
        )


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
