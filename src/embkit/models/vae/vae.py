import logging
from typing import List, Optional
import torch

from .base_vae import BaseVAE
from ...factory.mapping import nn_module, get_class_name
from ...factory.layers import Layer, LayerList

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
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            features: list[str] feature names (len(features) == input_dim)
            latent_dim: size of latent space
            encoder_layers: list of layer configs for Encoder
            decoder_layers: list of layer configs for Decoder
            batch_norm: enable encoder batch normalization blocks
            device: torch device used for module initialization
            dtype: torch dtype used for module initialization
        """
        super().__init__(features=features)

        if encoder_layers is None:
            encoder_layers = LayerList()
        elif isinstance(encoder_layers, list):
            encoder_layers = LayerList(encoder_layers)
        if decoder_layers is None:
            decoder_layers = LayerList()
        elif isinstance(decoder_layers, list):
            decoder_layers = LayerList(decoder_layers)

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

    @staticmethod
    def _layers_to_dict(layers) -> list:
        if layers is None:
            return []
        if isinstance(layers, LayerList):
            layers = layers.layers
        return [li.to_dict() for li in layers]

    def to_dict(self):
        return {
            "__class__" : get_class_name(VAE),
            "features": self.features,
            "latent_dim": self.latent_dim,
            "encoder_layers": self._layers_to_dict(self._encoder_layers_cfg),
            "decoder_layers": self._layers_to_dict(self._decoder_layers_cfg),
            "batch_norm": self._batch_norm
        }

    @classmethod
    def from_dict(cls, desc):
        return VAE(
            features=desc["features"],
            latent_dim=desc["latent_dim"],
            encoder_layers=LayerList([Layer.from_dict(li) for li in (desc.get("encoder_layers") or [])]),
            decoder_layers=LayerList([Layer.from_dict(li) for li in (desc.get("decoder_layers") or [])]),
            batch_norm=desc.get("batch_norm", False)
        )

