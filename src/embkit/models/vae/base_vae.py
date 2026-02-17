"""
Base VAE class
"""
from typing import Type, Any, List, Optional, Dict, overload, TypeVar, Union
from abc import ABC, abstractmethod
from pathlib import Path
import json
import logging
import pandas as pd
from torch import nn
import torch
from .encoder import Encoder
from .decoder import Decoder
from ...factory.layers import Layer, LayerList
from ... import get_device
import importlib
import inspect

logger = logging.getLogger(__name__)
T = TypeVar("T")


class BaseVAE(nn.Module, ABC):
    """
    Minimal VAE wrapper to hold encoder/decoder and provide forward().
    Allows late-binding of encoder/decoder by subclasses.
    """

    def __init__(self, features: List[str], encoder: Optional[Encoder] = None, decoder: Optional[Decoder] = None,
                 **kwargs):
        super().__init__()
        self.features = list(features)
        self.encoder: Optional[Encoder] = encoder
        self.decoder: Optional[Decoder] = decoder
        self.extra_args = kwargs  # for subclasses to stash configs


    @staticmethod
    def build_encoder(feature_dim: int, latent_dim: int,
                      layers: Optional[LayerList] = None,
                      batch_norm: bool = False,
                      device=None, dtype=None) -> Encoder:
        return Encoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=layers,
            batch_norm=batch_norm,
            device=device, dtype=None
        )

    @staticmethod
    def build_decoder(feature_dim: int, latent_dim: int,
                      layers: Optional[LayerList] = None,
                      batch_norm: bool = False,
                      device=None, dtype=None) -> Decoder:
        return Decoder(
            latent_dim=latent_dim,
            feature_dim=feature_dim,
            layers=layers,
            batch_norm=batch_norm,
            device=device, dtype=None
        )

    def to(self, device=None, dtype=None):
        self.encoder.to(device=device, dtype=dtype)
        self.decoder.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("VAE encoder/decoder not initialized.")
        mu, logvar, z = self.encoder(x)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    def encode(self, x:torch.Tensor):
        """
        Run encoder model and return encoded values
        """
        with torch.no_grad():
            _, _, z = self.encoder(x)
        return z

    @abstractmethod
    def to_dict():
        pass


class InferenceVAE(BaseVAE):
    """Concrete wrapper when no training container is available. Inference only."""
    def fit(self, X, **kwargs):
        raise RuntimeError("This loaded model is inference-only. Use a concrete VAE subclass to train.")


class SimpleEncoder(nn.Module):
    """
    Wrapper nn.Module for encoder. Mainly for passing encoder module to shap.DeepExplainer which does 
    type checking to figure out how to deal with classes
    """
    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        _, _, z = self.encoder(x)
        return z

def _import_obj(dotted: str):
    """Import 'pkg.mod.ClassName' -> object."""
    mod_name, _, attr = dotted.rpartition(".")
    if not mod_name or not attr:
        raise ImportError(f"Invalid dotted path: {dotted}")
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)