"""
Variational Autoencoder (VAE) base model and standard variant.

A VAE is embkit's core model for turning a high-dimensional feature matrix
(e.g. per-sample gene expression) into a compact latent embedding. Each sample
is mapped to a latent distribution (``mu``/``logvar``); a single vector is
sampled from it (``z``) and decoded back to the input. The balance between
faithful reconstruction and a well-shaped latent space is what produces useful
embeddings — one sample per point that preserves the structure of the original
data.

This module defines:

- :class:`VAEOutput` — the ``(recon, mu, logvar, z)`` bundle returned by
  ``forward``.
- :class:`VAE` — the minimal encoder/decoder container and its ``encode``/
  serialize helpers that all variants share.
- :class:`BaseVAE` — the standard fully-connected embedding VAE, built from
  layer configs; the default model produced by ``embkit model train``.
"""
import logging
from typing import Any, List, Optional, Dict, TypeVar, NamedTuple

import numpy as np
from torch import nn
import torch

from .encoder import Encoder
from .decoder import Decoder
from ...factory.layers import Layer, LayerList
from ...factory.mapping import nn_module
from ...factory import build


logger = logging.getLogger(__name__)
T = TypeVar("T")


class VAEOutput(NamedTuple):
    """Named output of :meth:`BaseVAE.forward`."""
    recon: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor


@nn_module
class VAE(nn.Module):
    """
    Base Variational Autoencoder that pairs an encoder with a decoder.

    This is the abstract building block every embkit embedding model inherits
    from. It owns no architecture itself — just the ``encoder``/``decoder`` pair
    and the shared behavior all variants rely on: the :meth:`forward` pass
    (encode -> sample -> decode), :meth:`encode` for producing a stable
    per-sample embedding (``mu``), and the :meth:`to_dict`/:meth:`from_dict`
    contract that lets a trained model be saved and rebuilt through the factory.

    Concrete models (e.g. :class:`BaseVAE` for plain expression data, or
    :class:`~embkit.models.vae.net_vae.NetVAE` for pathway-constrained data)
    configure the encoder/decoder to fit their use case.
    """

    def __init__(self, encoder: Encoder = None, decoder: Decoder = None, **kwargs):
        """
        Initialize a VAE from an encoder and a decoder.

        A VAE (Variational Autoencoder) is the core embedding model in embkit:
        it learns a compressed latent representation of each sample (``mu``/
        ``logvar``) and can reconstruct the input from it. This base class just
        pairs an :class:`~embkit.models.vae.encoder.Encoder` with a
        :class:`~embkit.models.vae.decoder.Decoder`; the concrete variants
        (e.g. :class:`~embkit.models.vae.net_vae.NetVAE`) configure those parts
        for a specific use case.

        Parameters:
            encoder: The encoder mapping input features to the latent space.
            decoder: The decoder mapping the latent space back to input features.
            **kwargs: Extra configuration stashed on :attr:`extra_args` so
                subclasses can serialize and rebuild their own settings.
        """
        super().__init__()
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.extra_args = kwargs  # for subclasses to stash configs

    def to(self, device=None, dtype=None):
        """
        Move the model's parameters to the given device/dtype (delegates to ``nn.Module``).
        """
        return super().to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        """
        Run one pass through the encoder and decoder.

        The input ``x`` (a batch of samples) is encoded into the latent
        statistics (``mu``, ``logvar``) and a sampled latent vector ``z``, then
        decoded back to the input space. Returns a :class:`VAEOutput` bundling
        the reconstruction and all three latent quantities, which is what the
        training losses consume.

        Parameters:
            x: Input batch tensor of shape ``(batch_size, features)``.

        Returns:
            VAEOutput: Tuple of ``(recon, mu, logvar, z)``.

        Raises:
            RuntimeError: If the encoder or decoder has not been initialized.
        """
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("VAE encoder/decoder not initialized.")
        encoder_out = self.encoder(x)
        recon = self.decoder(encoder_out.z)
        return VAEOutput(recon=recon, mu=encoder_out.mu, logvar=encoder_out.logvar, z=encoder_out.z)

    def encode(self, x:torch.Tensor):
        """
        Run encoder model and return the latent mean (mu) for stable embeddings.
        """
        with torch.no_grad():
            encoder_out = self.encoder(x)
        return encoder_out.mu

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the VAE components and constructor arguments."""
        return {
            "encoder": self.encoder.to_dict(),
            "decoder": self.decoder.to_dict(),
            "extra_args": self.extra_args,
        }

    @classmethod
    def from_dict(cls, data):
        """
        Rebuild a VAE from a serialized description (inverse of :meth:`to_dict`).

        Handles two formats: the current form (``encoder``/``decoder``
        descriptions resolved via the factory :func:`~embkit.factory.core.build`)
        and a legacy flat format (``encoder_layers``/``decoder_layers``), which is
        upgraded into a :class:`~embkit.models.vae.vae.BaseVAE`. This is the path
        that lets a trained model be restored from its ``.model`` file.

        Parameters:
            data: A description dict produced by :meth:`to_dict`.

        Returns:
            A reconstructed ``VAE`` (or ``BaseVAE`` for legacy descriptions).
        """
        if "encoder" not in data: # Legacy format 
            def build_layer_list(raw):
                return LayerList([Layer.from_dict(li) for li in raw]) if raw else None
            return BaseVAE(
                features=data["features"],
                latent_dim=data["latent_dim"],
                encoder_layers=build_layer_list(data.get("encoder_layers")),
                decoder_layers=build_layer_list(data.get("decoder_layers")),
                batch_norm=data.get("batch_norm", False),
                sampling=data.get("sampling", True),
            )
        return VAE(
            encoder=build(data["encoder"]),
            decoder=build(data["decoder"]),
            **data["extra_args"]
        )


    def verify_integrity(self) -> Dict[str, Any]:
        """
        Perform a health audit of the model's architecture, weights.
        
        Returns:
            A dictionary containing the audit results.
        """
        report = {
            "model_type": self.__class__.__name__,
            "features_count": len(self.features),
            "healthy": True,
            "issues": []
        }

        # 1. Parameter Health (NaNs/Infs) & Weight Norm Audit
        weight_norms = []
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                report["healthy"] = False
                report["issues"].append(f"NaN values detected in parameter: {name}")
            if torch.isinf(param).any():
                report["healthy"] = False
                report["issues"].append(f"Infinite values detected in parameter: {name}")
            
            # Audit weight magnitude
            if "weight" in name:
                weight_norms.append(torch.norm(param).item())

        if weight_norms:
            report["weight_norm_avg"] = float(np.mean(weight_norms))
            report["weight_norm_max"] = float(np.max(weight_norms))
            if report["weight_norm_max"] > 1000: # Paranoid threshold for gradient explosion
                report["healthy"] = False
                report["issues"].append(f"Extremely high weight norm detected ({report['weight_norm_max']:.2f}). Potential gradient explosion.")

        # 3. Mandatory Deep Audit (Manifold health)
        if self.encoder is not None:
            deep_audit = self._deep_integrity_check()
            report["deep_audit"] = deep_audit
            if deep_audit.get("collapsed", False):
                report["healthy"] = False
                report["issues"].append("Latent space collapse detected (dead units).")
            if deep_audit.get("reconstruction_mse", 0) > 100: # High MSE for normalized data
                 report["healthy"] = False
                 report["issues"].append(f"Extremely high reconstruction MSE ({deep_audit['reconstruction_mse']:.4f}).")

        return report

    def refresh_masks(self, device: Optional[torch.device] = None) -> None:
        """
        Iterate through all modules and refresh masks for any MaskedLinear layers.
        
        Args:
            device: The device to move the mask tensors to. If None, uses the model's current device.
        """
        if device is None:
            # Try to infer device from parameters
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        for module in self.modules():
            if hasattr(module, "refresh_mask"): # Check for MaskedLinear or custom refreshers
                try:
                    module.refresh_mask(device)
                except Exception as e:
                    logger.debug(f"Failed to refresh mask on {module}: {e}")

    def _deep_integrity_check(self) -> Dict[str, Any]:
        """Internal helper for deep integrity checks involving forward passes."""
        # Using a small batch of random data for generic VAE checks
        self.eval()
        device = next(self.parameters()).device
        dummy_input = torch.randn(100, len(self.features), device=device)
        
        with torch.no_grad():
            res = self.encoder(dummy_input)
            recon = self.decoder(res.mu) if self.decoder is not None else None

        # Check for latent collapse (dead units)
        variances = torch.var(res.mu, dim=0).cpu().numpy()
        dead_units = int(np.sum(variances < 1e-6))
        
        latent_dim = res.mu.shape[1]
        
        results = {
            "latent_dim": latent_dim,
            "dead_units": dead_units,
            "collapsed": (dead_units == latent_dim),
            "latent_variance_mean": float(np.mean(variances)),
            "latent_variance_max": float(np.max(variances))
        }
        
        if recon is not None:
            mse = torch.mean((dummy_input - recon)**2).item()
            results["reconstruction_mse"] = float(mse)
            
        return results



@nn_module
class BaseVAE(VAE):
    """
    The standard, fully-connected embedding VAE.

    This is the model you get when you train an embedding from a plain feature
    matrix (e.g. gene expression) with no structural constraints. It wires the
    generic :class:`~embkit.models.vae.encoder.Encoder` and
    :class:`~embkit.models.vae.decoder.Decoder` together from layer
    configurations, and is the default target of ``embkit model train``.

    A call — ``model(x)`` — returns a :class:`VAEOutput` of
    ``(recon, mu, logvar, z)``: the reconstruction, the latent mean, the latent
    log-variance, and the sampled latent vector. :meth:`encode` returns ``mu``
    directly, which is the stable per-sample embedding vector used downstream.

    Subclasses like :class:`~embkit.models.vae.net_vae.NetVAE` add structure
    (e.g. pathway-based connection constraints) on top of this base model.
    """

    def __init__(
            self,
            features: List[str],
            latent_dim: Optional[int] = None,
            encoder_layers: Optional[LayerList] = None,
            decoder_layers: Optional[LayerList] = None,
            batch_norm: bool = False,
            sampling: bool = True,
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
            sampling: enable reparameterization sampling during the forward pass (VAE).
                      Set to False for a deterministic autoencoder (mu is used as z).
            device: torch device used for module initialization
            dtype: torch dtype used for module initialization
        """
        if encoder_layers is None:
            encoder_layers = LayerList()
        elif isinstance(encoder_layers, list):
            encoder_layers = LayerList(encoder_layers)
        if decoder_layers is None:
            decoder_layers = LayerList()
        elif isinstance(decoder_layers, list):
            decoder_layers = LayerList(decoder_layers)

        feature_dim = len(features)

        if latent_dim is None:
            raise ValueError("latent_dim is required when encoder/decoder are not provided.")
        encoder = Encoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=encoder_layers,
            batch_norm=batch_norm,
            sampling=sampling,
            device=device, dtype=dtype
        )
        decoder = Decoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=decoder_layers,
            device=device, dtype=dtype
        )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            encoder_layers_cfg = encoder_layers,
            decoder_layers_cfg = decoder_layers,
            batch_norm = batch_norm,
            sampling = sampling,
            latent_dim = latent_dim, 
            features=features
        )
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.batch_norm = batch_norm
        self.sampling = sampling
        self.latent_dim = latent_dim
        self.features = features

    def to_dict(self):
        """
        Serialize the model description (architecture, not weights).

        Captures the feature list, latent dimension, and the encoder/decoder
        layer configurations so that :meth:`from_dict` can rebuild an
        equivalent model. Learned weights are serialized separately by
        :func:`embkit.factory.core.save` from the state dict.
        """
        return {
            "features": self.features,
            "latent_dim": self.latent_dim,
            "encoder_layers": self.encoder_layers.to_dict() if self.encoder_layers else None,
            "decoder_layers": self.decoder_layers.to_dict() if self.decoder_layers else None,
            "batch_norm": self.batch_norm,
            "sampling": self.sampling,
        }

    @classmethod
    def from_dict(cls, data):
        """
        Rebuild a :class:`BaseVAE` from its :meth:`to_dict` description.

        Resolves the encoder/decoder layer lists through the factory
        :func:`~embkit.factory.core.build` and reconstructs an equivalent model.
        Weight tensors must then be loaded from the state dict (see
        :func:`embkit.factory.core.load`) to recover a trained model.
        """
        return BaseVAE(
            features=data["features"],
            latent_dim=data["latent_dim"],
            encoder_layers=build(data["encoder_layers"]) if data["encoder_layers"] else None,
            decoder_layers=build(data["decoder_layers"]) if data["decoder_layers"] else None
        )
