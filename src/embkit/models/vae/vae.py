import logging
from typing import List, Optional
import torch

from ...factory.mapping import nn_module, get_class_name
from ...factory.layers import Layer, LayerList
from ...factory import build

from typing import Type, Any, List, Optional, Dict, overload, TypeVar, Union, NamedTuple
import logging
import numpy as np
from torch import nn
import torch
from .encoder import Encoder, EncoderOutput
from .decoder import Decoder
from ...factory.layers import Layer, LayerList
from ... import get_device
import importlib
import inspect

logger = logging.getLogger(__name__)
T = TypeVar("T")


class VAEOutput(NamedTuple):
    """Named output of :meth:`BaseVAE.forward`."""
    recon: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor


class VAE(nn.Module):
    """
    Minimal VAE wrapper to hold encoder/decoder and provide forward().
    """

    def __init__(self, encoder: Encoder = None, decoder: Decoder = None, **kwargs):
        super().__init__()
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.extra_args = kwargs  # for subclasses to stash configs

    def to(self, device=None, dtype=None):
        return super().to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
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
        return VAE(
            encoder=build( data["encoder"]), 
            decoder=build( data["decoder"]),
            **data["extra_args"]
        )

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Perform a health audit of the model's architecture, weights, and training history.
        
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

        # 2. History audit (training sanity check)
        history = getattr(self, "history", None)
        if history and "loss" in history and len(history["loss"]) > 0:
            losses = history["loss"]
            if any(np.isnan(losses)):
                report["healthy"] = False
                report["issues"].append("Training history contains NaNs. The model may be unstable.")
            
            # Check if loss actually improved
            initial_loss = losses[0]
            final_loss = losses[-1]
            if not (final_loss < initial_loss):
                report["healthy"] = False
                report["issues"].append(f"Model failed to improve during training (Loss started at {initial_loss:.4f} and ended at {final_loss:.4f}).")
            
            report["history_summary"] = {
                "epochs": len(losses),
                "initial_loss": float(initial_loss),
                "final_loss": float(final_loss),
                "improvement": float(initial_loss - final_loss)
            }
        else:
            report["healthy"] = False
            report["issues"].append("Training history missing; cannot assess learning trend.")

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
    Concrete VAE that composes the modular Encoder/Decoder from BaseVAE

    BaseVAE.forward(x) returns: recon, mu, logvar, z
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
        encoder = self.build_encoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=encoder_layers,
            batch_norm=batch_norm,
            sampling=sampling,
            device=device, dtype=dtype
        )
        decoder = self.build_decoder(
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
            atent_dim = latent_dim
        )
