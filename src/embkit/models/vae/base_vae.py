"""
Base VAE class
"""
from typing import Type, Any, List, Optional, Dict, overload, TypeVar, Union
from abc import ABC, abstractmethod
from pathlib import Path
import json
import logging
import pandas as pd
import numpy as np
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
            device=device, dtype=dtype
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
            device=device, dtype=dtype
        )

    def to(self, device=None, dtype=None):
        return super().to(device=device, dtype=dtype)

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
    def to_dict(self) -> Dict[str, Any]:
        pass

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
            mu, logvar, z = self.encoder(dummy_input)
            recon = self.decoder(z) if self.decoder is not None else None

        # Check for latent collapse (dead units)
        variances = torch.var(mu, dim=0).cpu().numpy()
        dead_units = int(np.sum(variances < 1e-6))
        
        latent_dim = mu.shape[1]
        
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
