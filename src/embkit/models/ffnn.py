"""
Feed Forward Neural Network Model
"""

import logging
from typing import Dict, Optional, List, Union, Any
import numpy as np
from collections.abc import Callable

from ..factory.mapping import Sequential, Linear, BatchNorm1d
from ..factory.layers import LayerList

from torch import nn


from .. import factory
import torch

logger = logging.getLogger(__name__)


@factory.nn_module
class FFNN(nn.Module):
    """
    FeedForward Neural Network Constructor
    """

    def __init__(self, input_dim: int, output_dim: int,
                 layers: Optional[LayerList] = None,
                 batch_norm: bool = False):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self._params = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "layers": layers,
            "batch_norm": batch_norm,
        }

        if layers:
            logger.info("Building encoder with %d layers", len(layers))
            self.layers = layers.build(input_dim, output_dim)
        else:
            self.layers = Sequential(
                Linear(self.input_dim, self.output_dim),
            )

        # Optional global BN on input
        if batch_norm:
            self.layers.insert(0, BatchNorm1d(input_dim))

    def forward(self, x):
        return self.layers(x)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "layers": self._params["layers"],
            "batch_norm": self._params["batch_norm"],
            "history": getattr(self, "history", {}) or {}
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        model = cls(
            input_dim=d["input_dim"],
            output_dim=d["output_dim"],
            layers=d.get("layers"),
            batch_norm=d.get("batch_norm", False)
        )
        model.history = d.get("history") or {}
        return model

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Perform a PARANOID health audit of the FFNN architecture, weights, and history.
        """
        report = {
            "model_type": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "healthy": True,
            "issues": []
        }

        # 1. Parameter Health (NaNs/Infs) & Weight Magnitude
        weight_norms = []
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                report["healthy"] = False
                report["issues"].append(f"NaN values detected in parameter: {name}")
            if torch.isinf(param).any():
                report["healthy"] = False
                report["issues"].append(f"Infinite values detected in parameter: {name}")
            if "weight" in name:
                weight_norms.append(torch.norm(param).item())

        if weight_norms:
             report["weight_norm_max"] = float(np.max(weight_norms))

        # 2. History Audit (Authenticity check)
        history = getattr(self, "history", None)
        if history and "loss" in history and len(history["loss"]) > 0:
            losses = history["loss"]
            if any(np.isnan(losses)):
                report["healthy"] = False
                report["issues"].append("Training history contains NaNs.")
            if not (losses[-1] < losses[0]):
                report["healthy"] = False
                report["issues"].append(f"FFNN failed to improve during training (Loss: {losses[0]:.4f} -> {losses[-1]:.4f}).")
        else:
            report["issues"].append("No training history found for FFNN.")

        # 3. Mandatory Forward Pass
        self.eval()
        device = next(self.parameters()).device
        dummy_input = torch.randn(100, self.input_dim, device=device)
        with torch.no_grad():
            output = self(dummy_input)
            if torch.var(output) < 1e-6:
                report["healthy"] = False
                report["issues"].append("Output layer has near-zero variance (potential activation collapse).")

        return report
