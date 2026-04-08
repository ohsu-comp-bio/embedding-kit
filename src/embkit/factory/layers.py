"""
LayerInfo - Layer Build description
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, List, Literal, Dict, Any
import torch
from torch import nn
from ..modules import MaskedLinear
from .mapping import Linear, Sequential, get_activation


class ConstraintInfo(ABC):
    """Abstract interface for constraints that produce masked-linear connectivity."""

    @abstractmethod
    def gen_mask(self, in_features: int, out_features: int) -> np.ndarray:
        """Generate a mask with shape ``(out_features, in_features)``."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize constraint configuration into a JSON-compatible dict."""

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ConstraintInfo":
        """Deserialize constraint configuration from a dict."""

class Layer:
    """
    Layer information for building a neural network layer.
    Holds configuration details for a layer, including the number of units,
    the type of operation (e.g., linear), the activation function, and whether
    to apply batch normalization.
    """

    def __init__(self, units: int, *, op: str = "linear",
                 activation: Optional[str] = "relu",
                 constraint: Optional[ConstraintInfo] = None,
                 batch_norm: bool = False, bias: bool = True):
        """
        Initialize LayerInfo with specified parameters.
        Args:
            units (int): Number of units in the layer.
            op (str): Type of operation, default is "linear".
            activation (Optional[str]): Activation function to use, default is "relu".
            batch_norm (bool): Whether to apply batch normalization, default is False.
            bias (bool): Whether to include a bias term in the layer, default is True.

        Raises:
            ValueError: If the specified operation is not supported.
        """
        self.units = units
        self.op = op
        self.activation = activation
        self.batch_norm = batch_norm
        self.constraint = constraint
        self.bias = bias
    
    def gen_layer(self, in_features: int, device=None, dtype=None) -> List[nn.Module]:
        out_features = self.units
        layers = []
        if self.op == "masked_linear":
            init_mask = None
            if self.constraint is not None:
                m = self.constraint.gen_mask(in_features, out_features)
                # Expect (out_features, in_features)
                if m.shape != (out_features, in_features):
                    raise ValueError(
                        f"Constraint mask shape {m.shape} does not match "
                        f"(units, in_features)=({out_features}, {in_features})."
                    )
                init_mask = torch.as_tensor(m, dtype=torch.float32, device=device)
            layers.append(MaskedLinear(in_features, out_features, bias=self.bias, mask=init_mask, device=device, dtype=dtype))
        elif self.op ==  "linear":
            layers.append(Linear(in_features, out_features, bias=self.bias, device=device, dtype=dtype))
        else:
            raise ValueError(f"Unknown Layer.op '{self.op}'")
        
        if self.activation is not None:
            act = get_activation(self.activation)
            if act is not None:
                layers.append(act())
        if self.batch_norm:
            layers.append(nn.BatchNorm1d(out_features, device=device, dtype=dtype))
        return layers

    def to_dict(self) -> dict:
        return {
            "units": self.units,
            "op": self.op,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
            "bias": self.bias,
            "constraint": (self.constraint.to_dict() if self.constraint else None),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Layer":
        c = d.get("constraint", None)
        constraint = ConstraintInfo.from_dict(c) if c is not None else None
        return Layer(
            units=int(d.get("units", d.get("size"))),  # tolerate old files that used "size"
            op=d.get("op", "linear"),
            activation=d.get("activation", "relu"),
            batch_norm=bool(d.get("batch_norm", False)),
            bias=bool(d.get("bias", True)),
            constraint=constraint,
        )

class LayerList:
    def __init__(self, layers: Optional[List[Layer]] = None, activation="relu", end_activation="relu"):
        new_layers = []
        if layers is not None:
            for i, l in enumerate(layers):
                if isinstance(l, int):
                    if i < len(layers)-1:
                        new_layers.append( Layer(l, activation=activation) )
                    else:
                        new_layers.append( Layer(l, activation=end_activation))
                else:
                    new_layers.append(l)
        self.layers = new_layers

    def build(self, input_dim:int, output_dim:int, device=None, dtype=None) -> nn.Module:
        if not self.layers:
            return Linear(in_features=input_dim, out_features=output_dim, device=device, dtype=dtype)
        
        cur_dim = input_dim
        layers = []
        for layer in self.layers:
            if isinstance(layer, Layer):
                layers.extend(layer.gen_layer(cur_dim, device=device, dtype=dtype))
                cur_dim = layer.units
            elif isinstance(layer, int):
                # Fallback handling in case raw integers are present in self.layers.
                # Treat the integer as the number of units for a Linear layer.
                layers.append(Linear(in_features=cur_dim, out_features=layer, device=device, dtype=dtype))
                cur_dim = layer
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")

        layers.append(Linear(in_features=cur_dim, out_features=output_dim, device=device, dtype=dtype))
        return Sequential(*layers)

    def __str__(self):
        o = []
        for i in self.layers:
            if isinstance(i, Layer):
                o.append(i.to_dict())
            else:
                o.append(str(i))
        return str(o)

    def __len__(self):
        return len(self.layers)
    
    def __iter__(self):
        return iter(self.layers)