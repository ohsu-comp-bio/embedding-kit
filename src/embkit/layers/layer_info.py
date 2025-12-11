"""
LayerInfo - Layer Build description
"""
from typing import Optional, List, Tuple, Any
from torch import nn
import torch
from .masked_linear import MaskedLinear
from .constraint_info import ConstraintInfo

class LayerInfo:
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
    
    def gen_layer(self, in_features: int, device=None):
        out_features = self.units
        if self.op == "masked_linear":
            init_mask = None
            if self.constraint is not None:
                m = self.constraint.gen_mask()
                # Expect (out_features, in_features)
                if m.shape != (out_features, in_features):
                    raise ValueError(
                        f"Constraint mask shape {m.shape} does not match "
                        f"(units, in_features)=({out_features}, {in_features})."
                    )
                init_mask = torch.as_tensor(m, dtype=torch.float32, device=device)
            return MaskedLinear(in_features, out_features, bias=self.bias, mask=init_mask, device=device)
        elif self.op == "linear":
            return nn.Linear(in_features, out_features, bias=self.bias, device=device)
        raise ValueError(f"Unknown LayerInfo.op '{self.op}'")

    def to_dict(self) -> dict:
        return {
            "units": self.units,
            "op": self.op,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
            "bias": self.bias,
            "constraint": (self.constraint.to_dict() if self.constraint else None),
        }

    @staticmethod
    def from_dict(d: dict) -> "LayerInfo":
        c = d.get("constraint")
        constraint = ConstraintInfo.from_dict(c) if c else None
        return LayerInfo(
            units=int(d.get("units", d.get("size"))),  # tolerate old files that used "size"
            op=d.get("op", "linear"),
            activation=d.get("activation", "relu"),
            batch_norm=bool(d.get("batch_norm", False)),
            bias=bool(d.get("bias", True)),
            constraint=constraint,
        )




def convert_activation(name: Optional[str]) -> Optional[nn.Module]:
    """
    Convert a string name to a PyTorch activation function module.
    Args:
        name (Optional[str]): Name of the activation function (e.g., "relu", "
    "tanh", "sigmoid", etc.). If None or empty, returns None.


    Returns:
        Optional[nn.Module]: Corresponding PyTorch activation function module or None if not found.
    """
    if not name:
        return None
    name = name.lower()
    return {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        None: None,
    }.get(name, None)
