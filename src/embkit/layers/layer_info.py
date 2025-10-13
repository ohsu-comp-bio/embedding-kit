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
        self.constraint = None
    
    def gen_layer(self, in_features):
        out_features = self.units
        if self.op == "masked_linear":
            init_mask = None
            if self.constraint is not None:
                init_mask = torch.tensor(self.constraint.gen_mask(), dtype=torch.float32)
            return MaskedLinear(in_features, out_features, bias=self.bias, mask=init_mask)
        elif self.op == "linear":
            return nn.Linear(in_features, out_features, bias=self.bias)
        raise ValueError(f"Unknown LayerInfo.op '{self.op}'")




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
