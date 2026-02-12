"""
LayerInfo - Layer Build description
"""
from typing import Optional, List, Tuple, Any
import torch
from torch import nn
from ..layers import MaskedLinear
from .mapping import Linear



class Layer:
    """
    Layer information for building a neural network layer.
    Holds configuration details for a layer, including the number of units,
    the type of operation (e.g., linear), the activation function, and whether
    to apply batch normalization.
    """

    def __init__(self, units: int, *, op: str = "linear",
                 activation: Optional[str] = "relu", 
                 # constraint: Optional[ConstraintInfo] = None,
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
        #self.constraint = constraint
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
            return Linear(in_features, out_features, bias=self.bias, device=device)
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

    @classmethod
    def from_dict(d: dict) -> "Layer":
        c = d.get("constraint")
        # constraint = ConstraintInfo.from_dict(c) if c else None
        return Layer(
            units=int(d.get("units", d.get("size"))),  # tolerate old files that used "size"
            op=d.get("op", "linear"),
            activation=d.get("activation", "relu"),
            batch_norm=bool(d.get("batch_norm", False)),
            bias=bool(d.get("bias", True)),
            # constraint=constraint,
        )


class LayerArray:
    def __init__(self, layers: Optional[List[Layer]] = None):
        self.layers = layers

    def build(self, input_dim, output_dim) -> nn.Module:
        if not self.layers:
            return Linear(in_features=input_dim, out_features=output_dim)
        
        cur_dim = input_dim
        layers = []
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.append( layer.gen_layer(cur_dim) )
                cur_dim = layer.units
            elif isinstance(layer, int):
                layer.append( layer.gen_layer(Linear(in_features=cur_dim, out_features=layer)) )
                cur_dim = layer.units
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")

        layers.append( Linear(in_features=cur_dim, out_features=output_dim) )
        return nn.Sequential(*layers)
