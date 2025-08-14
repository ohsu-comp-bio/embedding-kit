from typing import Optional
from torch import nn


class LayerInfo:
    def __init__(self, units: int, *, op: str = "linear",
                 activation: Optional[str] = "relu", batch_norm: bool = False, bias: bool = True):
        self.units = units
        self.op = op
        self.activation = activation
        self.batch_norm = batch_norm
        self.bias = bias


def convert_activation(name: Optional[str]) -> Optional[nn.Module]:
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
