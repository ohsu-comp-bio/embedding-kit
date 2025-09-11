from typing import Optional
from torch import nn


class LayerInfo:
    """
    Layer information for building a neural network layer.
    Holds configuration details for a layer, including the number of units,
    the type of operation (e.g., linear), the activation function, and whether
    to apply batch normalization.
    """

    def __init__(self, units: int, *, op: str = "linear",
                 activation: Optional[str] = "relu", batch_norm: bool = False, bias: bool = True):
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
        self.bias = bias


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
