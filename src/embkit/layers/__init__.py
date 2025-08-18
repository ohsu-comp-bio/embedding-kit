
"""
Custom layers

"""

from .masked_linear import MaskedLinear
from .layer_info import LayerInfo, convert_activation

"""
A custom module for building layers with masked linear operations and activation functions. This module provides a flexible way to create layers with various configurations of masked linear transformations followed by an optional non-linear activation function.

Classes:
    LayerModule: A custom layer that applies a masked linear transformation followed by an optional non-linear activation function.
"""