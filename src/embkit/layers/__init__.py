"""
A custom module for building layers with masked linear operations and activation functions.
This module provides a flexible way to create layers with various configurations of masked linear transformations followed by an optional non-linear activation function.

Classes:
    MaskedLinear:  Linear layer whose weight is elementwise-multiplied by a mask at forward time.
    LayerInfo: A data structure to hold information about each layer, including the number of units, activation function, and whether to use batch normalization.
    PairwiseComparison: A layer that performs pairwise comparisons between inputs.
"""

from .masked_linear import MaskedLinear
from .layer_info import LayerInfo, convert_activation
from .pairwise_comparison_layer import PairwiseComparison
