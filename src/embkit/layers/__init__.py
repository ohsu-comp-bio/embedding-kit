"""
A custom module for building layers with masked linear operations and activation functions.
This module provides a flexible way to create layers with various configurations of masked 
linear transformations followed by an optional non-linear activation function.

Classes:
    MaskedLinear:  Linear layer whose weight is elementwise-multiplied by a mask at forward time.
    LayerInfo: A data structure to hold information about each layer, including the number of units, 
      activation function, and whether to use batch normalization.
    PairwiseComparison: A layer that performs pairwise comparisons between inputs.
"""

from .masked_linear import MaskedLinear
from .layer_info import LayerInfo, convert_activation
from .constraint_info import ConstraintInfo
from .pairwise_comparison_layer import PairwiseComparison
from .tsp import TSPLayer

def build_layers(sizes, activation="relu", end_activation="relu"):
    """
    Simple layer builder

    end_activation: if the last layer it an output, allow activation to be set to None
    """
    out = []
    for i, s in enumerate(sizes):
        if i < len(sizes)-1:
            l = LayerInfo(s, activation=activation)
        else:
            print(f"end activation{end_activation}")
            l = LayerInfo(s, activation=end_activation)
        out.append(l)
    return out
