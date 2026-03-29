"""
A custom module for building layers with masked linear operations and activation functions.
This module provides a flexible way to create layers with various configurations of masked
linear transformations followed by an optional non-linear activation function.

Classes:
    MaskedLinear:  Linear layer whose weight is elementwise-multiplied by a mask at forward time.
    PairwiseComparison: A layer that performs pairwise comparisons between inputs.

Note:
    Layer configuration (previously ``LayerInfo``) now lives under ``embkit.factory.layers.Layer``.
"""

from .masked_linear import MaskedLinear
from .pairwise_comparison_layer import PairwiseComparison
from .tsp import TSPLayer


