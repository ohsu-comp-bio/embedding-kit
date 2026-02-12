"""
Feed Forward Neural Network Model
"""

import logging
from typing import Dict, Optional, List, Union
from collections.abc import Callable

from ..factory.mapping import Sequential, Linear, BatchNorm1d
from ..factory.layers import LayerArray

from torch import nn


logger = logging.getLogger(__name__)


class FFNN(nn.Module):
    """
    FeedForward Neural Network Constructor
    """

    def __init__(self, input_dim: int, output_dim: int,
                 layers: Optional[LayerArray] = None,
                 batch_norm: bool = False):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self._params = {
            "input_dim": input_dim,
            "output_dim": output_dim, 
            "layers": layers,
            "batch_norm": batch_norm,
        }

        if layers:
            logger.info("Building encoder with %d layers", len(layers))
            self.layers = layers.build(input_dim, output_dim)
        else:
            self.layers = Sequential(
                Linear(self.input_dim, self.output_dim),
            )

        # Optional global BN on input
        if batch_norm:
            self.layers.insert(0, BatchNorm1d(input_dim))

    def forward(self, x):
        return self.layers(x)
