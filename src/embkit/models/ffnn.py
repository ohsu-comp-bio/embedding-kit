"""
Feed Forward Neural Network Model
"""

import logging
from typing import Dict, Optional, List, Union
from collections.abc import Callable

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from tqdm.autonotebook import tqdm

from ..layers import LayerInfo, convert_activation
from .. import get_device, dataframe_loader

logger = logging.getLogger(__name__)


class FFNN(nn.Module):
    """
    FeedForward Neural Network
    """

    def __init__(self, input_dim: int, output_dim: int,
                 layers: Optional[List[LayerInfo]] = None,
                 batch_norm: bool = False):
        super().__init__()

        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.history: Dict[str, list] = {"loss": []}

        in_features = input_dim

        # Optional global BN on input
        if batch_norm:
            self.layers.append(nn.BatchNorm1d(input_dim))

        if layers:
            logger.info("Building encoder with %d layers", len(layers))
            for li in layers:
                out_features = li.units

                layer = li.gen_layer(in_features)
                self.layers.append(layer)

                if li.activation is not None:
                    act = convert_activation(li.activation)
                    if act is not None:
                        self.layers.append(act)

                if li.batch_norm:
                    self.layers.append(nn.BatchNorm1d(out_features))

                in_features = out_features
        else:
            pass

        if in_features != self.output_dim:
            raise Exception(f"Layer issue")

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return h
