
from .. import factory

import torch
from torch import nn

@factory.nn_module
class PairPredictor(nn.Module):
    def __init__(self, item_module, context_module, item_dim, context_dim, learning_dim = 256, device=None, dtype=None):
        super().__init__()
        self.item_module = item_module
        self.context_module = context_module
        self.learning_dim = learning_dim
        self.item_dim = item_dim
        self.context_dim = context_dim
        input_dim = item_dim + item_dim + context_dim

        self.pair_predict = nn.Sequential(
            nn.BatchNorm1d( input_dim, dtype=dtype),
            nn.Linear( input_dim, learning_dim, dtype=dtype ), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( learning_dim, learning_dim, dtype=dtype), nn.ReLU(),
            # nn.Linear( learning_dim, learning_dim, dtype=dtype), nn.ReLU(),
            nn.Linear( learning_dim, 1, dtype=dtype)
        ).to(device)

    def forward(self, x):
        item1, item2, context = x
        h1 = self.item_module(item1)
        h2 = self.item_module(item2)
        c = self.context_module(context)
        pin = torch.cat([h1,h2,c], dim=-1)
        return self.pair_predict(pin)

    def to_dict(self):
        return {
            "item_module": self.item_module.to_dict(),
            "context_module": self.context_module.to_dict(),
            "item_dim": self.item_dim,
            "context_dim": self.context_dim,
            "learning_dim": self.learning_dim
        }

    @classmethod
    def from_dict(cls, params):
        return PairPredictor(
            item_module=factory.build(params["item_module"]),
            context_module=factory.build(params["context_module"]),
            item_dim=params["item_dim"],
            context_dim=params["context_dim"],
            learning_dim=params["learning_dim"]
        )
