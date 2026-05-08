import unittest

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from embkit.optimize import (
    _move_to_device,
    _resolve_phases,
    fit,
    fit_vae,
)
from embkit.losses import bce_with_logits
from embkit.models.vae.vae import VAE


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 1)

    def forward(self, x):
        return self.lin(x)


class TestOptimizeHelpers(unittest.TestCase):
    def test_move_to_device_nested(self):
        x = torch.ones(2)
        nested = {"a": [x, (x,)]}
        out = _move_to_device(nested, torch.device("cpu"))
        self.assertEqual(out["a"][0].device.type, "cpu")
        self.assertEqual(out["a"][1][0].device.type, "cpu")

    def test_resolve_phases(self):
        self.assertEqual(_resolve_phases(epochs=3, beta=0.5, beta_schedule=None), [(0.5, 3)])
        self.assertEqual(_resolve_phases(epochs=3, beta=0.5, beta_schedule=[(0.1, 2)]), [(0.1, 2)])

    def test_fit_tensor_requires_y(self):
        model = TinyModel()
        with self.assertRaises(ValueError):
            fit(model=model, X=torch.randn(3, 2), y=None, epochs=1, progress=False)

    def test_fit_vae_guards(self):
        vae = VAE(features=["G1", "G2"], latent_dim=1)
        df = pd.DataFrame([[0.1, 0.2], [0.2, 0.3]], columns=["G1", "G2"])

        with self.assertRaises(ValueError):
            fit_vae(vae, df, epochs=1, loss=None, progress=False)

        bad_df = pd.DataFrame([[0.1, 0.2]], columns=["X1", "X2"])
        with self.assertRaises(ValueError):
            fit_vae(vae, bad_df, epochs=1, loss=bce_with_logits, progress=False)

    def test_fit_vae_accepts_dataloader(self):
        vae = VAE(features=["G1", "G2"], latent_dim=1)
        x = torch.tensor([[0.1, 0.2], [0.2, 0.3]], dtype=torch.float32)
        loader = DataLoader(TensorDataset(x), batch_size=1, shuffle=False)
        out = fit_vae(vae, loader, epochs=1, loss=bce_with_logits, progress=False)
        self.assertIsInstance(out, float)


if __name__ == "__main__":
    unittest.main()
