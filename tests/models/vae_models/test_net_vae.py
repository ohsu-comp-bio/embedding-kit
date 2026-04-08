import unittest

import torch

import numpy as np
import pandas as pd

from embkit.models.vae.net_vae import NetVAE
from embkit.modules import MaskedLinear
from embkit.losses import bce_with_logits
from embkit.optimize import fit_vae

class TestNetVAE(unittest.TestCase):
    def test_fit_applies_constraint_mask(self):
        df = pd.DataFrame(
            np.random.rand(6, 3),
            columns=["G1", "G2", "G3"],
        )

        latent_index = ["TF1", "TF2"]
        latent_groups = {
            "TF1": ["G1", "G3"],
            "TF2": ["G2"],
        }

        model = NetVAE(features=list(df.columns), latent_groups=latent_groups, latent_index=latent_index)
        fit_vae(
            df,
            model,
            epochs=0,
            batch_size=2,
            lr=1e-3,
            device="cpu",
        )

        constraint = model.encoder.constraint
        self.assertIsNotNone(constraint)

        constraint.set_active(True)
        model.encoder.refresh_mask(device=torch.device("cpu"))

        masked_layers = [m for m in model.encoder.net if isinstance(m, MaskedLinear)]
        self.assertTrue(masked_layers)

        mask = masked_layers[0].mask.cpu().numpy()
        expected = np.array(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(mask, expected)

    def test_masked_edges_remain_zero_during_training(self):
        df = pd.DataFrame(
            np.random.rand(8, 3),
            columns=["G1", "G2", "G3"],
        )

        latent_index = ["TF1", "TF2"]
        latent_groups = {
            "TF1": ["G1"],
            "TF2": ["G2"],
        }

        model = NetVAE(features=list(df.columns), latent_groups=latent_groups, latent_index=latent_index, group_layer_size=[1,1])
        fit_vae(
            df,
            model,
            epochs=0,
            batch_size=4,
            lr=1e-3,
            device="cpu",
        )

        constraint = model.encoder.constraint
        constraint.set_active(True)
        model.encoder.refresh_mask(device=torch.device("cpu"))

        masked_layer = next(m for m in model.encoder.net if isinstance(m, MaskedLinear))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        x = torch.tensor(df.values, dtype=torch.float32)
        mu, logvar, z = model.encoder(x)
        recon = model.decoder(z)
        total, _, _ = bce_with_logits(recon, x, mu, logvar, beta=1.0)
        opt.zero_grad()
        total.backward()

        # Gradients for masked weights should be zero after backward.
        weight = masked_layer.linear.weight
        mask = masked_layer.mask
        self.assertIsNotNone(weight.grad)
        self.assertTrue(torch.all(weight.grad[mask == 0] == 0))

        # After an optimizer step, masked weights should remain unchanged.
        weight_before = weight.detach().clone()
        opt.step()
        weight_after = weight.detach()
        self.assertTrue(torch.all(weight_after[mask == 0] == weight_before[mask == 0]))


if __name__ == "__main__":
    unittest.main()
