import unittest

import numpy as np
import pandas as pd
import torch

from embkit import optimize
from embkit.models.vae.rna_vae import RNAVAE


class TestRNAVAE(unittest.TestCase):
    def test_fit_smoke(self):
        df = pd.DataFrame(
            np.random.rand(8, 6).astype(np.float32),
            columns=[f"G{i}" for i in range(6)],
        )

        model = RNAVAE(features=list(df.columns), latent_dim=3)
        history = optimize.fit_vae(
            model,
            df,
            epochs=1,
            batch_size=4,
            # kappa=1.0,
            # early_stopping_patience=2,
            device=torch.device("cpu"),
            progress=False,
        )

        self.assertIn("loss", history)
        self.assertGreaterEqual(len(history["loss"]), 1)
        self.assertEqual(model.latent_dim, 3)

    def test_decoder_output_is_sigmoid_layer(self):
        # includes_output=True: the sigmoid Layer is the output, no Linear appended after it
        model = RNAVAE(features=[f"G{i}" for i in range(6)], latent_dim=3)
        net = list(model.decoder.net)
        self.assertEqual(len(net), 2)
        self.assertIsInstance(net[0], torch.nn.Linear)
        self.assertIsInstance(net[1], torch.nn.Sigmoid)
        self.assertEqual(net[0].out_features, 6)


if __name__ == "__main__":
    unittest.main()
