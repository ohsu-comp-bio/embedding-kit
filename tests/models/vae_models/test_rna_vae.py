import unittest

import numpy as np
import pandas as pd
import torch

from embkit.models.vae.rna_vae import RNAVAE


class TestRNAVAE(unittest.TestCase):
    def test_fit_smoke(self):
        df = pd.DataFrame(
            np.random.rand(8, 6).astype(np.float32),
            columns=[f"G{i}" for i in range(6)],
        )

        model = RNAVAE(features=list(df.columns), latent_dim=3, lr=1e-3)
        history = model.fit(
            df,
            epochs=1,
            batch_size=4,
            kappa=1.0,
            early_stopping_patience=2,
            device=torch.device("cpu"),
            progress=False,
        )

        self.assertIn("loss", history)
        self.assertGreaterEqual(len(history["loss"]), 1)
        self.assertEqual(model.latent_dim, 3)


if __name__ == "__main__":
    unittest.main()
