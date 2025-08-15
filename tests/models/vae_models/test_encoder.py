import unittest
import torch
from torch import nn

from embkit.models.vae_model.encoder import Encoder
from embkit.layers import LayerInfo, MaskedLinear


class TestEncoder(unittest.TestCase):
    def test_auto_projection_when_no_layers(self):
        enc = Encoder(feature_dim=6, latent_dim=3, layers=None, batch_norm=False)
        # Should create auto projection + latent heads
        self.assertTrue(hasattr(enc, "z_mean") and isinstance(enc.z_mean, nn.Linear))
        self.assertTrue(hasattr(enc, "z_log_var") and isinstance(enc.z_log_var, nn.Linear))

        x = torch.randn(4, 6)
        mu, logvar, z = enc(x)
        self.assertEqual(tuple(mu.shape), (4, 3))
        self.assertEqual(tuple(logvar.shape), (4, 3))
        self.assertEqual(tuple(z.shape), (4, 3))

    def test_layers_provided_require_final_equals_latent_dim(self):
        layers = [LayerInfo(units=8), LayerInfo(units=6)]
        # final width=6 != latent_dim=4 -> should raise
        with self.assertRaises(ValueError):
            Encoder(feature_dim=5, latent_dim=4, layers=layers, batch_norm=False)

    def test_layers_provided_with_matching_latent_dim(self):
        layers = [LayerInfo(units=8), LayerInfo(units=4)]
        enc = Encoder(feature_dim=5, latent_dim=4, layers=layers, batch_norm=False)

        # latent heads should exist
        self.assertIsInstance(enc.z_mean, nn.Linear)
        self.assertIsInstance(enc.z_log_var, nn.Linear)

        x = torch.randn(3, 5)
        mu, logvar, z = enc(x)
        self.assertEqual(tuple(z.shape), (3, 4))

    def test_encoder_with_masked_linear(self):
        layers = [LayerInfo(units=7, op="masked_linear", activation=None)]
        enc = Encoder(feature_dim=7, latent_dim=7, layers=layers, batch_norm=False)

        # masked layer present
        self.assertTrue(any(isinstance(m, MaskedLinear) for m in enc.net))

        x = torch.randn(2, 7)
        mu, logvar, z = enc(x)
        self.assertEqual(tuple(mu.shape), (2, 7))


if __name__ == "__main__":
    unittest.main()