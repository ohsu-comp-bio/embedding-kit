import unittest
import torch
from torch import nn

from embkit.models.vae.encoder import Encoder
from embkit.factory.layers import Layer, LayerList
from embkit.modules import MaskedLinear
from embkit.constraints import NetworkConstraint


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

    def test_layers_provided_with_matching_latent_dim(self):
        layers = LayerList([Layer(units=8), Layer(units=4)])
        enc = Encoder(feature_dim=5, latent_dim=4, layers=layers, batch_norm=False)

        # latent heads should exist
        self.assertIsInstance(enc.z_mean, nn.Linear)
        self.assertIsInstance(enc.z_log_var, nn.Linear)

        x = torch.randn(3, 5)
        mu, logvar, z = enc(x)
        self.assertEqual(tuple(z.shape), (3, 4))

    def test_encoder_with_masked_linear(self):
        layers = LayerList([Layer(units=7, op="masked_linear", activation=None)])
        enc = Encoder(feature_dim=7, latent_dim=7, layers=layers, batch_norm=False)

        # masked layer present
        self.assertTrue(any(isinstance(m, MaskedLinear) for m in enc.net))

        x = torch.randn(2, 7)
        mu, logvar, z = enc(x)
        self.assertEqual(tuple(mu.shape), (2, 7))

    def test_encoder_initializes_with_constraint_and_sets_mask(self):
        feature_index = ["f1", "f2"]
        latent_index = ["z1", "z2"]  # updated to 2 rows
        latent_membership = {
            "z1": ["f1"],
            "z2": ["f2"]
        }
        constraint = NetworkConstraint(feature_index, latent_index, latent_membership)

        layers = LayerList([Layer(units=2, op="masked_linear")])  # matches latent_index length
        enc = Encoder(feature_dim=2, latent_dim=2, layers=layers, constraint=constraint)

        # Should be able to call refresh_mask without error
        enc.refresh_mask(device=torch.device("cpu"))
        self.assertTrue(any(isinstance(m, MaskedLinear) for m in enc.net))

    def test_encoder_raises_on_invalid_layer_op(self):
        layers = LayerList([Layer(units=4, op="unknown_op")])
        with self.assertRaises(ValueError) as ctx:
            Encoder(feature_dim=4, latent_dim=4, layers=layers)
        self.assertIn("Unknown Layer.op", str(ctx.exception))

    def test_encoder_auto_projection_without_activation(self):
        enc = Encoder(feature_dim=5, latent_dim=3, layers=None, default_activation=None)
        acts = [m for m in enc.net if isinstance(m, (nn.ReLU, nn.Tanh, nn.Sigmoid))]
        self.assertEqual(len(acts), 0)

    def test_refresh_mask_noop_when_no_constraint(self):
        enc = Encoder(feature_dim=3, latent_dim=2, layers=None)
        try:
            enc.refresh_mask(device=torch.device("cpu"))
        except Exception:
            self.fail("refresh_mask should be a no-op if no constraint is present")

    def test_encoder_returns_hidden_without_latent_heads(self):
        enc = Encoder(feature_dim=4, latent_dim=2, layers=None, make_latent_heads=False)
        x = torch.randn(2, 4)
        h = enc(x)
        self.assertEqual(tuple(h.shape), (2, 2))

    def test_encoder_global_batch_norm_on_input(self):
        enc = Encoder(feature_dim=4, latent_dim=2, layers=None, batch_norm=True)
        # First layer should be global batch norm
        assert isinstance(enc.net[0], nn.BatchNorm1d)

        x = torch.randn(2, 4)
        mu, logvar, z = enc(x)
        assert z.shape == (2, 2)

    def test_encoder_layer_batch_norm(self):
        layers = LayerList([
            Layer(units=6, activation="relu", batch_norm=True),  # BN should be added
            Layer(units=2, activation=None)
        ])
        enc = Encoder(feature_dim=4, latent_dim=2, layers=layers, batch_norm=False)

        # Should include: Linear -> ReLU -> BN -> Linear
        modules = list(enc.net)
        assert isinstance(modules[0], nn.Linear)
        assert isinstance(modules[1], nn.ReLU)
        assert isinstance(modules[2], nn.BatchNorm1d)
        assert isinstance(modules[3], nn.Linear)

        x = torch.randn(2, 4)
        mu, logvar, z = enc(x)
        assert z.shape == (2, 2)

    def test_sampling_enabled_z_differs_from_mu(self):
        """When sampling=True, z should differ from mu due to reparameterization noise."""
        torch.manual_seed(0)
        enc = Encoder(feature_dim=8, latent_dim=4, layers=None, sampling=True)
        enc.eval()  # eval mode doesn't disable sampling — it's controlled by the flag
        x = torch.randn(16, 8)

        mu, logvar, z = enc(x)

        self.assertEqual(mu.shape, z.shape)
        # With sampling enabled, z should NOT equal mu (reparameterization adds noise)
        self.assertFalse(torch.allclose(z, mu),
                         "z should differ from mu when sampling=True")

    def test_sampling_disabled_z_equals_mu(self):
        """When sampling=False, z should equal mu (no reparameterization noise)."""
        enc = Encoder(feature_dim=8, latent_dim=4, layers=None, sampling=False)
        x = torch.randn(16, 8)

        mu, logvar, z = enc(x)

        self.assertEqual(mu.shape, z.shape)
        # With sampling disabled, z must be identical to mu
        self.assertTrue(torch.allclose(z, mu),
                        "z should equal mu when sampling=False")

    def test_default_sampling_is_true(self):
        """Encoder default should have sampling enabled for proper VAE training."""
        enc = Encoder(feature_dim=6, latent_dim=3)
        self.assertTrue(enc._sampling, "Default sampling should be True")

    def test_forward_always_returns_three_tuple_with_latent_heads(self):
        """forward() must always return (mu, logvar, z) when make_latent_heads=True."""
        for sampling in (True, False):
            enc = Encoder(feature_dim=6, latent_dim=3, sampling=sampling)
            x = torch.randn(4, 6)
            result = enc(x)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3,
                             f"Expected 3-tuple with sampling={sampling}")
            mu, logvar, z = result
            self.assertEqual(mu.shape, z.shape)
            self.assertEqual(mu.shape, logvar.shape)


if __name__ == "__main__":
    unittest.main()