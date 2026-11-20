"""Unit tests for ``embkit.modules.conv`` (``ConvBackbone``, ``ConvDecoder``)."""

import os
import tempfile
import unittest

import torch

from embkit.factory.core import build, save, load
from embkit.models.vae import VAE, VAEEncoder
from embkit.modules.conv import ConvBackbone, ConvDecoder, augment_images


class TestConvBackbone(unittest.TestCase):
    def test_flat_and_grid_inputs(self):
        bb = ConvBackbone(in_channels=3, spatial_shape=(16, 16), channels=(8, 16), hidden_dim=32)
        flat = torch.rand(4, 3 * 16 * 16)
        self.assertEqual(bb(flat).shape, (4, 32))
        self.assertEqual(bb(flat.reshape(4, 3, 16, 16)).shape, (4, 32))
        self.assertEqual(bb.feature_dim, 32)
        self.assertEqual(bb.flat_dim, 768)

    def test_no_hidden_layer_and_mean_pool(self):
        bb = ConvBackbone(in_channels=1, spatial_shape=(8, 8), channels=(4,), hidden_dim=None, pool="mean")
        self.assertEqual(bb.feature_dim, 4)
        self.assertEqual(bb(torch.rand(2, 64)).shape, (2, 4))

    def test_one_dimensional_signal(self):
        bb = ConvBackbone(in_channels=2, spatial_shape=(32,), channels=(4, 8), norm="batch", hidden_dim=16)
        self.assertEqual(bb(torch.rand(3, 64)).shape, (3, 16))

    def test_augment_in_training_only(self):
        torch.manual_seed(0)
        bb = ConvBackbone(in_channels=1, spatial_shape=(8, 8), channels=(4,), augment=("flip", "rot90"),
                          dropout=0.1, norm="none", activation="relu")
        bb.train()
        self.assertEqual(bb(torch.rand(2, 64)).shape, (2, 256))
        bb.eval()
        x = torch.rand(2, 64)
        self.assertTrue(torch.equal(bb(x), bb(x)))

    def test_augment_images_preserves_shape(self):
        x = torch.rand(2, 3, 8, 8)
        self.assertEqual(augment_images(x, ("flip", "rot90")).shape, x.shape)

    def test_invalid_configs(self):
        bad = [
            dict(spatial_shape=(4, 4, 4)),
            dict(in_channels=0),
            dict(channels=()),
            dict(spatial_shape=(10, 10)),
            dict(kernel_size=2),
            dict(norm="layer"),
            dict(activation="tanh"),
            dict(dropout=1.0),
            dict(augment=("blur",)),
            dict(spatial_shape=(16,), augment=("flip",)),
            dict(pool="max"),
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ConvBackbone(**{**dict(spatial_shape=(16, 16), channels=(4, 8)), **kwargs})

    def test_wrong_input_shape(self):
        bb = ConvBackbone(in_channels=1, spatial_shape=(8, 8), channels=(4,))
        with self.assertRaises(ValueError):
            bb(torch.rand(2, 10))
        with self.assertRaises(ValueError):
            bb(torch.rand(2, 1, 4, 4))

    def test_roundtrip(self):
        bb = ConvBackbone(in_channels=1, spatial_shape=(8, 8), channels=(4,), hidden_dim=12, pool="mean")
        d = bb.to_dict()
        self.assertIn("__class__", d)
        rebuilt = build(d)
        self.assertIsInstance(rebuilt, ConvBackbone)
        self.assertEqual(rebuilt.feature_dim, 12)
        self.assertEqual(rebuilt.pool, "mean")


class TestConvDecoder(unittest.TestCase):
    def test_output_shape_and_sigmoid(self):
        dec = ConvDecoder(latent_dim=5, in_channels=3, spatial_shape=(16, 16), channels=(8, 16),
                          output_activation="sigmoid")
        out = dec(torch.randn(4, 5))
        self.assertEqual(out.shape, (4, 768))
        self.assertTrue(((out >= 0) & (out <= 1)).all())

    def test_transpose_upsampling_1d(self):
        dec = ConvDecoder(latent_dim=3, in_channels=2, spatial_shape=(16,), channels=(4, 8), upsample="transpose")
        self.assertEqual(dec(torch.randn(2, 3)).shape, (2, 32))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            ConvDecoder(latent_dim=3, spatial_shape=(8, 8), channels=(4,), upsample="bicubic")
        with self.assertRaises(ValueError):
            ConvDecoder(latent_dim=3, spatial_shape=(8, 8), channels=(4,), output_activation="tanh")

    def test_roundtrip(self):
        dec = ConvDecoder(latent_dim=3, in_channels=1, spatial_shape=(8, 8), channels=(4,), hidden_dim=10)
        rebuilt = build(dec.to_dict())
        self.assertEqual(rebuilt.hidden_dim, 10)
        self.assertEqual(rebuilt.feature_dim, 64)


class TestConvVAE(unittest.TestCase):
    def _model(self):
        bb = ConvBackbone(in_channels=1, spatial_shape=(8, 8), channels=(4, 8), hidden_dim=16,
                          device=torch.device("cpu"), dtype=torch.float32)
        encoder = VAEEncoder(bb, feature_dim=bb.feature_dim, latent_dim=3)
        decoder = ConvDecoder(latent_dim=3, in_channels=1, spatial_shape=(8, 8), channels=(4, 8),
                              output_activation="sigmoid", device=torch.device("cpu"))
        return VAE(encoder=encoder, decoder=decoder)

    def test_forward(self):
        out = self._model()(torch.rand(5, 64))
        self.assertEqual(out.recon.shape, (5, 64))
        self.assertEqual(out.mu.shape, (5, 3))

    def test_save_load_roundtrip(self):
        model = self._model().eval()
        x = torch.rand(3, 64)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "conv.model")
            save(model, path)
            loaded = load(path).eval()
        self.assertTrue(torch.allclose(model.encode(x), loaded.encode(x)))
        self.assertTrue(torch.allclose(model(x).recon, loaded(x).recon))


if __name__ == "__main__":
    unittest.main()
