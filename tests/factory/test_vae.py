
"""Tests for embkit.factory.mapping utilities."""

import unittest
import tempfile
from pathlib import Path

from embkit.models.vae.vae import VAE
from embkit import factory


class TestVAESave(unittest.TestCase):
    def test_save_and_load(self):

        features = list(str(i) for i in range(10))
        vae = VAE(features, latent_dim=5)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "vae.pth"
            factory.save(vae, model_path)

            new_vae = factory.load(model_path)

        self.assertEqual(new_vae.features, vae.features)
        self.assertEqual(new_vae.latent_dim, vae.latent_dim)
        self.assertEqual(type(new_vae.encoder), type(vae.encoder))
        self.assertEqual(type(new_vae.decoder), type(vae.decoder))

    def test_from_dict_without_layer_keys(self):
        model = VAE.from_dict({
            "features": ["a", "b", "c"],
            "latent_dim": 2,
            "batch_norm": False,
        })

        self.assertEqual(model.features, ["a", "b", "c"])
        self.assertEqual(model.latent_dim, 2)

