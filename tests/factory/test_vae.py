
"""Tests for embkit.factory.mapping utilities."""

import unittest
import tempfile
from pathlib import Path

from embkit.models.vae.vae import VAE
from embkit.models.vae.net_vae import NetVAE
from embkit.constraints import PathwayConstraintInfo
from embkit.factory.layers import ConstraintInfo
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

    def test_netvae_save_and_load_roundtrip(self):
        features = ["G1", "G2", "G3"]
        latent_groups = {
            "TF1": ["G1", "G3"],
            "TF2": ["G2"],
        }
        model = NetVAE(features=features, latent_groups=latent_groups, group_layer_size=[2, 1])

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "netvae.pth"
            factory.save(model, model_path)
            loaded = factory.load(model_path)

        self.assertIsInstance(loaded, NetVAE)
        self.assertEqual(loaded.features, features)
        self.assertEqual(loaded.latent_groups, latent_groups)
        self.assertEqual(loaded.group_layer_size, [2, 1])

    def test_netvae_from_dict_rejects_deprecated_group_layer_scaling(self):
        desc = {
            "features": ["G1", "G2"],
            "latent_groups": {"TF1": ["G1"], "TF2": ["G2"]},
            "group_layer_scaling": [3, 1],
        }
        with self.assertRaises(ValueError):
            NetVAE.from_dict(desc)

    def test_constraintinfo_from_dict_pathway_dispatch(self):
        payload = {
            "op": "features-to-group",
            "feature_map": {"TF1": ["G1"], "TF2": ["G2"]},
            "in_group_scaling": 1,
            "out_group_scaling": 2,
        }
        constraint = ConstraintInfo.from_dict(payload)
        self.assertIsInstance(constraint, PathwayConstraintInfo)
