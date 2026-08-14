
"""Tests for embkit.factory.mapping utilities."""

import unittest
import tempfile
from pathlib import Path

from embkit.models.vae import VAE, BaseVAE, NetVAE, Encoder, Decoder
from embkit.constraints import PathwayConstraintInfo
from embkit.constraints.pathway_constraint import PathwayConstraintInfo
from embkit import factory


class TestVAESave(unittest.TestCase):
    def test_save_and_load(self):

        features = list(str(i) for i in range(10))
        vae = VAE(Encoder(feature_dim=50, latent_dim=2), Decoder(feature_dim=50, latent_dim=2))

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "vae.pth"
            factory.save(vae, model_path)

            new_vae = factory.load(model_path)

        self.assertEqual(type(new_vae.encoder), type(vae.encoder))
        self.assertEqual(type(new_vae.decoder), type(vae.decoder))

    def test_netvae_save_and_load_roundtrip(self):
        features = ["G1", "G2", "G3"]
        latent_groups = {
            "TF1": ["G1", "G3"],
            "TF2": ["G2"],
        }
        model = NetVAE(features=features, latent_groups=latent_groups, group_layer_scale=[2, 1])

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "netvae.pth"
            factory.save(model, model_path)
            loaded = factory.load(model_path)

        self.assertIsInstance(loaded, NetVAE)
        self.assertEqual(loaded.features, features)
        self.assertEqual(loaded.latent_groups, latent_groups)
        self.assertEqual(loaded.group_layer_scale, [2, 1])

    def test_constraintinfo_from_dict_pathway_dispatch(self):
        pc = PathwayConstraintInfo(
            op="features-to-group", 
            feature_map={"TF1": ["G1"], "TF2": ["G2"]},
            in_group_scaling=1,
            out_group_scaling=2)
        payload = pc.to_dict()
        constraint = factory.build(payload)
        self.assertIsInstance(constraint, PathwayConstraintInfo)
