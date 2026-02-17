
"""Tests for embkit.factory.mapping utilities."""

import unittest
from torch import nn

from embkit.models import VAE
from embkit import factory


class TestVAESave(unittest.TestCase):
    def test_save_and_load(self):

        features = list(str(i) for i in range(10))
        vae = VAE(features, latent_dim=5)

        factory.save(vae, "vae.pth")

        new_vae = factory.load("vae.pth")
        print(new_vae)

