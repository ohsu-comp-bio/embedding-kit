import unittest
import torch
from torch import nn

from embkit import factory
from embkit.models.vae.encoder import VAEEncoder, EncoderOutput


@factory.nn_module
class _LinearBackbone(nn.Module):
    """Tiny registry-registered backbone used to exercise
    :meth:`VAEEncoder.to_dict` / :meth:`VAEEncoder.from_dict`."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

    def to_dict(self):
        return {"in_dim": self.in_dim, "out_dim": self.out_dim}

    @classmethod
    def from_dict(cls, d):
        return cls(d["in_dim"], d["out_dim"])


class TestVAEEncoder(unittest.TestCase):
    def setUp(self):
        self.feature_dim = 6
        self.latent_dim = 3
        self.batch = 5
        self.enc = VAEEncoder(
            backbone=_LinearBackbone(4, self.feature_dim),
            feature_dim=self.feature_dim,
            latent_dim=self.latent_dim,
        )

    def test_forward_train_mode_shapes(self):
        self.enc.train()
        out = self.enc(torch.randn(self.batch, 4))
        self.assertIsInstance(out, EncoderOutput)
        self.assertEqual(tuple(out.mu.shape), (self.batch, self.latent_dim))
        self.assertEqual(tuple(out.logvar.shape), (self.batch, self.latent_dim))
        self.assertEqual(tuple(out.z.shape), (self.batch, self.latent_dim))
        for t in (out.mu, out.logvar, out.z):
            self.assertTrue(torch.isfinite(t).all())

    def test_forward_eval_mode_is_deterministic(self):
        self.enc.eval()
        x = torch.randn(self.batch, 4)
        with torch.no_grad():
            a = self.enc(x)
            b = self.enc(x)
        # eval reparameterization returns mu exactly -> identical z
        self.assertTrue(torch.allclose(a.z, a.mu))
        self.assertTrue(torch.allclose(a.z, b.z))

    def test_reparameterize_train_scales_by_logvar(self):
        self.enc.train()
        mu = torch.zeros(2, self.latent_dim)
        # zero logvar -> std 1; large logvar -> large std
        small_std = self.enc.reparameterize(mu, torch.zeros(2, self.latent_dim))
        big_std = self.enc.reparameterize(mu, torch.full((2, self.latent_dim), 4.0))
        self.assertEqual(tuple(small_std.shape), (2, self.latent_dim))
        # samples with large variance should not be all zero in general
        self.assertTrue(big_std.abs().sum() > 0)

    def test_reparameterize_eval_returns_mu(self):
        self.enc.eval()
        mu = torch.randn(2, self.latent_dim)
        logvar = torch.randn(2, self.latent_dim)
        z = self.enc.reparameterize(mu, logvar)
        self.assertTrue(torch.allclose(z, mu))

    def test_compute_kl_elements(self):
        mu = torch.zeros(2, self.latent_dim)
        logvar = torch.zeros(2, self.latent_dim)
        kl = self.enc.compute_kl_elements(mu, logvar)
        # For mu=0, logvar=0 -> std^2=1 -> KL element = -0.5*(1+0-0-1)=0
        self.assertEqual(tuple(kl.shape), (2, self.latent_dim))
        self.assertTrue(torch.allclose(kl, torch.zeros_like(kl)))

    def test_flatten_branch_for_spatial_backbone(self):
        # Backbone that outputs a 3D tensor (B, C, L) to hit the `h.dim() > 2`
        # flatten branch.
        class SpatialBackbone(nn.Module):
            def __init__(self, out_channels, length):
                super().__init__()
                self.out_channels = out_channels
                self.length = length

            def forward(self, x):
                b = x.shape[0]
                return torch.zeros(b, self.out_channels, self.length)

        enc = VAEEncoder(
            backbone=SpatialBackbone(3, 2),  # 3*2 = 6 = feature_dim
            feature_dim=6,
            latent_dim=3,
        )
        enc.eval()
        out = enc(torch.randn(self.batch, 4))
        self.assertEqual(tuple(out.mu.shape), (self.batch, 3))

    def test_to_dict_keys(self):
        d = self.enc.to_dict()
        self.assertIn("backbone", d)
        self.assertEqual(d["feature_dim"], self.feature_dim)
        self.assertEqual(d["latent_dim"], self.latent_dim)
        # backbone stored as a buildable registry dict
        self.assertEqual(d["backbone"]["in_dim"], 4)
        self.assertEqual(d["backbone"]["out_dim"], self.feature_dim)

    def test_to_dict_injects_class_marker(self):
        # factory.nn_module wraps to_dict to attach __class__
        self.assertIn("__class__", d := self.enc.to_dict())

    def test_from_dict_reconstructs(self):
        d = self.enc.to_dict()
        new_enc = VAEEncoder.from_dict(d)
        self.assertIsInstance(new_enc.backbone, _LinearBackbone)
        self.assertEqual(new_enc.feature_dim, self.feature_dim)
        self.assertEqual(new_enc.latent_dim, self.latent_dim)

        # forward works on the reconstructed encoder
        new_enc.eval()
        out = new_enc(torch.randn(self.batch, 4))
        self.assertEqual(tuple(out.mu.shape), (self.batch, self.latent_dim))


if __name__ == "__main__":
    unittest.main()
