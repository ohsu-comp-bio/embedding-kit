import unittest
import warnings
import torch
from torch import nn
from embkit.losses import (
    MSEVAELoss, BCEVAELoss, BCEWithLogitsVAELoss, BCEKLWeightedVAELoss,
    MSELoss, BCELoss, BCEWithLogitsLoss, BCEKLWeightedLoss,
    VAELoss, get_vae_loss, VAE_LOSS_REGISTRY, get_loss, LOSS_REGISTRY,
)
from embkit.losses import vae_loss as _vae_loss_mod


class TestVAELossBase(unittest.TestCase):
    """Tests for the VAELoss base class and BetaWarmupMixin."""

    def test_vae_loss_is_nn_module(self):
        for cls in (MSEVAELoss, BCEVAELoss, BCEWithLogitsVAELoss, BCEKLWeightedVAELoss):
            self.assertIsInstance(cls(), nn.Module)
            self.assertIsInstance(cls(), VAELoss)

    def test_step_beta_increments(self):
        loss = BCEVAELoss(beta=0.0)
        loss.step_beta(kappa=0.1)
        self.assertAlmostEqual(loss.beta, 0.1)

    def test_step_beta_clamps_at_max(self):
        loss = BCEVAELoss(beta=0.95)
        loss.step_beta(kappa=0.1, max_beta=1.0)
        self.assertAlmostEqual(loss.beta, 1.0)

    def test_beta_can_be_updated_directly(self):
        loss = BCEVAELoss(beta=1.0)
        loss.beta = 0.5
        self.assertAlmostEqual(loss.beta, 0.5)

    def test_kl_divergence_static(self):
        mu = torch.zeros(4, 2)
        logvar = torch.zeros(4, 2)
        kl = VAELoss._kl_divergence(mu, logvar)
        self.assertEqual(kl.shape, (4,))
        # KL of N(0,1) vs N(0,1) should be 0
        self.assertTrue(torch.allclose(kl, torch.zeros(4)))


class TestMSEVAELoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch = 5
        self.dim = 10
        self.x = torch.rand(self.batch, self.dim)
        self.recon = torch.rand_like(self.x)
        self.mu = torch.zeros(self.batch, 4)
        self.logvar = torch.zeros(self.batch, 4)

    def test_output_shapes(self):
        total, recon, kl = MSEVAELoss()(self.recon, self.x, self.mu, self.logvar)
        for t in (total, recon, kl):
            self.assertEqual(t.shape, ())

    def test_reduction_kwarg(self):
        loss = MSEVAELoss(reduction="sum")
        total, _, _ = loss(self.recon, self.x, self.mu, self.logvar)
        self.assertEqual(total.shape, ())

    def test_beta_zero_ignores_kl(self):
        # Use non-trivial mu/logvar so KL != 0
        mu = torch.randn(self.batch, 4)
        logvar = torch.randn(self.batch, 4)
        total_b0, recon_b0, _ = MSEVAELoss(beta=0.0)(self.recon, self.x, mu, logvar)
        total_b1, recon_b1, _ = MSEVAELoss(beta=1.0)(self.recon, self.x, mu, logvar)
        self.assertAlmostEqual(float(recon_b0), float(recon_b1), places=5)
        # total differs because of KL term
        self.assertNotAlmostEqual(float(total_b0), float(total_b1), places=3)


class TestBCEVAELoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch = 5
        self.dim = 10
        self.x = torch.rand(self.batch, self.dim)
        self.recon = torch.rand_like(self.x)
        self.mu = torch.zeros(self.batch, 4)
        self.logvar = torch.zeros(self.batch, 4)

    def test_output_shapes_and_sign(self):
        total, recon, kl = BCEVAELoss()(self.recon, self.x, self.mu, self.logvar)
        for t in (total, recon, kl):
            self.assertEqual(t.shape, ())
        self.assertGreaterEqual(total.item(), 0)
        self.assertGreaterEqual(recon.item(), 0)
        self.assertGreaterEqual(kl.item(), 0)

    def test_beta_affects_total(self):
        mu = torch.randn(self.batch, 4)
        logvar = torch.randn(self.batch, 4)
        total_b0, _, _ = BCEVAELoss(beta=0.0)(self.recon, self.x, mu, logvar)
        total_b1, _, _ = BCEVAELoss(beta=1.0)(self.recon, self.x, mu, logvar)
        self.assertNotAlmostEqual(float(total_b0), float(total_b1), places=3)


class TestBCEWithLogitsVAELoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.batch = 5
        self.dim = 10
        self.x = torch.rand(self.batch, self.dim)
        self.logits = torch.randn(self.batch, self.dim)  # unbounded
        self.mu = torch.zeros(self.batch, 4)
        self.logvar = torch.zeros(self.batch, 4)

    def test_output_shapes(self):
        total, recon, kl = BCEWithLogitsVAELoss()(self.logits, self.x, self.mu, self.logvar)
        for t in (total, recon, kl):
            self.assertEqual(t.shape, ())

    def test_accepts_logits_outside_01(self):
        # Should not raise even with values far outside [0, 1]
        logits = torch.randn(self.batch, self.dim) * 10
        BCEWithLogitsVAELoss()(logits, self.x, self.mu, self.logvar)


class TestBCEKLWeightedVAELoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.batch = 5
        self.dim = 10
        self.x = torch.rand(self.batch, self.dim)
        self.recon = torch.rand_like(self.x)
        self.mu = torch.zeros(self.batch, 4)
        self.logvar = torch.zeros(self.batch, 4)

    def test_output_shapes(self):
        total, recon, kl = BCEKLWeightedVAELoss(beta=1.0, kl_weight=5.0)(
            self.recon, self.x, self.mu, self.logvar)
        for t in (total, recon, kl):
            self.assertEqual(t.shape, ())

    def test_kl_weight_scales_total(self):
        mu = torch.randn(self.batch, 4)
        logvar = torch.randn(self.batch, 4)
        total_w1, _, _ = BCEKLWeightedVAELoss(beta=1.0, kl_weight=1.0)(
            self.recon, self.x, mu, logvar)
        total_w5, _, _ = BCEKLWeightedVAELoss(beta=1.0, kl_weight=5.0)(
            self.recon, self.x, mu, logvar)
        self.assertNotAlmostEqual(float(total_w1), float(total_w5), places=3)


class TestGetLoss(unittest.TestCase):
    def test_registry_keys(self):
        self.assertIn("mse", VAE_LOSS_REGISTRY)
        self.assertIn("bce", VAE_LOSS_REGISTRY)
        self.assertIn("bce-logit", VAE_LOSS_REGISTRY)
        self.assertEqual(VAE_LOSS_REGISTRY, LOSS_REGISTRY)

    def test_get_loss_returns_correct_type(self):
        self.assertIsInstance(get_vae_loss("mse"), MSEVAELoss)
        self.assertIsInstance(get_vae_loss("bce"), BCEVAELoss)
        self.assertIsInstance(get_vae_loss("bce-logit"), BCEWithLogitsVAELoss)
        self.assertIsInstance(get_loss("mse"), MSEVAELoss)

    def test_get_loss_passes_kwargs(self):
        loss = get_vae_loss("bce", beta=0.5)
        self.assertAlmostEqual(loss.beta, 0.5)

    def test_get_loss_unknown_raises(self):
        with self.assertRaises(KeyError):
            get_vae_loss("unknown")


class TestCompatibilityAliases(unittest.TestCase):
    def test_class_aliases(self):
        self.assertIs(MSELoss, MSEVAELoss)
        self.assertIs(BCELoss, BCEVAELoss)
        self.assertIs(BCEWithLogitsLoss, BCEWithLogitsVAELoss)
        self.assertIs(BCEKLWeightedLoss, BCEKLWeightedVAELoss)


class TestDeprecatedFreeFunctions(unittest.TestCase):
    """The legacy module-level functions all emit a DeprecationWarning but
    still delegate to the equivalent ``VAELoss`` subclass."""

    def _sample(self):
        batch, dim = 6, 4
        recon = torch.sigmoid(torch.randn(batch, dim))
        x = (torch.rand(batch, dim) > 0.5).float()
        mu = torch.randn(batch, dim)
        logvar = torch.randn(batch, dim)
        return recon, x, mu, logvar

    def _assert_three_tensors(self, out):
        self.assertEqual(len(out), 3)
        for t in out:
            self.assertIsInstance(t, torch.Tensor)
            self.assertEqual(t.dim(), 0)

    def test_mse_warns_and_returns(self):
        recon, x, mu, logvar = self._sample()
        with self.assertWarns(DeprecationWarning):
            out = _vae_loss_mod.mse(recon, x, mu, logvar, beta=1.0)
        self._assert_three_tensors(out)
        self.assertTrue(torch.isfinite(out[0]))

    def test_bce_warns_and_returns(self):
        recon, x, mu, logvar = self._sample()
        with self.assertWarns(DeprecationWarning):
            out = _vae_loss_mod.bce(recon, x, mu, logvar, beta=1.0)
        self._assert_three_tensors(out)

    def test_bce_with_logits_warns_and_returns(self):
        recon, x, mu, logvar = self._sample()
        with self.assertWarns(DeprecationWarning):
            out = _vae_loss_mod.bce_with_logits(recon, x, mu, logvar, beta=1.0)
        self._assert_three_tensors(out)

    def test_bce_kl_weighted_warns_and_returns(self):
        recon, x, mu, logvar = self._sample()
        with self.assertWarns(DeprecationWarning):
            out = _vae_loss_mod.bce_kl_weighted(recon, x, mu, logvar, beta=1.0, kl_weight=1.0)
        self._assert_three_tensors(out)

    def test_net_vae_loss_bce_branch(self):
        # Reconstruction kept inside [0, 1] -> BCE path.
        class EncOut:
            def __init__(self, mu, logvar, z):
                self.mu, self.logvar, self.z = mu, logvar, z

        class Enc(nn.Module):
            def forward(self, x):
                mu = torch.zeros_like(x)
                logvar = torch.zeros_like(x)
                z = mu + torch.randn_like(x) * torch.exp(0.5 * logvar)
                return EncOut(mu, logvar, z)

        class Dec(nn.Module):
            def forward(self, z):
                return torch.sigmoid(z)

        class Model:
            encoder = Enc()
            decoder = Dec()

        x = (torch.rand(4, 3) > 0.5).float()
        with self.assertWarns(DeprecationWarning):
            out = _vae_loss_mod.net_vae_loss(Model(), x, beta=1.0)
        self._assert_three_tensors(out)
        self.assertTrue(torch.isfinite(out[0]))

    def test_net_vae_loss_bce_logit_branch(self):
        # Reconstruction pushed outside [0, 1] -> BCEWithLogits path.
        class EncOut:
            def __init__(self, mu, logvar, z):
                self.mu, self.logvar, self.z = mu, logvar, z

        class Enc(nn.Module):
            def forward(self, x):
                return EncOut(torch.zeros_like(x), torch.zeros_like(x),
                              torch.zeros_like(x))

        class Dec(nn.Module):
            def forward(self, z):
                # raw logits, unbounded and pushed beyond [0, 1]
                return z + 5.0

        class Model:
            encoder = Enc()
            decoder = Dec()

        x = (torch.rand(4, 3) > 0.5).float()
        with self.assertWarns(DeprecationWarning):
            out = _vae_loss_mod.net_vae_loss(Model(), x, beta=1.0)
        self._assert_three_tensors(out)
        self.assertTrue(torch.isfinite(out[0]))


if __name__ == '__main__':
    unittest.main()