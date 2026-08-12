import unittest
import warnings
import torch
from torch import nn
from embkit.losses import (
    bce, net_vae_loss,
    MSELoss, BCELoss, BCEWithLogitsLoss, BCEKLWeightedLoss,
    VAELoss, get_loss, LOSS_REGISTRY,
)


class DummyVAE:
    def __init__(self, output_dim=10):
        self.encoder_called = False
        self.decoder_called = False
        self.output_dim = output_dim

    def encoder(self, x):
        self.encoder_called = True
        batch_size = x.size(0)
        mu = torch.zeros(batch_size, 4)
        logvar = torch.zeros(batch_size, 4)
        z = torch.randn(batch_size, 4)
        return mu, logvar, z

    def decoder(self, z):
        self.decoder_called = True
        batch_size = z.size(0)
        return torch.sigmoid(torch.randn(batch_size, self.output_dim))  # match input dim


class TestVAELossBase(unittest.TestCase):
    """Tests for the VAELoss base class and BetaWarmupMixin."""

    def test_vae_loss_is_nn_module(self):
        for cls in (MSELoss, BCELoss, BCEWithLogitsLoss, BCEKLWeightedLoss):
            self.assertIsInstance(cls(), nn.Module)
            self.assertIsInstance(cls(), VAELoss)

    def test_step_beta_increments(self):
        loss = BCELoss(beta=0.0)
        loss.step_beta(kappa=0.1)
        self.assertAlmostEqual(loss.beta, 0.1)

    def test_step_beta_clamps_at_max(self):
        loss = BCELoss(beta=0.95)
        loss.step_beta(kappa=0.1, max_beta=1.0)
        self.assertAlmostEqual(loss.beta, 1.0)

    def test_beta_can_be_updated_directly(self):
        loss = BCELoss(beta=1.0)
        loss.beta = 0.5
        self.assertAlmostEqual(loss.beta, 0.5)

    def test_kl_divergence_static(self):
        mu = torch.zeros(4, 2)
        logvar = torch.zeros(4, 2)
        kl = VAELoss._kl_divergence(mu, logvar)
        self.assertEqual(kl.shape, (4,))
        # KL of N(0,1) vs N(0,1) should be 0
        self.assertTrue(torch.allclose(kl, torch.zeros(4)))


class TestMSELoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch = 5
        self.dim = 10
        self.x = torch.rand(self.batch, self.dim)
        self.recon = torch.rand_like(self.x)
        self.mu = torch.zeros(self.batch, 4)
        self.logvar = torch.zeros(self.batch, 4)

    def test_output_shapes(self):
        total, recon, kl = MSELoss()(self.recon, self.x, self.mu, self.logvar)
        for t in (total, recon, kl):
            self.assertEqual(t.shape, ())

    def test_reduction_kwarg(self):
        loss = MSELoss(reduction="sum")
        total, _, _ = loss(self.recon, self.x, self.mu, self.logvar)
        self.assertEqual(total.shape, ())

    def test_beta_zero_ignores_kl(self):
        # Use non-trivial mu/logvar so KL != 0
        mu = torch.randn(self.batch, 4)
        logvar = torch.randn(self.batch, 4)
        total_b0, recon_b0, _ = MSELoss(beta=0.0)(self.recon, self.x, mu, logvar)
        total_b1, recon_b1, _ = MSELoss(beta=1.0)(self.recon, self.x, mu, logvar)
        self.assertAlmostEqual(float(recon_b0), float(recon_b1), places=5)
        # total differs because of KL term
        self.assertNotAlmostEqual(float(total_b0), float(total_b1), places=3)


class TestBCELoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch = 5
        self.dim = 10
        self.x = torch.rand(self.batch, self.dim)
        self.recon = torch.rand_like(self.x)
        self.mu = torch.zeros(self.batch, 4)
        self.logvar = torch.zeros(self.batch, 4)

    def test_output_shapes_and_sign(self):
        total, recon, kl = BCELoss()(self.recon, self.x, self.mu, self.logvar)
        for t in (total, recon, kl):
            self.assertEqual(t.shape, ())
        self.assertGreaterEqual(total.item(), 0)
        self.assertGreaterEqual(recon.item(), 0)
        self.assertGreaterEqual(kl.item(), 0)

    def test_beta_affects_total(self):
        mu = torch.randn(self.batch, 4)
        logvar = torch.randn(self.batch, 4)
        total_b0, _, _ = BCELoss(beta=0.0)(self.recon, self.x, mu, logvar)
        total_b1, _, _ = BCELoss(beta=1.0)(self.recon, self.x, mu, logvar)
        self.assertNotAlmostEqual(float(total_b0), float(total_b1), places=3)


class TestBCEWithLogitsLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.batch = 5
        self.dim = 10
        self.x = torch.rand(self.batch, self.dim)
        self.logits = torch.randn(self.batch, self.dim)  # unbounded
        self.mu = torch.zeros(self.batch, 4)
        self.logvar = torch.zeros(self.batch, 4)

    def test_output_shapes(self):
        total, recon, kl = BCEWithLogitsLoss()(self.logits, self.x, self.mu, self.logvar)
        for t in (total, recon, kl):
            self.assertEqual(t.shape, ())

    def test_accepts_logits_outside_01(self):
        # Should not raise even with values far outside [0, 1]
        logits = torch.randn(self.batch, self.dim) * 10
        BCEWithLogitsLoss()(logits, self.x, self.mu, self.logvar)


class TestBCEKLWeightedLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.batch = 5
        self.dim = 10
        self.x = torch.rand(self.batch, self.dim)
        self.recon = torch.rand_like(self.x)
        self.mu = torch.zeros(self.batch, 4)
        self.logvar = torch.zeros(self.batch, 4)

    def test_output_shapes(self):
        total, recon, kl = BCEKLWeightedLoss(beta=1.0, kl_weight=5.0)(
            self.recon, self.x, self.mu, self.logvar)
        for t in (total, recon, kl):
            self.assertEqual(t.shape, ())

    def test_kl_weight_scales_total(self):
        mu = torch.randn(self.batch, 4)
        logvar = torch.randn(self.batch, 4)
        total_w1, _, _ = BCEKLWeightedLoss(beta=1.0, kl_weight=1.0)(
            self.recon, self.x, mu, logvar)
        total_w5, _, _ = BCEKLWeightedLoss(beta=1.0, kl_weight=5.0)(
            self.recon, self.x, mu, logvar)
        self.assertNotAlmostEqual(float(total_w1), float(total_w5), places=3)


class TestGetLoss(unittest.TestCase):
    def test_registry_keys(self):
        self.assertIn("mse", LOSS_REGISTRY)
        self.assertIn("bce", LOSS_REGISTRY)
        self.assertIn("bce-logit", LOSS_REGISTRY)

    def test_get_loss_returns_correct_type(self):
        self.assertIsInstance(get_loss("mse"), MSELoss)
        self.assertIsInstance(get_loss("bce"), BCELoss)
        self.assertIsInstance(get_loss("bce-logit"), BCEWithLogitsLoss)

    def test_get_loss_passes_kwargs(self):
        loss = get_loss("bce", beta=0.5)
        self.assertAlmostEqual(loss.beta, 0.5)

    def test_get_loss_unknown_raises(self):
        with self.assertRaises(KeyError):
            get_loss("unknown")


class TestDeprecatedFunctions(unittest.TestCase):
    """Deprecated free functions still work (with DeprecationWarning)."""

    def setUp(self):
        torch.manual_seed(42)
        self.batch = 5
        self.dim = 10
        self.x = torch.rand(self.batch, self.dim)
        self.recon = torch.rand_like(self.x)
        self.mu = torch.zeros(self.batch, 4)
        self.logvar = torch.zeros(self.batch, 4)

    def test_bce_deprecated_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            total, recon, kl = bce(self.recon, self.x, self.mu, self.logvar)
            self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
        self.assertEqual(total.shape, ())
        self.assertGreaterEqual(total.item(), 0)

    def test_net_vae_loss_deprecated_warns(self):
        model = DummyVAE()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            total, recon, kl = net_vae_loss(model, self.x)
            self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
        self.assertTrue(model.encoder_called)
        self.assertTrue(model.decoder_called)
        self.assertEqual(total.shape, ())
        self.assertGreaterEqual(total.item(), 0)


if __name__ == '__main__':
    unittest.main()