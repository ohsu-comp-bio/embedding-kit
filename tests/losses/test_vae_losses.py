import unittest
import torch
from embkit.losses import vae_loss, net_vae_loss  # adjust to your path


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


class TestVAELoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 5
        self.input_dim = 10
        self.x = torch.rand(self.batch_size, self.input_dim)
        self.recon_x = torch.rand_like(self.x)
        self.mu = torch.zeros(self.batch_size, 4)
        self.logvar = torch.zeros(self.batch_size, 4)

    def test_vae_loss_output_shapes(self):
        total, recon, kl = vae_loss(self.recon_x, self.x, self.mu, self.logvar)
        self.assertIsInstance(total, torch.Tensor)
        self.assertEqual(total.shape, ())
        self.assertEqual(recon.shape, ())
        self.assertEqual(kl.shape, ())

    def test_vae_loss_outputs_positive(self):
        total, recon, kl = vae_loss(self.recon_x, self.x, self.mu, self.logvar)
        self.assertGreaterEqual(total.item(), 0)
        self.assertGreaterEqual(recon.item(), 0)
        self.assertGreaterEqual(kl.item(), 0)

    def test_net_vae_loss_runs_and_calls_model(self):
        model = DummyVAE()
        total, recon, kl = net_vae_loss(model, self.x)
        self.assertTrue(model.encoder_called)
        self.assertTrue(model.decoder_called)
        self.assertEqual(total.shape, ())
        self.assertGreaterEqual(total.item(), 0)
        self.assertGreaterEqual(recon.item(), 0)
        self.assertGreaterEqual(kl.item(), 0)


if __name__ == '__main__':
    unittest.main()