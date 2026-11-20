"""Unit tests for ``embkit.losses.alignment``."""

import unittest

import torch

from embkit.losses import VAELoss, kl_diag_gaussians, pairwise_alignment_loss, info_nce_loss, mmd_loss


class TestKLDiagGaussians(unittest.TestCase):
    def test_matches_torch_distributions(self):
        torch.manual_seed(0)
        mq, lq, mp, lp = (torch.randn(5, 3) for _ in range(4))
        ref = torch.distributions.kl_divergence(
            torch.distributions.Normal(mq, torch.exp(0.5 * lq)),
            torch.distributions.Normal(mp, torch.exp(0.5 * lp))).sum(-1)
        self.assertTrue(torch.allclose(kl_diag_gaussians(mq, lq, mp, lp), ref, atol=1e-5))

    def test_standard_normal_prior_matches_vae_loss_kl(self):
        mu, lv = torch.randn(4, 2), torch.randn(4, 2)
        kl = kl_diag_gaussians(mu, lv, torch.zeros_like(mu), torch.zeros_like(lv))
        self.assertTrue(torch.allclose(kl, VAELoss._kl_divergence(mu, lv), atol=1e-6))


class TestPairwiseAlignment(unittest.TestCase):
    def test_two_views(self):
        a, b = torch.randn(4, 3), torch.randn(4, 3)
        self.assertTrue(torch.allclose(pairwise_alignment_loss([a, b]), ((a - b) ** 2).sum(-1)))

    def test_mapping_three_views_is_mean_of_pairs(self):
        a, b, c = torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3)
        expected = (((a - b) ** 2).sum(-1) + ((a - c) ** 2).sum(-1) + ((b - c) ** 2).sum(-1)) / 3
        self.assertTrue(torch.allclose(pairwise_alignment_loss({"a": a, "b": b, "c": c}), expected))

    def test_single_view_is_zero(self):
        self.assertTrue(torch.equal(pairwise_alignment_loss([torch.randn(4, 3)]), torch.zeros(4)))

    def test_no_views(self):
        with self.assertRaises(ValueError):
            pairwise_alignment_loss([])


class TestInfoNCE(unittest.TestCase):
    def test_aligned_views_have_lower_loss(self):
        torch.manual_seed(0)
        a = torch.randn(16, 4) * 3
        aligned = info_nce_loss(a, a + 0.01 * torch.randn(16, 4)).mean()
        shuffled = info_nce_loss(a, a[torch.randperm(16)]).mean()
        self.assertLess(float(aligned), float(shuffled))
        self.assertEqual(info_nce_loss(a, a).shape, (16,))

    def test_gradients_flow(self):
        a = torch.randn(8, 3, requires_grad=True)
        info_nce_loss(a, torch.randn(8, 3), temperature=0.5).mean().backward()
        self.assertIsNotNone(a.grad)

    def test_errors(self):
        with self.assertRaises(ValueError):
            info_nce_loss(torch.randn(4, 3), torch.randn(5, 3))
        with self.assertRaises(ValueError):
            info_nce_loss(torch.randn(1, 3), torch.randn(1, 3))
        with self.assertRaises(ValueError):
            info_nce_loss(torch.randn(4, 3), torch.randn(4, 3), temperature=0.0)


class TestMMD(unittest.TestCase):
    def test_same_distribution_smaller_than_shifted(self):
        torch.manual_seed(0)
        a, b = torch.randn(64, 2), torch.randn(64, 2)
        self.assertLess(float(mmd_loss(a, b)), float(mmd_loss(a, b + 3.0)))

    def test_identical_sets_zero(self):
        a = torch.randn(32, 2)
        self.assertAlmostEqual(float(mmd_loss(a, a)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
