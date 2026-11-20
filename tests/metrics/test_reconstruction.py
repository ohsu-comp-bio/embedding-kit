"""Unit tests for ``embkit.metrics.reconstruction``."""

import unittest

import numpy as np
import torch

from embkit.metrics.reconstruction import (
    mse_per_feature, pearson_corr_per_feature, auc_per_feature_binary, reconstruction_metrics, deviation_scores,
)


class TestPerFeature(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.normal(size=(50, 4))

    def test_mse(self):
        pred = self.x.copy()
        pred[:, 1] += 2.0
        self.assertTrue(np.allclose(mse_per_feature(self.x, pred), [0.0, 4.0, 0.0, 0.0]))

    def test_pearson(self):
        r = pearson_corr_per_feature(self.x, 3 * self.x + 1)
        self.assertTrue(np.allclose(r, 1.0, atol=1e-5))
        self.assertTrue(np.allclose(pearson_corr_per_feature(self.x, -self.x), -1.0, atol=1e-5))

    def test_auc(self):
        y = (self.x > 0).astype(int)
        y[:, 3] = 1  # constant column -> NaN
        aucs = auc_per_feature_binary(y, self.x)
        self.assertTrue(np.allclose(aucs[:3], 1.0))
        self.assertTrue(np.isnan(aucs[3]))

    def test_shape_mismatch(self):
        for fn in (mse_per_feature, pearson_corr_per_feature, auc_per_feature_binary):
            with self.assertRaises(ValueError):
                fn(self.x, self.x[:, :2])


class TestReconstructionMetrics(unittest.TestCase):
    def test_continuous(self):
        x = np.random.default_rng(0).normal(size=(30, 5))
        out = reconstruction_metrics(x, x)
        self.assertEqual(out["mse_mean"], 0.0)
        self.assertAlmostEqual(out["pearson_median"], 1.0, places=5)
        self.assertEqual(out["mse_per_feature"].shape, (5,))

    def test_binary(self):
        rng = np.random.default_rng(0)
        score = rng.normal(size=(40, 3))
        out = reconstruction_metrics((score > 0).astype(int), score, kind="binary")
        self.assertEqual(out["auc_mean"], 1.0)

    def test_bad_kind(self):
        with self.assertRaises(ValueError):
            reconstruction_metrics(np.zeros((2, 2)), np.zeros((2, 2)), kind="count")


class TestDeviationScores(unittest.TestCase):
    def test_robust_z(self):
        null = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = deviation_scores(np.array([3.0, 3.0 + 1.4826]), null)
        self.assertTrue(np.allclose(z, [0.0, 1.0]))

    def test_accepts_tensors(self):
        z = deviation_scores(torch.tensor([3.0]), torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
        self.assertEqual(z.shape, (1,))

    def test_zero_spread(self):
        with self.assertRaises(ValueError):
            deviation_scores([1.0], [2.0, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
