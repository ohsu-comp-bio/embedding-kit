"""Unit tests for ``embkit.metrics.alignment``."""

import unittest

import numpy as np

from embkit.metrics.alignment import foscttm, match_recall_at_k, modality_mixing, modality_entropy


class TestFOSCTTM(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.Z = rng.normal(size=(40, 5))

    def test_identical_embeddings_score_zero(self):
        for metric in ("euclidean", "cosine"):
            self.assertEqual(foscttm(self.Z, self.Z, metric=metric, block_size=7), 0.0)

    def test_shuffled_embeddings_score_near_half(self):
        rng = np.random.default_rng(1)
        score = foscttm(self.Z, self.Z[rng.permutation(40)])
        self.assertGreater(score, 0.2)

    def test_hand_computed(self):
        Z1 = np.array([[0.0], [1.0], [2.0]])
        Z2 = np.array([[0.0], [2.0], [1.0]])  # samples 1 and 2 swapped
        m, per = foscttm(Z1, Z2, return_per_sample=True)
        self.assertTrue(np.allclose(per, [0.0, 0.5, 0.5]))
        self.assertAlmostEqual(m, 1 / 3, places=6)

    def test_return_options(self):
        m, s = foscttm(self.Z, self.Z + 0.1, return_sem=True)
        self.assertIsInstance(s, float)
        m, s, per = foscttm(self.Z, self.Z + 0.1, return_sem=True, return_per_sample=True)
        self.assertEqual(per.shape, (40,))

    def test_single_sample(self):
        self.assertEqual(foscttm(self.Z[:1], self.Z[:1]), 0.0)

    def test_errors(self):
        with self.assertRaises(ValueError):
            foscttm(self.Z, self.Z[:5])
        with self.assertRaises(ValueError):
            foscttm(self.Z, self.Z, metric="manhattan")


class TestRecallAtK(unittest.TestCase):
    def test_perfect_and_partial(self):
        rng = np.random.default_rng(0)
        Z = rng.normal(size=(30, 4))
        self.assertEqual(match_recall_at_k(Z, Z, k=1), 1.0)
        self.assertEqual(match_recall_at_k(Z, Z, k=1, metric="cosine", block_size=4), 1.0)
        shuffled = match_recall_at_k(Z, Z[rng.permutation(30)], k=1)
        self.assertLess(shuffled, 0.5)

    def test_k_equal_n_is_one(self):
        rng = np.random.default_rng(0)
        Z1, Z2 = rng.normal(size=(10, 3)), rng.normal(size=(10, 3))
        self.assertEqual(match_recall_at_k(Z1, Z2, k=50), 1.0)

    def test_edge_cases(self):
        Z = np.ones((1, 3))
        self.assertEqual(match_recall_at_k(Z, Z, return_sem=True, return_per_sample=True)[0], 1.0)
        with self.assertRaises(ValueError):
            match_recall_at_k(np.zeros((0, 3)), np.zeros((0, 3)))


class TestMixingAndEntropy(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.Z_sep = np.concatenate([rng.normal(size=(30, 2)), rng.normal(size=(30, 2)) + 50])
        self.Z_mix = rng.normal(size=(60, 2))
        self.labels = np.repeat(["rna", "adt"], 30)

    def test_mixing(self):
        self.assertEqual(modality_mixing(self.Z_sep, self.labels, k=5), 0.0)
        self.assertGreater(modality_mixing(self.Z_mix, self.labels, k=5), 0.3)
        _, per = modality_mixing(self.Z_mix, self.labels, k=5, return_per_sample=True)
        self.assertEqual(per.shape, (60,))

    def test_entropy(self):
        self.assertEqual(modality_entropy(self.Z_sep, self.labels, k=5), 0.0)
        self.assertGreater(modality_entropy(self.Z_mix, self.labels, k=10), 0.6)

    def test_degenerate_inputs(self):
        self.assertEqual(modality_mixing(self.Z_mix[:1], self.labels[:1]), 0.0)
        self.assertEqual(modality_entropy(self.Z_mix, np.zeros(60)), 0.0)
        with self.assertRaises(ValueError):
            modality_mixing(self.Z_mix, self.labels[:10])
        with self.assertRaises(ValueError):
            modality_entropy(self.Z_mix, self.labels[:10])


if __name__ == "__main__":
    unittest.main()
