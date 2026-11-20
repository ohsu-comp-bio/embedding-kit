"""Unit tests for ``embkit.metrics.neighbors``."""

import unittest

import numpy as np
import pandas as pd

from embkit.metrics.neighbors import label_transfer_knn, transfer_labels, label_purity, batch_mixing


def _two_clusters(seed=0, offset=20.0):
    rng = np.random.default_rng(seed)
    z = np.concatenate([rng.normal(size=(25, 3)), rng.normal(size=(25, 3)) + offset])
    labels = np.repeat(["T", "B"], 25)
    return z, labels


class TestLabelTransferKNN(unittest.TestCase):
    def test_perfect_transfer(self):
        z, labels = _two_clusters()
        pred, acc, cm, order, f1 = label_transfer_knn(z, labels, z + 0.01, labels, k=5,
                                                      return_label_order=True, return_f1=True)
        self.assertEqual(acc, 1.0)
        self.assertTrue((pred == labels).all())
        self.assertEqual(list(order), ["B", "T"])
        self.assertEqual(cm.trace(), 50)
        self.assertEqual(f1["macro_f1"], 1.0)

    def test_without_targets(self):
        z, labels = _two_clusters()
        pred, acc, cm = label_transfer_knn(z, labels, z[:5], k=3)
        self.assertIsNone(acc)
        self.assertEqual(cm.size, 0)
        self.assertEqual(len(pred), 5)

    def test_minimal_return_with_targets(self):
        z, labels = _two_clusters()
        self.assertEqual(len(label_transfer_knn(z, labels, z, labels)), 3)

    def test_empty_source(self):
        with self.assertRaises(ValueError):
            label_transfer_knn(np.zeros((0, 3)), [], np.zeros((2, 3)))


class TestTransferLabels(unittest.TestCase):
    def test_labels_confidence_and_novelty(self):
        z, labels = _two_clusters()
        query = np.vstack([z[:3] + 0.01, np.full((1, 3), 500.0)])
        out = transfer_labels(z, labels, query, k=5)
        self.assertIsInstance(out, pd.DataFrame)
        self.assertEqual(list(out["label"][:3]), ["T", "T", "T"])
        self.assertTrue((out["confidence"][:3] == 1.0).all())
        self.assertGreater(out["novelty"].iloc[3], 10 * out["novelty"].iloc[:3].max())

    def test_hyperbolic_geometry(self):
        z, labels = _two_clusters(offset=3.0)
        out = transfer_labels(z * 0.2, labels, z[:4] * 0.2, k=3, geometry="hyperbolic")
        self.assertEqual(len(out), 4)


class TestPurityAndMixing(unittest.TestCase):
    def test_purity(self):
        z, labels = _two_clusters()
        self.assertEqual(label_purity(z, labels, k=5), 1.0)
        self.assertEqual(label_purity(z * 0.05, labels, k=5, geometry="hyperbolic"), 1.0)

    def test_mixing_separate_vs_mixed(self):
        z, labels = _two_clusters()
        self.assertEqual(batch_mixing(z, labels, k=5), 0.0)
        rng = np.random.default_rng(3)
        self.assertGreater(batch_mixing(rng.normal(size=(50, 3)), labels, k=10), 0.7)

    def test_mixing_product_geometry(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=(40, 4)) * 0.3
        val = batch_mixing(z, np.repeat([0, 1], 20), k=5, geometry="product", hyperbolic_dim=2)
        self.assertGreater(val, 0.5)

    def test_duplicate_points_exclude_self(self):
        z = np.zeros((10, 2))
        labels = np.repeat(["a", "b"], 5)
        self.assertGreaterEqual(label_purity(z, labels, k=3), 0.0)

    def test_errors(self):
        z, labels = _two_clusters()
        with self.assertRaises(ValueError):
            batch_mixing(z, np.zeros(50))
        with self.assertRaises(ValueError):
            label_purity(z[:1], labels[:1])


if __name__ == "__main__":
    unittest.main()
