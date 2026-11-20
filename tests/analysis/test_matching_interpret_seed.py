"""Unit tests for ``embkit.matching``, ``embkit.interpret`` and ``embkit.utilities.seed``."""

import random
import unittest

import numpy as np
import torch

from embkit.interpret import integrated_gradients
from embkit.matching import bipartite_match, mnn_anchors
from embkit.utilities import set_seed


class TestBipartiteMatch(unittest.TestCase):
    def test_recovers_permutation(self):
        rng = np.random.default_rng(0)
        XA = rng.normal(size=(20, 3))
        perm = rng.permutation(20)
        XB = XA[perm] + 0.001
        ia, ib = bipartite_match(XA, XB)
        self.assertTrue(np.array_equal(perm[ib], ia))

    def test_subsampling(self):
        rng = np.random.default_rng(0)
        ia, ib = bipartite_match(rng.normal(size=(30, 2)), rng.normal(size=(25, 2)), max_samples=10)
        self.assertEqual(len(ia), 10)
        self.assertEqual(len(set(ia.tolist())), 10)


class TestMNNAnchors(unittest.TestCase):
    def test_paired_points_are_anchors(self):
        rng = np.random.default_rng(1)
        XA = rng.normal(size=(15, 2)) * 10
        ia, ib = mnn_anchors(XA, XA + 0.001, k=1)
        self.assertTrue(np.array_equal(ia, ib))
        self.assertEqual(len(ia), 15)

    def test_only_mutual_pairs_kept(self):
        XA = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
        XB = np.array([[100.0, 0.0], [10.0, 0.0]])
        # every A point's nearest B is [10, 0], whose nearest A is [0.2, 0]: one mutual pair
        ia, ib = mnn_anchors(XA, XB, k=1)
        self.assertEqual((ia.tolist(), ib.tolist()), ([2], [1]))

    def test_larger_k_adds_anchors(self):
        rng = np.random.default_rng(2)
        XA, XB = rng.normal(size=(20, 2)), rng.normal(size=(20, 2))
        self.assertGreaterEqual(len(mnn_anchors(XA, XB, k=5)[0]), len(mnn_anchors(XA, XB, k=1)[0]))


class TestIntegratedGradients(unittest.TestCase):
    def test_linear_model_is_exact(self):
        w = torch.tensor([1.0, -2.0, 0.5])
        x = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.0, -1.0]])
        attr = integrated_gradients(lambda z: z @ w, x, steps=8)
        self.assertTrue(torch.allclose(attr, x * w))

    def test_completeness_with_baseline(self):
        torch.manual_seed(0)
        W = torch.randn(4, 1)
        f = lambda z: torch.tanh(z @ W).squeeze(-1)
        x, base = torch.randn(3, 4), torch.randn(3, 4)
        attr = integrated_gradients(f, x, baseline=base, steps=256)
        self.assertTrue(torch.allclose(attr.sum(1), f(x) - f(base), atol=1e-2))

    def test_scalar_output(self):
        attr = integrated_gradients(lambda z: (z ** 2).sum(), torch.ones(2, 2), steps=4)
        self.assertEqual(attr.shape, (2, 2))


class TestSetSeed(unittest.TestCase):
    def test_reproducible(self):
        set_seed(123)
        a = (random.random(), np.random.rand(), torch.rand(1).item())
        set_seed(123, deterministic=True)
        b = (random.random(), np.random.rand(), torch.rand(1).item())
        self.assertEqual(a, b)
        self.assertTrue(torch.backends.cudnn.deterministic)
        torch.backends.cudnn.deterministic = False


if __name__ == "__main__":
    unittest.main()
