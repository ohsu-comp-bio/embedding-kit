"""Unit tests for ``embkit.hierarchy``."""

import unittest

import numpy as np

from embkit import hierarchy as H


def _clusters(seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    centres = {"w": (0.0, 0.0), "x": (3.0, 0.0), "y": (6.0, 0.0), "z": (3.0, 3.0)}
    z = np.concatenate([rng.normal(loc=c, scale=0.1, size=(25, 2)) for c in centres.values()]) * scale
    labels = np.repeat(list(centres), 25)
    return z, labels


class TestClusterCentroids(unittest.TestCase):
    def test_centroids_euclidean(self):
        z, labels = _clusters()
        cents, names = H.cluster_centroids(z, labels)
        self.assertEqual(names, ["w", "x", "y", "z"])
        self.assertTrue(np.allclose(cents[1], z[labels == "x"].mean(0)))

    def test_centroids_hyperbolic_shape(self):
        z, labels = _clusters(scale=0.2)
        cents, _ = H.cluster_centroids(z, labels, geometry="hyperbolic")
        self.assertEqual(cents.shape, (4, 2))


class TestClusterTree(unittest.TestCase):
    def test_tree_recovers_layout(self):
        z, labels = _clusters()
        tree = H.cluster_tree(z, labels, root="w")
        self.assertEqual(tree["parents"], {"w": None, "x": "w", "y": "x", "z": "x"})
        self.assertAlmostEqual(tree["edge_length"]["x"], 3.0, delta=0.2)
        self.assertEqual(tree["distances"].shape, (4, 4))

    def test_default_root_is_closest_to_origin(self):
        z, labels = _clusters()
        self.assertIsNone(H.cluster_tree(z, labels)["parents"]["w"])

    def test_hyperbolic_tree(self):
        z, labels = _clusters(scale=0.2)
        tree = H.cluster_tree(z, labels, root="w", geometry="hyperbolic")
        self.assertEqual(tree["parents"]["x"], "w")

    def test_unknown_root(self):
        z, labels = _clusters()
        with self.assertRaises(ValueError):
            H.cluster_tree(z, labels, root="nope")


class TestGraphPseudotime(unittest.TestCase):
    def test_increases_along_line(self):
        z = np.linspace(0, 10, 50)[:, None]
        root = np.zeros(50, dtype=bool)
        root[0] = True
        pt = H.graph_pseudotime(z, root, k=3)
        self.assertEqual(pt[0], 0.0)
        self.assertTrue(np.all(np.diff(pt) > 0))
        self.assertAlmostEqual(pt[-1], 10.0, places=5)

    def test_disconnected_components_get_finite_values(self):
        rng = np.random.default_rng(0)
        z = np.concatenate([rng.normal(size=(15, 2)), rng.normal(size=(15, 2)) + 100])
        root = np.zeros(30, dtype=bool)
        root[0] = True
        pt = H.graph_pseudotime(z, root, k=3, chunk=8)
        self.assertTrue(np.isfinite(pt).all())
        self.assertGreater(pt[15:].min(), 50)

    def test_hyperbolic(self):
        z = np.linspace(0, 1, 20)[:, None].repeat(2, axis=1)
        root = np.arange(20) == 0
        pt = H.graph_pseudotime(z, root, k=2, geometry="hyperbolic")
        self.assertTrue(np.all(np.diff(pt) > 0))

    def test_bad_root_mask(self):
        with self.assertRaises(ValueError):
            H.graph_pseudotime(np.zeros((5, 2)), np.zeros(5, dtype=bool))


class TestTreeAgreement(unittest.TestCase):
    def test_identical_trees(self):
        p = {"r": None, "a": "r", "b": "r", "c": "a"}
        out = H.tree_agreement(p, dict(p))
        self.assertEqual(out["edge_jaccard"], 1.0)
        self.assertAlmostEqual(out["path_spearman"], 1.0)
        self.assertEqual(out["parent_accuracy"], 1.0)

    def test_different_trees(self):
        pa = {"r": None, "a": "r", "b": "r", "c": "a"}
        pb = {"r": None, "a": "r", "b": "a", "c": "a"}
        out = H.tree_agreement(pa, pb)
        self.assertAlmostEqual(out["edge_jaccard"], 2 / 4)
        self.assertEqual(out["parent_accuracy"], 0.75)

    def test_small_tree_spearman_nan(self):
        out = H.tree_agreement({"r": None, "a": "r"}, {"r": None, "a": "r"})
        self.assertTrue(np.isnan(out["path_spearman"]))

    def test_node_mismatch(self):
        with self.assertRaises(ValueError):
            H.tree_agreement({"r": None}, {"q": None})


if __name__ == "__main__":
    unittest.main()
