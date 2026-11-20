"""Unit tests for ``embkit.geometry`` (hyperbolic / product latent geometry)."""

import unittest

import numpy as np
import torch

from embkit import geometry as G


class TestSplitLatentDims(unittest.TestCase):
    def test_euclidean(self):
        self.assertEqual(G.split_latent_dims(8, "euclidean"), (8, 0))

    def test_hyperbolic(self):
        self.assertEqual(G.split_latent_dims(8, "hyperbolic"), (0, 8))

    def test_product_default_half_rounded_up(self):
        self.assertEqual(G.split_latent_dims(7, "product"), (3, 4))

    def test_product_explicit(self):
        self.assertEqual(G.split_latent_dims(8, "Product", hyperbolic_dim=2), (6, 2))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            G.split_latent_dims(8, "spherical")
        with self.assertRaises(ValueError):
            G.split_latent_dims(8, "euclidean", hyperbolic_dim=2)
        with self.assertRaises(ValueError):
            G.split_latent_dims(8, "hyperbolic", hyperbolic_dim=2)
        with self.assertRaises(ValueError):
            G.split_latent_dims(8, "product", hyperbolic_dim=8)


class TestLorentzPrimitives(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.v = torch.randn(10, 4, dtype=torch.float64)

    def test_expmap_on_hyperboloid(self):
        x = G.expmap0(self.v)
        # Lorentz norm <x, x>_L = -x0^2 + |xb|^2 = -1
        lorentz = -x[:, 0] ** 2 + (x[:, 1:] ** 2).sum(-1)
        self.assertTrue(torch.allclose(lorentz, -torch.ones(10, dtype=torch.float64), atol=1e-8))

    def test_logmap_inverts_expmap(self):
        self.assertTrue(torch.allclose(G.logmap0(G.expmap0(self.v)), self.v, atol=1e-8))

    def test_distance_from_origin_is_tangent_norm(self):
        origin = G.expmap0(torch.zeros(1, 4, dtype=torch.float64))
        d = G.lorentz_distance(G.expmap0(self.v), origin)
        self.assertTrue(torch.allclose(d, self.v.norm(dim=-1), atol=1e-6))

    def test_boost_is_isometry_and_moves_center_to_origin(self):
        c = G.expmap0(self.v[:1])
        x, y = G.expmap0(self.v[1:5]), G.expmap0(self.v[5:9])
        bx, by = G.boost(c, x), G.boost(c, y)
        self.assertTrue(torch.allclose(G.lorentz_distance(bx, by), G.lorentz_distance(x, y), atol=1e-6))
        self.assertTrue(torch.allclose(G.logmap0(G.boost(c, c)), torch.zeros(1, 4, dtype=torch.float64), atol=1e-6))
        # inverse boost undoes the boost
        self.assertTrue(torch.allclose(G.boost(c, bx, inverse=True), x, atol=1e-6))

    def test_centroid_of_identical_points(self):
        x = G.expmap0(self.v[:1]).repeat(5, 1)
        self.assertTrue(torch.allclose(G.lorentz_centroid(x), x[0], atol=1e-8))

    def test_log_sinhc_small_and_large(self):
        r = torch.tensor([1e-5, 0.5, 5.0, 50.0], dtype=torch.float64)
        expected = torch.log(torch.sinh(r[:3]) / r[:3])
        self.assertTrue(torch.allclose(G.log_sinhc(r)[:3], expected, atol=1e-8))
        self.assertTrue(torch.isfinite(G.log_sinhc(r)).all())


class TestWrappedNormal(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.mu = 0.5 * torch.randn(6, 3)
        self.logvar = 0.3 * torch.randn(6, 3)

    def test_rsample_shapes_and_dtype(self):
        t, log_q = G.wrapped_normal_rsample(self.mu, self.logvar, curvature=0.5)
        self.assertEqual(t.shape, (6, 3))
        self.assertEqual(log_q.shape, (6,))
        self.assertEqual(t.dtype, self.mu.dtype)

    def test_log_prob_of_sample_matches_log_q(self):
        t, log_q = G.wrapped_normal_rsample(self.mu, self.logvar)
        lp = G.wrapped_normal_log_prob(t, self.mu, self.logvar)
        self.assertTrue(torch.allclose(lp, log_q, atol=1e-4))

    def test_log_prob_reduces_to_gaussian_in_one_dim(self):
        # in 1-D the sinhc correction vanishes and the wrapped normal is a Gaussian
        t = torch.tensor([[0.3]], dtype=torch.float64)
        mu = torch.tensor([[0.1]], dtype=torch.float64)
        lv = torch.tensor([[0.2]], dtype=torch.float64)
        expected = torch.distributions.Normal(mu, torch.exp(0.5 * lv)).log_prob(t).sum(-1)
        self.assertTrue(torch.allclose(G.wrapped_normal_log_prob(t, mu, lv), expected, atol=1e-8))

    def test_kl_of_identical_distributions_is_zero(self):
        kl = G.wrapped_normal_kl(self.mu, self.logvar, self.mu, self.logvar)
        self.assertTrue(torch.allclose(kl, torch.zeros(6), atol=1e-4))

    def test_kl_default_prior_is_finite(self):
        self.assertTrue(torch.isfinite(G.wrapped_normal_kl(self.mu, self.logvar)).all())

    def test_bad_curvature(self):
        with self.assertRaises(ValueError):
            G.wrapped_normal_rsample(self.mu, self.logvar, curvature=0.0)


class TestArrayHelpers(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.a = rng.normal(size=(12, 4))
        self.b = rng.normal(size=(12, 4))

    def test_euclidean_distance_matches_numpy(self):
        d = G.latent_distance(self.a, self.b)
        self.assertTrue(np.allclose(d, np.linalg.norm(self.a - self.b, axis=1), atol=1e-6))

    def test_hyperbolic_distance_from_origin_equals_radius(self):
        d = G.latent_distance(self.a, np.zeros_like(self.a), "hyperbolic")
        self.assertTrue(np.allclose(d, G.latent_radius(self.a), atol=1e-5))

    def test_curvature_scales_distance(self):
        # radius stays the tangent norm; distance between two points depends on c
        d1 = G.latent_distance(self.a, self.b, "hyperbolic", curvature=1.0)
        d2 = G.latent_distance(self.a, self.b, "hyperbolic", curvature=4.0)
        self.assertFalse(np.allclose(d1, d2))

    def test_product_distance_combines_blocks(self):
        d = G.latent_distance(self.a, self.b, "product", hyperbolic_dim=2)
        de = np.linalg.norm(self.a[:, :2] - self.b[:, :2], axis=1)
        dh = G.latent_distance(self.a[:, 2:], self.b[:, 2:], "hyperbolic")
        self.assertTrue(np.allclose(d, np.sqrt(de ** 2 + dh ** 2), atol=1e-5))

    def test_pairwise_matches_rowwise(self):
        for geo in ("euclidean", "hyperbolic", "product"):
            D = G.pairwise_latent_distance(self.a, self.b, geo, chunk=5)
            self.assertEqual(D.shape, (12, 12))
            self.assertTrue(np.allclose(np.diag(D), G.latent_distance(self.a, self.b, geo), atol=1e-5))

    def test_pairwise_self_symmetric_zero_diagonal(self):
        D = G.pairwise_latent_distance(torch.as_tensor(self.a), geometry="hyperbolic", chunk=5)
        self.assertTrue(np.allclose(D, D.T, atol=1e-6))
        self.assertTrue(np.allclose(np.diag(D), 0.0, atol=1e-5))

    def test_centroid(self):
        self.assertTrue(np.allclose(G.latent_centroid(self.a), self.a.mean(0)))
        same = np.repeat(self.a[:1], 4, axis=0)
        self.assertTrue(np.allclose(G.latent_centroid(same, "hyperbolic"), self.a[0], atol=1e-6))
        with self.assertRaises(ValueError):
            G.latent_centroid(np.zeros((0, 4)))

    def test_poincare_inside_unit_ball(self):
        p = G.to_poincare(self.a, "product", hyperbolic_dim=2)
        self.assertEqual(p.shape, (12, 2))
        self.assertTrue((np.linalg.norm(p, axis=1) < 1.0).all())
        with self.assertRaises(ValueError):
            G.to_poincare(self.a, "euclidean")

    def test_rows_per_chunk(self):
        self.assertEqual(G._rows_per_chunk(10, 1024), 1024)
        self.assertEqual(G._rows_per_chunk(10 ** 9, 1024), 8)


class TestTrees(unittest.TestCase):
    def setUp(self):
        self.parents = {"root": None, "a": "root", "b": "root", "a1": "a", "a2": "a"}

    def test_tree_distance_matrix(self):
        dist, names, depth = G.tree_distance_matrix(self.parents, lengths={"a1": 2.0})
        idx = {n: i for i, n in enumerate(names)}
        self.assertEqual(dist[idx["a1"], idx["a2"]], 3.0)
        self.assertEqual(dist[idx["a1"], idx["b"]], 4.0)
        self.assertTrue(np.allclose(dist, dist.T))
        self.assertEqual(depth[idx["a1"]], 3.0)

    def test_tree_order(self):
        order = ["a2", "a1", "a", "b", "root"]
        _, names, _ = G.tree_distance_matrix(self.parents, order=order)
        self.assertEqual(names, order)

    def test_tree_errors(self):
        with self.assertRaises(ValueError):
            G.tree_distance_matrix({"a": None, "b": None})
        with self.assertRaises(ValueError):
            G.tree_distance_matrix({"r": None, "a": "b", "b": "a"})
        with self.assertRaises(ValueError):
            G.tree_distance_matrix({"r": None, "a": "missing"})
        with self.assertRaises(ValueError):
            G.tree_distance_matrix(self.parents, order=["root"])

    def test_hierarchy_metrics_perfect_embedding(self):
        # place each sample exactly at its label's position on a line tree: root - a - a1
        parents = {"root": None, "a": "root", "a1": "a"}
        dist, names, depth = G.tree_distance_matrix(parents)
        labels = np.repeat(np.arange(3), 20)
        z = np.zeros((60, 2))
        z[:, 0] = depth[labels]
        out = G.hierarchy_metrics(z, labels, dist, depth, n_pairs=2000)
        self.assertGreater(out["dist_spearman"], 0.99)
        self.assertLess(out["rel_distortion"], 1e-3)
        self.assertGreater(out["radius_depth"], 0.99)

    def test_hierarchy_metrics_without_depth(self):
        dist, _, _ = G.tree_distance_matrix(self.parents)
        rng = np.random.default_rng(0)
        out = G.hierarchy_metrics(rng.normal(size=(50, 3)), rng.integers(0, 5, 50), dist,
                                  geometry="hyperbolic", n_pairs=200)
        self.assertEqual(set(out), {"dist_spearman", "rel_distortion"})


if __name__ == "__main__":
    unittest.main()
