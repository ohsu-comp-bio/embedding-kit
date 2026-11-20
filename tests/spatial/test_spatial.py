"""Unit tests for ``embkit.spatial``."""

import unittest

import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader, TensorDataset
import torch

from embkit import spatial as S


def _grid(n=6):
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


class TestKnnGraph(unittest.TestCase):
    def test_excludes_self(self):
        idx, dist = S.knn_graph(_grid(), k=4)
        self.assertEqual(idx.shape, (36, 4))
        self.assertFalse((idx == np.arange(36)[:, None]).any())
        # interior grid point has four neighbours at distance 1
        self.assertTrue(np.allclose(dist[7], 1.0))

    def test_duplicate_coordinates(self):
        coords = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0], [9.0, 9.0]])
        idx, dist = S.knn_graph(coords, k=1)
        self.assertEqual(idx[0, 0], 1)
        self.assertEqual(dist[0, 0], 0.0)

    def test_median_distance(self):
        self.assertEqual(S.median_neighbor_distance(_grid()), 1.0)

    def test_errors(self):
        with self.assertRaises(ValueError):
            S.knn_graph(np.zeros(5), 1)
        with self.assertRaises(ValueError):
            S.knn_graph(_grid(), k=36)


class TestNeighborhood(unittest.TestCase):
    def test_weights_row_normalized(self):
        for kernel in ("uniform", "gaussian"):
            W = S.neighborhood_weights(_grid(), k=4, kernel=kernel, include_self=True)
            self.assertTrue(np.allclose(np.asarray(W.sum(1)).ravel(), 1.0))

    def test_aggregate_constant_is_constant(self):
        X = np.full((36, 3), 2.5)
        self.assertTrue(np.allclose(S.neighborhood_aggregate(X, _grid(), k=4), 2.5))

    def test_aggregate_sparse(self):
        X = sp.random(36, 5, density=0.3, format="csr", random_state=0)
        out = S.neighborhood_aggregate(X, _grid(), k=3)
        self.assertTrue(sp.issparse(out))
        dense = S.neighborhood_aggregate(X.toarray(), _grid(), k=3)
        self.assertTrue(np.allclose(out.toarray(), dense))

    def test_aggregate_groups_do_not_mix(self):
        coords = np.vstack([_grid(), _grid()])  # two slides on top of each other
        X = np.vstack([np.zeros((36, 2)), np.ones((36, 2))])
        groups = np.repeat(["s1", "s2"], 36)
        out = S.neighborhood_aggregate(X, coords, k=4, groups=groups)
        self.assertTrue(np.allclose(out[:36], 0.0) and np.allclose(out[36:], 1.0))
        out_sp = S.neighborhood_aggregate(sp.csr_matrix(X), coords, k=4, groups=groups)
        self.assertTrue(np.allclose(out_sp.toarray(), out))

    def test_kernel_errors(self):
        with self.assertRaises(ValueError):
            S.neighborhood_weights(_grid(), kernel="cosine")
        coords = np.zeros((5, 2))
        with self.assertRaises(ValueError):
            S.neighborhood_weights(coords, k=2, kernel="gaussian")


class TestTiles(unittest.TestCase):
    def setUp(self):
        self.img = (np.random.default_rng(0).uniform(size=(40, 50, 3)) * 255).astype(np.uint8)

    def test_tile_content_and_layout(self):
        tiles = S.extract_tiles(self.img, [[25.0, 20.0]], tile_size=4, flatten=False)
        self.assertEqual(tiles.shape, (1, 3, 4, 4))
        expected = self.img[18:22, 23:27].transpose(2, 0, 1) / 255.0
        self.assertTrue(np.allclose(tiles[0], expected, atol=1e-6))

    def test_flatten_and_unflatten(self):
        flat = S.extract_tiles(self.img, [[10, 10], [49, 39]], tile_size=8)
        self.assertEqual(flat.shape, (2, 3 * 8 * 8))
        self.assertEqual(S.unflatten_tiles(sp.csr_matrix(flat), (3, 8, 8)).shape, (2, 3, 8, 8))

    def test_resize(self):
        self.assertEqual(S.extract_tiles(self.img, [[20, 20]], 8, out_size=4).shape, (1, 48))
        self.assertEqual(S.extract_tiles(self.img, [[20, 20]], 8, out_size=5).shape, (1, 75))

    def test_grayscale_float_and_small_image(self):
        img = np.random.default_rng(0).uniform(size=(6, 6))
        self.assertEqual(S.extract_tiles(img, [[3, 3]], 8, flatten=False).shape, (1, 1, 8, 8))

    def test_errors(self):
        with self.assertRaises(ValueError):
            S.extract_tiles(np.zeros((2, 2, 2, 2)), [[0, 0]], 2)
        with self.assertRaises(ValueError):
            S.extract_tiles(self.img, [[0, 0]], 0)
        with self.assertRaises(ValueError):
            S.extract_tiles(self.img, [0, 0], 2)
        with self.assertRaises(ValueError):
            S.extract_tiles(self.img, [[100, 0]], 2)


class TestSpatialBlockSampler(unittest.TestCase):
    def test_covers_all_points_with_local_batches(self):
        coords = _grid(8)
        sampler = S.SpatialBlockSampler(coords, batch_size=9, seed=0)
        batches = list(sampler)
        seen = set(i for b in batches for i in b)
        self.assertEqual(seen, set(range(64)))
        self.assertEqual(len(sampler), 8)
        for b in batches:
            spread = coords[b].max(0) - coords[b].min(0)
            self.assertTrue((spread <= 4).all())

    def test_groups_and_epochs(self):
        coords = np.vstack([_grid(4), _grid(4)])
        groups = np.repeat(["a", "b"], 16)
        sampler = S.SpatialBlockSampler(coords, batch_size=4, groups=groups, seed=1)
        first = list(sampler)
        for b in first:
            self.assertEqual(len(set(groups[b])), 1)
        self.assertNotEqual(first, list(sampler))  # new epoch, new order

    def test_singleton_group_skipped(self):
        coords = np.vstack([_grid(3), [[100.0, 100.0]]])
        sampler = S.SpatialBlockSampler(coords, batch_size=4, groups=["a"] * 9 + ["b"])
        self.assertNotIn([9], list(sampler))

    def test_with_dataloader(self):
        coords = _grid(4)
        loader = DataLoader(TensorDataset(torch.arange(16)), batch_sampler=S.SpatialBlockSampler(coords, 4))
        self.assertEqual(sum(len(b[0]) for b in loader) >= 16, True)

    def test_bad_batch_size(self):
        with self.assertRaises(ValueError):
            S.SpatialBlockSampler(_grid(), batch_size=1)


if __name__ == "__main__":
    unittest.main()
