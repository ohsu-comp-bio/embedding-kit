"""Unit tests for ``embkit.preprocessing`` transforms, ``split_by_label`` and Macenko stain normalization."""

import unittest

import numpy as np
import pandas as pd

from embkit.preprocessing import (
    library_size_norm, clr_norm, TfidfLSI, split_by_label, MacenkoNormalizer, rgb_to_od, od_to_rgb,
)


def _counts(n=40, m=25, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.poisson(1.5, size=(n, m)).astype(float)
    return pd.DataFrame(x, index=[f"s{i}" for i in range(n)], columns=[f"f{j}" for j in range(m)])


class TestLibrarySizeNorm(unittest.TestCase):
    def test_totals_before_log(self):
        df = _counts()
        out = library_size_norm(df, target_sum=100.0, log1p=False)
        self.assertIsInstance(out, pd.DataFrame)
        self.assertTrue(np.allclose(out.sum(axis=1), 100.0))
        self.assertTrue(out.index.equals(df.index))

    def test_log1p_on_array(self):
        x = _counts().to_numpy()
        out = library_size_norm(x)
        self.assertIsInstance(out, np.ndarray)
        self.assertTrue(np.allclose(np.expm1(out).sum(axis=1), 1e4))

    def test_rejects_bad_counts(self):
        with self.assertRaises(ValueError):
            library_size_norm(np.array([[1.0, -1.0]]))
        with self.assertRaises(ValueError):
            library_size_norm(np.array([[1.0, np.nan]]))
        with self.assertRaises(ValueError):
            library_size_norm(np.array([1.0, 2.0]))


class TestCLR(unittest.TestCase):
    def test_rows_centred(self):
        out = clr_norm(_counts())
        self.assertTrue(np.allclose(out.mean(axis=1), 0.0))

    def test_invariant_to_constant_shift_in_log_space(self):
        x = np.array([[1.0, 3.0, 7.0]])
        y = np.exp(np.log1p(x) + 1.0) - 1.0
        self.assertTrue(np.allclose(clr_norm(x), clr_norm(y)))


class TestTfidfLSI(unittest.TestCase):
    def test_methods_shapes(self):
        df = _counts()
        for method in ("sklearn", "signac", "tea"):
            with self.subTest(method=method):
                lsi = TfidfLSI(n_components=6, method=method, drop_first=True).fit(df)
                out = lsi.transform(df)
                self.assertEqual(out.shape, (40, 5))
                self.assertEqual(list(out.columns), [f"LSI_{i}" for i in range(1, 6)])
                self.assertTrue(np.allclose(out.mean(axis=0), 0.0, atol=1e-5))

    def test_array_input_and_no_scaling(self):
        x = _counts().to_numpy()
        out = TfidfLSI(n_components=4, scale=False).fit_transform(x)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (40, 4))

    def test_feature_filter(self):
        x = _counts().to_numpy()
        x[:, 0] = 0.0  # never detected
        lsi = TfidfLSI(n_components=3, max_fraction=0.99).fit(x)
        self.assertFalse(lsi.features_[0])
        self.assertEqual(lsi.transform(x[:5]).shape, (5, 3))

    def test_errors(self):
        x = _counts().to_numpy()
        with self.assertRaises(ValueError):
            TfidfLSI(method="bm25").fit(x)
        with self.assertRaises(ValueError):
            TfidfLSI(min_fraction=0.5, max_fraction=0.4).fit(x)
        with self.assertRaises(ValueError):
            TfidfLSI(n_components=100).fit(x)
        with self.assertRaises(ValueError):
            TfidfLSI(n_components=1, drop_first=True).fit(x)
        with self.assertRaises(RuntimeError):
            TfidfLSI().transform(x)
        with self.assertRaises(ValueError):
            TfidfLSI(n_components=3).fit(x).transform(x[:, :5])


class TestSplitByLabel(unittest.TestCase):
    def setUp(self):
        self.labels = np.repeat(["a", "b", "c"], [50, 30, 20])

    def test_partition_and_stratification(self):
        out = split_by_label(self.labels, train_fraction=0.6, val_fraction=0.2, seed=1)
        allidx = np.concatenate([out["train"], out["val"], out["test"]])
        self.assertEqual(sorted(allidx.tolist()), list(range(100)))
        self.assertEqual((self.labels[out["train"]] == "a").sum(), 30)
        self.assertEqual((self.labels[out["val"]] == "c").sum(), 4)

    def test_deterministic(self):
        a, b = split_by_label(self.labels, seed=5), split_by_label(self.labels, seed=5)
        self.assertTrue(all(np.array_equal(a[k], b[k]) for k in a))

    def test_caps(self):
        out = split_by_label(self.labels, train_cap=5, val_cap=1)
        self.assertEqual(len(out["train"]), 15)
        self.assertEqual(len(out["val"]), 3)
        out = split_by_label(self.labels, max_per_label=10)
        self.assertEqual(len(out["train"]), 24)
        self.assertEqual(len(out["test"]) + len(out["val"]) + len(out["train"]), 100)

    def test_errors(self):
        with self.assertRaises(ValueError):
            split_by_label(self.labels, max_per_label=5, train_cap=2)
        with self.assertRaises(ValueError):
            split_by_label(np.array(["a", None], dtype=object))
        with self.assertRaises(ValueError):
            split_by_label(self.labels, train_fraction=0.9, val_fraction=0.1)


class TestMacenko(unittest.TestCase):
    def _slide(self, seed, tint):
        """Synthetic H&E-like tiles: mixtures of two stain colours in optical density space."""
        rng = np.random.default_rng(seed)
        stains = np.array([[0.65, 0.07], [0.70, 0.99], [0.29, 0.11]]) * np.asarray(tint)[:, None]
        conc = rng.gamma(2.0, 0.4, size=(2, 4 * 16 * 16))
        od = (stains @ conc).T
        return od_to_rgb(od).reshape(4, 16, 16, 3).transpose(0, 3, 1, 2)

    def test_od_roundtrip(self):
        rgb = np.random.default_rng(0).uniform(0.05, 1.0, size=(10, 3))
        self.assertTrue(np.allclose(od_to_rgb(rgb_to_od(rgb)), rgb))

    def test_fit_and_transform(self):
        target, source = self._slide(0, (1.0, 1.0, 1.0)), self._slide(1, (1.3, 0.8, 1.1))
        norm = MacenkoNormalizer().fit(target)
        self.assertEqual(norm.stain_matrix.shape, (3, 2))
        self.assertTrue(np.allclose(np.linalg.norm(norm.stain_matrix, axis=0), 1.0))
        out = norm.transform(source)
        self.assertEqual(out.shape, source.shape)
        self.assertEqual(out.dtype, np.float32)
        # normalized source is closer in mean colour to the target than the raw source is
        before = np.abs(source.mean(axis=(0, 2, 3)) - target.mean(axis=(0, 2, 3))).sum()
        after = np.abs(out.mean(axis=(0, 2, 3)) - target.mean(axis=(0, 2, 3))).sum()
        self.assertLess(after, before)

    def test_single_tile_and_subsampling(self):
        tile = self._slide(2, (1.0, 1.0, 1.0))[0]
        norm = MacenkoNormalizer(max_pixels=150).fit(tile)
        self.assertEqual(norm.transform(tile).shape, tile.shape)

    def test_errors(self):
        with self.assertRaises(RuntimeError):
            MacenkoNormalizer().transform(self._slide(0, (1, 1, 1)))
        with self.assertRaises(ValueError):
            MacenkoNormalizer().fit(np.ones((2, 3, 8, 8)))  # all background
        with self.assertRaises(ValueError):
            MacenkoNormalizer().fit(np.zeros((2, 4, 8, 8)))
        with self.assertRaises(ValueError):
            MacenkoNormalizer().fit(np.full((2, 3, 8, 8), 2.0))


if __name__ == "__main__":
    unittest.main()
