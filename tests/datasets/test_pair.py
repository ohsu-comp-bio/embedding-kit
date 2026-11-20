"""Unit tests for ``embkit.datasets.pair`` (blooms + PairwiseDataset)."""
import os
import tempfile
import unittest
from unittest import mock

from torch.utils.data import Dataset

from embkit.datasets.pair import (
    DatasetKeyBloom,
    PairBloomFilter,
    PairwiseDataset,
    _canonical_pair,
)


class _ListDataset(Dataset):
    """Minimal in-memory dataset fixture."""
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class TestCanonicalPair(unittest.TestCase):
    def test_symmetry(self):
        self.assertEqual(_canonical_pair("a", "b"), _canonical_pair("b", "a"))
        self.assertEqual(_canonical_pair(1, 2), _canonical_pair(2, 1))

    def test_identity(self):
        self.assertEqual(_canonical_pair("x", "x"), ("x", "x"))

    def test_non_comparable_fallback(self):
        # Mixed types that fail `<` still produce a deterministic,
        # order-insensitive pair.
        a, b = object(), object()
        self.assertEqual(_canonical_pair(a, b), _canonical_pair(b, a))


class TestDatasetKeyBloom(unittest.TestCase):
    def test_membership_with_identity_keys(self):
        b = DatasetKeyBloom(["k1", "k2", "k3"], hash_item=lambda r: r, bloom_capacity=100)
        for k in ("k1", "k2", "k3"):
            self.assertIn(k, b)
        self.assertNotIn("not-in-filter", b)

    def test_membership_with_key_extractor(self):
        rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        b = DatasetKeyBloom(rows, hash_item=lambda r: r["id"], bloom_capacity=100)
        self.assertTrue(b.has("a"))
        self.assertFalse(b.has("zzz"))

    def test_save_load_roundtrip(self):
        b = DatasetKeyBloom(["a", "b", "c"], hash_item=lambda r: r, bloom_capacity=100)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "bloom.bin")
            b.save(p)
            b2 = DatasetKeyBloom([], hash_item=lambda r: r, bloom_file=p)
            for k in ("a", "b", "c"):
                self.assertIn(k, b2)
            self.assertNotIn("zzz", b2)


class TestPairBloomFilter(unittest.TestCase):
    def setUp(self):
        self.rows = ["a", "b", "c"]

    def test_present_pair(self):
        pf = PairBloomFilter(self.rows, hash_item=lambda r: r)
        self.assertTrue(pf.has("a", "b"))
        self.assertTrue(pf.has("b", "c"))

    def test_absent_pair(self):
        pf = PairBloomFilter(self.rows, hash_item=lambda r: r)
        self.assertFalse(pf.has("a", "zzz"))

    def test_symmetry(self):
        pf = PairBloomFilter(self.rows, hash_item=lambda r: r)
        self.assertEqual(pf.has("a", "b"), pf.has("b", "a"))


class TestPairwiseDataset(unittest.TestCase):
    def setUp(self):
        # 3 source datasets of 4 rows each:
        #   d0: a b c d
        #   d1: a b e f
        #   d2: a c g h
        # Known facts for denominator math (N=3):
        #   (a,c): D_both=2 (d0,d2), D_either=3 (d0,d1,d2) -> any=2/3, total=2/3
        #   (e,f): D_both=1 (d1),     D_either=1           -> any=1,   total=1/3
        #   (a,d): D_both=1 (d0),     D_either=3           -> any=1/3, total=1/3
        #   (zzz,qqq): D_both=0, D_either=0 -> 0.0 in both modes
        self.d0 = _ListDataset(["a", "b", "c", "d"])
        self.d1 = _ListDataset(["a", "b", "e", "f"])
        self.d2 = _ListDataset(["a", "c", "g", "h"])
        self.datasets = [self.d0, self.d1, self.d2]

    def _make(self, **kw):
        kw.setdefault("hash_item", lambda r: r)
        return PairwiseDataset(self.datasets, **kw)

    def test_len_matches_n_pairs(self):
        self.assertEqual(len(self._make(n_pairs=37)), 37)

    def test_getitem_returns_3tuple(self):
        ds = self._make(n_pairs=10, seed=42)
        item_a, item_b, pct = ds[0]
        self.assertIsInstance(item_a, str)
        self.assertIsInstance(item_b, str)
        self.assertIsInstance(pct, float)
        self.assertGreaterEqual(pct, 0.0)
        self.assertLessEqual(pct, 1.0)

    def test_negative_index_and_out_of_range(self):
        ds = self._make(n_pairs=5, seed=1)
        self.assertEqual(ds[-1], ds[4])
        with self.assertRaises(IndexError):
            ds[5]
        with self.assertRaises(IndexError):
            ds[-6]

    def test_deterministic_same_seed(self):
        d1 = self._make(n_pairs=50, seed=7)
        d2 = self._make(n_pairs=50, seed=7)
        for i in range(50):
            self.assertEqual(d1[i], d2[i])

    def test_different_seeds_differ(self):
        d1 = self._make(n_pairs=100, seed=1)
        d2 = self._make(n_pairs=100, seed=2)
        diffs = sum(1 for i in range(100) if d1[i] != d2[i])
        self.assertGreater(diffs, 0)

    def test_denominator_any_vs_total(self):
        ds_any = self._make(n_pairs=5, seed=0, denominator="any")
        ds_tot = self._make(n_pairs=5, seed=0, denominator="total")
        self.assertAlmostEqual(ds_any._co_occurrence_pct("a", "c"), 2.0 / 3.0, places=6)
        self.assertAlmostEqual(ds_tot._co_occurrence_pct("a", "c"), 2.0 / 3.0, places=6)
        self.assertAlmostEqual(ds_any._co_occurrence_pct("e", "f"), 1.0, places=6)
        self.assertAlmostEqual(ds_tot._co_occurrence_pct("e", "f"), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(ds_any._co_occurrence_pct("a", "d"), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(ds_tot._co_occurrence_pct("a", "d"), 1.0 / 3.0, places=6)

    def test_neither_present_pct_zero(self):
        ds = self._make(n_pairs=5, seed=0)
        self.assertEqual(ds._co_occurrence_pct("zzz", "qqq"), 0.0)

    def test_cache_roundtrip(self):
        ds = self._make(n_pairs=20, seed=3)
        for i in range(20):
            ds[i]
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "cache.pkl")
            ds.save_cache(p)
            ds2 = self._make(n_pairs=20, seed=3)
            ds2.load_cache(p)
            for i in range(20):
                self.assertEqual(ds[i], ds2[i])

    def test_cache_auto_load_at_construction(self):
        ds = self._make(n_pairs=10, seed=5)
        for i in range(10):
            ds[i]
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "cache.pkl")
            ds.save_cache(p)
            ds2 = self._make(n_pairs=10, seed=5, cache_file=p)
            for i in range(10):
                self.assertEqual(ds[i], ds2[i])

    def test_denominator_mismatch_on_load(self):
        ds_any = self._make(n_pairs=10, seed=0, denominator="any")
        for i in range(5):
            ds_any[i]
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "cache.pkl")
            ds_any.save_cache(p)
            ds_tot = self._make(n_pairs=10, seed=0, denominator="total")
            with self.assertRaises(ValueError):
                ds_tot.load_cache(p)

    def test_n_pairs_mismatch_on_load(self):
        ds = self._make(n_pairs=10, seed=0)
        ds[0]
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "cache.pkl")
            ds.save_cache(p)
            ds2 = self._make(n_pairs=11, seed=0)
            with self.assertRaises(ValueError):
                ds2.load_cache(p)

    def test_negative_examples_with_pair_bloom(self):
        # Pairs internal to d0 (a,b,c,d) are positive; any pair with a key
        # outside d0 (e, f, g, h) is a negative example.
        pb = PairBloomFilter(self.d0, hash_item=lambda r: r)
        ds = PairwiseDataset(self.datasets, n_pairs=200,
                             hash_item=lambda r: r, seed=0, pairwise_bloom=pb)
        negatives = [i for i in range(200) if ds.is_negative(i)]
        self.assertGreater(len(negatives), 0)
        positives = [i for i in range(200) if not ds.is_negative(i)]
        self.assertGreater(len(positives), 0)
        for i in negatives:
            _, _, pct = ds[i]
            self.assertEqual(pct, 0.0)

    def test_is_negative_false_without_pair_bloom(self):
        ds = self._make(n_pairs=10, seed=0)
        for i in range(10):
            self.assertFalse(ds.is_negative(i))

    def test_worker_determinism(self):
        class _FakeWI:
            def __init__(self, id):
                self.id = id

        with mock.patch("torch.utils.data.get_worker_info", return_value=_FakeWI(0)):
            plan_w0 = list(PairwiseDataset(self.datasets, n_pairs=50,
                                           hash_item=lambda r: r, seed=11)._pair_plan)
        with mock.patch("torch.utils.data.get_worker_info", return_value=_FakeWI(1)):
            plan_w1 = list(PairwiseDataset(self.datasets, n_pairs=50,
                                           hash_item=lambda r: r, seed=11)._pair_plan)
        with mock.patch("torch.utils.data.get_worker_info", return_value=_FakeWI(0)):
            plan_w0_again = list(PairwiseDataset(self.datasets, n_pairs=50,
                                                 hash_item=lambda r: r, seed=11)._pair_plan)
        self.assertNotEqual(plan_w0, plan_w1)
        self.assertEqual(plan_w0, plan_w0_again)

    def test_worker_seeds_are_distinct_from_main_process(self):
        class _FakeWI:
            def __init__(self, id):
                self.id = id

        with mock.patch("torch.utils.data.get_worker_info", return_value=None):
            plan_main = list(PairwiseDataset(self.datasets, n_pairs=50,
                                             hash_item=lambda r: r, seed=11)._pair_plan)
        with mock.patch("torch.utils.data.get_worker_info", return_value=_FakeWI(0)):
            plan_w0 = list(PairwiseDataset(self.datasets, n_pairs=50,
                                           hash_item=lambda r: r, seed=11)._pair_plan)
        self.assertNotEqual(plan_main, plan_w0)

    def test_reuses_dataset_key_bloom_instance(self):
        # A dataset that already exposes .has() is used directly.
        b = DatasetKeyBloom(["a", "b"], hash_item=lambda r: r, bloom_capacity=50)
        d = _ListDataset(["a", "b"])
        ds = PairwiseDataset([d], n_pairs=4,
                             hash_item=lambda r: r, seed=0)
        self.assertIsNot(ds._key_blooms[0], b)  # plain dataset gets a new bloom
        # But if the dataset itself IS a DatasetKeyBloom, it is reused.
        ds2 = PairwiseDataset([b], n_pairs=4, hash_item=lambda r: r, seed=0)
        self.assertIs(ds2._key_blooms[0], b)

    def test_error_on_empty_datasets(self):
        with self.assertRaises(ValueError):
            PairwiseDataset([], n_pairs=10)

    def test_error_on_invalid_denominator(self):
        with self.assertRaises(ValueError):
            self._make(n_pairs=10, denominator="bogus")

    def test_error_on_invalid_n_pairs(self):
        with self.assertRaises(ValueError):
            PairwiseDataset(self.datasets, n_pairs=0)

    def test_error_on_dataset_without_len(self):
        class NoLen:
            def __iter__(self):
                return iter(["a", "b"])

        with self.assertRaises(ValueError):
            PairwiseDataset([NoLen()], n_pairs=10)

    def test_error_on_too_few_rows(self):
        with self.assertRaises(ValueError):
            PairwiseDataset([_ListDataset(["solo"])], n_pairs=10)

    def test_works_with_mixed_sized_datasets(self):
        big = _ListDataset([f"k{i}" for i in range(20)])
        small = _ListDataset(["x", "y"])
        ds = PairwiseDataset([big, small], n_pairs=50, hash_item=lambda r: r, seed=0)
        for i in range(50):
            a, b, pct = ds[i]
            self.assertIsInstance(a, str)
            self.assertIsInstance(pct, float)
            self.assertGreaterEqual(pct, 0.0)
            self.assertLessEqual(pct, 1.0)


if __name__ == "__main__":
    unittest.main()
