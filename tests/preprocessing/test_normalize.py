import os
import unittest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from embkit.preprocessing import (
    quantile_max_norm,
    exp_max_norm,
    ExpMinMaxScaler,
    get_dataset_nonzero_mask,
)


class TestNormalizationUtils(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9]
        }, index=["g1", "g2", "g3"])

    def test_quantile_max_norm_basic(self):
        norm_df = quantile_max_norm(self.df, quantile_max=0.9)
        self.assertIsInstance(norm_df, pd.DataFrame)
        self.assertEqual(norm_df.shape, self.df.shape)
        self.assertTrue((norm_df <= 1.0).all().all())
        self.assertTrue((norm_df >= 0.0).all().all())

    def test_quantile_max_norm_all_zero(self):
        zero_df = pd.DataFrame(0, index=["a", "b"], columns=["x", "y"])
        norm_df = quantile_max_norm(zero_df)
        self.assertTrue((norm_df == 0.0).all().all())

    def test_exp_max_norm_values_and_shape(self):
        norm_df = exp_max_norm(self.df)
        self.assertEqual(norm_df.shape, self.df.shape)
        self.assertTrue((norm_df <= 1.0).all().all())
        self.assertTrue((norm_df >= 0.0).all().all())
        # max of each row should be 1.0
        np.testing.assert_allclose(norm_df.max(axis=1), 1.0, atol=1e-6)

    def test_exp_min_max_scaler_fit_transform_inverse(self):
        X = np.array([[0, 1, 2], [3, 4, 5]], dtype=float)
        scaler = ExpMinMaxScaler()
        scaler.fit(X)
        transformed = scaler.transform(X)
        self.assertTrue(np.all(transformed >= 0.0) and np.all(transformed <= 1.0))

        inverse = scaler.inverse_transform(transformed)
        # Should be close to original X
        np.testing.assert_allclose(inverse, X, atol=1e-6)

    def test_exp_min_max_scaler_fit_on_zeros(self):
        X = np.zeros((3, 3))
        scaler = ExpMinMaxScaler()
        scaler.fit(X)
        transformed = scaler.transform(X)
        self.assertTrue(np.allclose(transformed, 0.0))


class _SimpleDataset(Dataset):
    """A minimal Dataset wrapping a list of row tuples for testing."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return tuple(torch.tensor(f, dtype=torch.float32) for f in self.rows[idx])


class TestGetDatasetNonzeroMask(unittest.TestCase):

    def test_all_nonzero_returns_all_true_mask(self):
        # No zeros → zero fraction = 0.0 for every element → all True
        data = [([1.0, 2.0, 3.0],), ([4.0, 5.0, 6.0],)]
        masks = get_dataset_nonzero_mask(_SimpleDataset(data), threshold=0.5)
        self.assertEqual(len(masks), 1)
        self.assertTrue(masks[0].all().item())

    def test_all_zero_returns_all_false_mask(self):
        # All zeros → zero fraction = 1.0 for every element → all False
        data = [([0.0, 0.0, 0.0],), ([0.0, 0.0, 0.0],)]
        masks = get_dataset_nonzero_mask(_SimpleDataset(data), threshold=0.5)
        self.assertEqual(len(masks), 1)
        self.assertFalse(masks[0].any().item())

    def test_mixed_data_with_threshold(self):
        # Column fractions: f0=3/3=1.0, f1=0/3=0.0, f2=1/3≈0.333
        # With threshold=0.5: f0→False, f1→True, f2→True
        data = [
            ([0.0, 1.0, 2.0],),
            ([0.0, 2.0, 0.0],),
            ([0.0, 3.0, 4.0],),
        ]
        masks = get_dataset_nonzero_mask(_SimpleDataset(data), threshold=0.5)
        self.assertEqual(len(masks), 1)
        expected = torch.tensor([False, True, True])
        self.assertTrue(torch.equal(masks[0], expected))

    def test_multi_feature_dataset(self):
        # Two feature tensors per row
        # feature0 zero fractions: [0/2=0, 2/2=1] → [True, False]
        # feature1 zero fractions: [2/2=1, 0/2=0] → [False, True]
        data = [
            ([1.0, 0.0], [0.0, 2.0]),
            ([2.0, 0.0], [0.0, 3.0]),
        ]
        masks = get_dataset_nonzero_mask(_SimpleDataset(data), threshold=0.5)
        self.assertEqual(len(masks), 2)
        self.assertTrue(torch.equal(masks[0], torch.tensor([True, False])))
        self.assertTrue(torch.equal(masks[1], torch.tensor([False, True])))

    def test_threshold_boundary_is_exclusive(self):
        # 1 out of 2 rows is zero → fraction = 0.5 exactly
        # 0.5 < 0.5 is False, so that column is masked out
        data = [([0.0, 1.0],), ([1.0, 1.0],)]
        masks = get_dataset_nonzero_mask(_SimpleDataset(data), threshold=0.5)
        self.assertEqual(len(masks), 1)
        expected = torch.tensor([False, True])
        self.assertTrue(torch.equal(masks[0], expected))

    def test_from_csv_file(self):
        # Load a small CSV file and verify mask matches expected values.
        # CSV columns: f1 (all zeros), f2 (no zeros), f3 (50% zeros), f4 (25% zeros)
        # With threshold=0.5: f1→False, f2→True, f3→False (=0.5 not < 0.5), f4→True
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_nonzero_mask.csv")
        df = pd.read_csv(csv_path)

        rows = [([row[col] for col in df.columns],) for _, row in df.iterrows()]
        masks = get_dataset_nonzero_mask(_SimpleDataset(rows), threshold=0.5)

        self.assertEqual(len(masks), 1)
        self.assertEqual(masks[0].shape[0], len(df.columns))
        expected = torch.tensor([False, True, False, True])
        self.assertTrue(torch.equal(masks[0], expected))


if __name__ == '__main__':
    unittest.main()