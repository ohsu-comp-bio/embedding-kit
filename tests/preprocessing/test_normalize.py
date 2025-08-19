import unittest
import numpy as np
import pandas as pd
from embkit.preprocessing import (
    quantile_max_norm,
    exp_max_norm,
    ExpMinMaxScaler
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


if __name__ == '__main__':
    unittest.main()