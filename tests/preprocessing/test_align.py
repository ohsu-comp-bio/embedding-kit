import unittest
import numpy as np
import pandas as pd
from embkit.preprocessing import calc_rmsd, matrix_spearman_alignment_set, procrustes


class TestAlignmentUtils(unittest.TestCase):

    def test_calc_rmsd_correct(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 4.0])
        rmsd = calc_rmsd(a, b)
        expected = np.sqrt(((0)**2 + (0)**2 + (1)**2)/3)
        self.assertAlmostEqual(rmsd, expected)

    def test_calc_rmsd_mismatched_length_raises(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            calc_rmsd(a, b)

    def test_matrix_spearman_alignment_set_basic(self):
        # Construct two aligned datasets
        a = pd.DataFrame({
            "feat1": [1, 2, 3],
            "feat2": [4, 5, 6]
        }, index=["a1", "a2", "a3"])

        b = pd.DataFrame({
            "feat1": [1.1, 2.1, 3.1],
            "feat2": [4.1, 5.1, 6.1]
        }, index=["b1", "b2", "b3"])

        out = matrix_spearman_alignment_set(a, b)
        self.assertIsInstance(out, dict)
        self.assertEqual(len(out), 3)
        for key, (matched, score) in out.items():
            self.assertIn(matched, b.index)
            self.assertGreaterEqual(score, 0.9)

    def test_matrix_spearman_alignment_with_cutoff(self):
        a = pd.DataFrame({
            "feat1": [1, 2],
            "feat2": [3, 4]
        }, index=["a1", "a2"])
        b = pd.DataFrame({
            "feat1": [7, 8],
            "feat2": [5, 6]
        }, index=["b1", "b2"])

        result = matrix_spearman_alignment_set(a, b, cuttoff=1.0)
        self.assertEqual(result, {})  # Nothing passes the cutoff

    def test_procrustes_identity(self):
        X = np.eye(3)
        Y = np.eye(3)
        R = procrustes(X, Y)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_procrustes_rotation(self):
        theta = np.pi / 4
        R_true = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta),  np.cos(theta)]])
        X = np.random.randn(5, 2)
        Y = X @ R_true.T
        R = procrustes(X, Y)
        np.testing.assert_array_almost_equal(R, R_true.T, decimal=5)

    def test_procrustes_assertion(self):
        X = np.random.rand(5, 2)
        Y = np.random.rand(6, 2)
        with self.assertRaises(AssertionError):
            procrustes(X, Y)


if __name__ == '__main__':
    unittest.main()