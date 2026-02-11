import unittest
import numpy as np
import pandas as pd

from embkit.align import (
    calc_rmsd,
    procrustes,
    matrix_spearman_alignment_linear,
)


class TestAlignmentUtils(unittest.TestCase):

    # -------- calc_rmsd --------
    def test_calc_rmsd_correct(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 4.0])
        rmsd = calc_rmsd(a, b)
        expected = np.sqrt(((0)**2 + (0)**2 + (1)**2) / 3)
        self.assertAlmostEqual(rmsd, expected)

    def test_calc_rmsd_mismatched_length_raises(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            calc_rmsd(a, b)

    # -------- matrix_spearman_alignment_linear --------
    def test_matrix_spearman_alignment_linear_unique_mapping(self):
        # Distinct rank patterns (no constants, ≥3 features) -> unique best mapping
        a = pd.DataFrame(
            {
                "f1": [1, 4, 2],
                "f2": [2, 3, 4],
                "f3": [3, 2, 1],
                "f4": [4, 1, 3],
            },
            index=["a1", "a2", "a3"],
        )
        # Permute rows; tiny noise keeps ranks and breaks ties deterministically
        perm = [2, 0, 1]  # b3 <- a3, b1 <- a1, b2 <- a2
        b_vals = a.values[perm, :].astype(float) + 1e-9
        b = pd.DataFrame(b_vals, columns=a.columns, index=["b3", "b1", "b2"])

        out_a, out_b, out_score = matrix_spearman_alignment_linear(a, b)

        self.assertEqual(len(out_a), 3)
        self.assertEqual(len(out_b), 3)
        self.assertEqual(len(out_score), 3)

        self.assertTrue(set(out_a).issubset(a.index))
        self.assertTrue(set(out_b).issubset(b.index))
        for s in out_score:
            self.assertGreaterEqual(s, 0.9)

        # Linear assignment maximizes total score; with these patterns the
        # intended permutation should be chosen.
        matched = dict(zip(out_a, out_b))
        self.assertEqual(matched["a1"], "b1")
        self.assertEqual(matched["a2"], "b2")
        self.assertEqual(matched["a3"], "b3")

    def test_matrix_spearman_alignment_linear_high_cutoff_filters_all(self):
        # Use any non-constant data; cutoff > 1 guarantees empty result
        a = pd.DataFrame(
            {
                "f1": [1, 2, 3],
                "f2": [3, 2, 1],
                "f3": [2, 3, 1],
            },
            index=["a1", "a2", "a3"],
        )
        b = pd.DataFrame(
            {
                "f1": [3, 1, 2],
                "f2": [1, 3, 2],
                "f3": [2, 1, 3],
            },
            index=["b1", "b2", "b3"],
        )
        out_a, out_b, out_score = matrix_spearman_alignment_linear(a, b, cutoff=1.1)
        self.assertEqual(out_a, [])
        self.assertEqual(out_b, [])
        self.assertEqual(out_score, [])


    # -------- procrustes --------
    def test_procrustes_identity(self):
        X = np.eye(3)
        Y = np.eye(3)
        R = procrustes(X, Y)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_procrustes_pure_rotation(self):
        theta = np.pi / 4
        R_true = np.array(
            [[np.cos(theta), -np.sin(theta)],
             [np.sin(theta),  np.cos(theta)]]
        )
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 2))
        Y = X @ R_true
        R = procrustes(X, Y)
        np.testing.assert_array_almost_equal(R, R_true, decimal=6)

    def test_procrustes_reflection_is_corrected_and_optimal(self):
        theta = np.pi / 6
        R_rot = np.array(
            [[np.cos(theta), -np.sin(theta)],
             [np.sin(theta),  np.cos(theta)]]
        )
        F = np.array([[1.0, 0.0], [0.0, -1.0]])  # reflection

        rng = np.random.default_rng(1)
        X = rng.standard_normal((120, 2))
        Y = X @ (R_rot @ F)

        R = procrustes(X, Y)
        self.assertGreater(np.linalg.det(R), 0.0)
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(2), decimal=6)

        # Optimality: recovered R should fit Y better than using R_rot directly
        err_R = np.linalg.norm(X @ R - Y)
        err_Rrot = np.linalg.norm(X @ R_rot - Y)
        self.assertLess(err_R, err_Rrot)


if __name__ == '__main__':
    unittest.main()