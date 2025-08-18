import unittest
import torch
from embkit.layers import PairwiseComparison  # adjust if necessary


class TestPairwiseComparison(unittest.TestCase):
    def setUp(self):
        # Input with 4 features => 6 pairwise comparisons: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        self.x = torch.tensor([[1.0, 2.0, 4.0, 8.0]], dtype=torch.float)  # shape: (1, 4)
        self.expected_num_pairs = 6

    def test_difference(self):
        layer = PairwiseComparison("difference")
        out = layer(self.x)
        expected = torch.tensor([[1-2, 1-4, 1-8, 2-4, 2-8, 4-8]], dtype=torch.float)
        self.assertEqual(out.shape, (1, self.expected_num_pairs))
        self.assertTrue(torch.allclose(out, expected))

    def test_ratio(self):
        layer = PairwiseComparison("ratio")
        out = layer(self.x)
        expected = torch.tensor([[1/2, 1/4, 1/8, 2/4, 2/8, 4/8]], dtype=torch.float)
        self.assertEqual(out.shape, (1, self.expected_num_pairs))
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_greater_than(self):
        layer = PairwiseComparison("greater_than")
        out = layer(self.x)
        expected = torch.tensor([[0., 0., 0., 0., 0., 0.]], dtype=torch.float)
        self.assertEqual(out.shape, (1, self.expected_num_pairs))
        self.assertTrue(torch.equal(out, expected))

    def test_less_than(self):
        layer = PairwiseComparison("less_than")
        out = layer(self.x)
        expected = torch.tensor([[1., 1., 1., 1., 1., 1.]], dtype=torch.float)
        self.assertEqual(out.shape, (1, self.expected_num_pairs))
        self.assertTrue(torch.equal(out, expected))

    def test_invalid_comparison_type(self):
        with self.assertRaises(ValueError):
            PairwiseComparison("not_supported")

    def test_invalid_input_shape(self):
        layer = PairwiseComparison("difference")
        invalid_input = torch.randn(2, 3, 4)  # Too many dimensions
        with self.assertRaises(ValueError):
            _ = layer(invalid_input)

    def test_compute_output_shape(self):
        layer = PairwiseComparison("difference")
        shape = layer.compute_output_shape((8, 5))
        self.assertEqual(shape, (8, 10))  # 5 choose 2 = 10

    def test_greater_than_mixed(self):
        x = torch.tensor([[5.0, 2.0, 3.0, 1.0]], dtype=torch.float)
        layer = PairwiseComparison("greater_than")
        out = layer(x)
        expected = torch.tensor([[1., 1., 1., 0., 1., 1.]], dtype=torch.float)
        self.assertEqual(out.shape, (1, 6))
        self.assertTrue(torch.equal(out, expected))

    def test_less_than_mixed(self):
        x = torch.tensor([[5.0, 2.0, 3.0, 6.0]], dtype=torch.float)
        layer = PairwiseComparison("less_than")
        out = layer(x)
        expected = torch.tensor([[0., 0., 1., 1., 1., 1.]], dtype=torch.float)
        self.assertEqual(out.shape, (1, 6))
        self.assertTrue(torch.equal(out, expected))