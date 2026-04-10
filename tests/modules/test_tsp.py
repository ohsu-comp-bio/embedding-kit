import unittest

import torch

from embkit.modules import TSPLayer


class TestTSPLayer(unittest.TestCase):
    def test_soft_votes_shape(self):
        layer = TSPLayer(pairs=[(0, 1), (2, 3)], beta=10.0, hard=False)
        x = torch.tensor([[2.0, 1.0, 0.0, 3.0]])
        out = layer(x)
        self.assertEqual(tuple(out.shape), (1, 2))
        self.assertTrue(torch.all((out > 0) & (out < 1)))

    def test_hard_votes_and_chunking(self):
        layer = TSPLayer(pairs=[(0, 1), (2, 3), (1, 0)], hard=True, chunk_size=2)
        x = torch.tensor([[2.0, 1.0, 0.0, 3.0]])
        out = layer(x)
        self.assertEqual(tuple(out.shape), (1, 3))
        self.assertTrue(torch.equal(out, torch.tensor([[1.0, 0.0, 0.0]])))

    def test_learnable_weight_aggregation(self):
        layer = TSPLayer(pairs=[(0, 1), (1, 0)], hard=True, learnable_weights=True)
        with torch.no_grad():
            layer.weights.copy_(torch.tensor([2.0, 3.0]))
        x = torch.tensor([[2.0, 1.0]])
        out = layer(x)
        self.assertEqual(tuple(out.shape), (1,))
        self.assertAlmostEqual(float(out.item()), 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
