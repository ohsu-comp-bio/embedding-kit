"""Unit tests for ``embkit.models.pair`` (``PairPredictor``)."""

import unittest

import torch
from torch import nn

from embkit import factory
from embkit.models.pair import PairPredictor


@factory.nn_module
class TinyEmbedding(nn.Module):
    """A minimal registry-registered stand-in for an item/context module."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.lin(x)

    def to_dict(self):
        return {"in_dim": self.in_dim, "out_dim": self.out_dim}

    @classmethod
    def from_dict(cls, d):
        return cls(d["in_dim"], d["out_dim"])


def _tiny_module(in_dim: int, out_dim: int) -> TinyEmbedding:
    return TinyEmbedding(in_dim, out_dim)


class TestPairPredictor(unittest.TestCase):
    def test_forward_output_shape(self):
        item_module = _tiny_module(4, 8)
        context_module = _tiny_module(3, 8)
        pred = PairPredictor(
            item_module=item_module,
            context_module=context_module,
            item_dim=8,
            context_dim=8,
            learning_dim=16,
        )
        item1 = torch.randn(5, 4)
        item2 = torch.randn(5, 4)
        context = torch.randn(5, 3)
        out = pred((item1, item2, context))
        self.assertEqual(out.shape, (5, 1))

    def test_forward_accepts_list_input(self):
        item_module = _tiny_module(4, 8)
        context_module = _tiny_module(3, 8)
        pred = PairPredictor(
            item_module=item_module,
            context_module=context_module,
            item_dim=8,
            context_dim=8,
            learning_dim=16,
        )
        item1 = torch.randn(3, 4)
        item2 = torch.randn(3, 4)
        context = torch.randn(3, 3)
        out = pred([item1, item2, context])
        self.assertEqual(out.shape, (3, 1))

    def test_to_dict_keys(self):
        item_module = _tiny_module(4, 8)
        context_module = _tiny_module(3, 8)
        pred = PairPredictor(
            item_module=item_module,
            context_module=context_module,
            item_dim=8,
            context_dim=8,
            learning_dim=16,
        )
        d = pred.to_dict()
        self.assertEqual(d["item_dim"], 8)
        self.assertEqual(d["context_dim"], 8)
        self.assertEqual(d["learning_dim"], 16)
        self.assertIn("item_module", d)
        self.assertIn("context_module", d)
        self.assertIn("__class__", d)

    def test_from_dict_reconstructs(self):
        item_module = _tiny_module(4, 8)
        context_module = _tiny_module(3, 8)
        pred = PairPredictor(
            item_module=item_module,
            context_module=context_module,
            item_dim=8,
            context_dim=8,
            learning_dim=16,
        )
        params = pred.to_dict()
        self.assertIn("__class__", params["item_module"])
        rebuilt = PairPredictor.from_dict(params)
        self.assertIsInstance(rebuilt, PairPredictor)
        self.assertIsInstance(rebuilt.item_module, TinyEmbedding)
        self.assertIsInstance(rebuilt.context_module, TinyEmbedding)
        self.assertEqual(rebuilt.item_dim, 8)
        self.assertEqual(rebuilt.context_dim, 8)
        self.assertEqual(rebuilt.learning_dim, 16)

        # the reconstructed model runs the same forward shape
        out = rebuilt((torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 3)))
        self.assertEqual(out.shape, (4, 1))

    def test_output_is_finite(self):
        item_module = _tiny_module(4, 8)
        context_module = _tiny_module(3, 8)
        pred = PairPredictor(
            item_module=item_module,
            context_module=context_module,
            item_dim=8,
            context_dim=8,
            learning_dim=16,
        )
        pred.eval()
        with torch.no_grad():
            out = pred((torch.randn(5, 4), torch.randn(5, 4), torch.randn(5, 3)))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
