"""Unit tests for ``embkit.modules.mha`` (``MHABlock`` and ``MHAPooling``)."""

import unittest

import torch
from torch import nn

from embkit.factory.core import build
from embkit.modules.mha import MHABlock, MHAPooling


class TestMHABlock(unittest.TestCase):
    def test_forward_shape_same_dims(self):
        block = MHABlock(embed_dim=8, num_heads=2, out_dim=8)
        x = torch.randn(4, 6, 8)
        out = block(x)
        self.assertEqual(out.shape, (4, 6, 8))

    def test_forward_residual_proj_when_out_dim_differs(self):
        # out_dim != embed_dim exercises the nn.Linear residual_proj branch.
        block = MHABlock(embed_dim=8, num_heads=2, out_dim=16)
        x = torch.randn(3, 5, 8)
        out = block(x)
        self.assertEqual(out.shape, (3, 5, 16))
        self.assertIsInstance(block.residual_proj, nn.Linear)

    def test_default_ffn_inner_dim(self):
        block = MHABlock(embed_dim=8, num_heads=2, out_dim=8)
        self.assertEqual(block.ffn_inner_dim, 8 * 4)

    def test_custom_ffn_inner_dim(self):
        block = MHABlock(embed_dim=8, num_heads=2, out_dim=8, ffn_inner_dim=32)
        self.assertEqual(block.ffn_inner_dim, 32)

    def test_to_dict_keys(self):
        block = MHABlock(embed_dim=8, num_heads=2, out_dim=8, dropout=0.05)
        d = block.to_dict()
        self.assertEqual(d["embed_dim"], 8)
        self.assertEqual(d["num_heads"], 2)
        self.assertEqual(d["out_dim"], 8)
        self.assertEqual(d["dropout"], 0.05)
        self.assertEqual(d["ffn_inner_dim"], 32)
        self.assertIn("__class__", d)

    def test_roundtrip_from_dict(self):
        block = MHABlock(embed_dim=8, num_heads=2, out_dim=8, dropout=0.05)
        rebuilt = MHABlock.from_dict(block.to_dict())
        self.assertIsInstance(rebuilt, MHABlock)
        self.assertEqual(rebuilt.embed_dim, block.embed_dim)
        self.assertEqual(rebuilt.num_heads, block.num_heads)
        self.assertEqual(rebuilt.out_dim, block.out_dim)
        self.assertEqual(rebuilt.ffn_inner_dim, block.ffn_inner_dim)

    def test_factory_build_roundtrip(self):
        block = MHABlock(embed_dim=8, num_heads=2, out_dim=8)
        rebuilt = build(block)
        self.assertIsInstance(rebuilt, MHABlock)
        self.assertEqual(rebuilt.embed_dim, block.embed_dim)


class TestMHAPooling(unittest.TestCase):
    def test_forward_output_shape(self):
        pool = MHAPooling(embed_dim=8, num_heads=2)
        x = torch.randn(4, 6, 8)
        out = pool(x)
        self.assertEqual(out.shape, (4, 8))

    def test_forward_with_padding_mask(self):
        pool = MHAPooling(embed_dim=8, num_heads=2)
        x = torch.randn(4, 6, 8)
        mask = torch.zeros(4, 6, dtype=torch.bool)
        mask[:, -2:] = True  # mask the last two positions
        out = pool(x, key_padding_mask=mask)
        self.assertEqual(out.shape, (4, 8))
        self.assertFalse(torch.isnan(out).any())

    def test_to_dict_keys(self):
        pool = MHAPooling(embed_dim=8, num_heads=2)
        d = pool.to_dict()
        self.assertEqual(d["embed_dim"], 8)
        self.assertEqual(d["num_heads"], 2)
        self.assertIn("__class__", d)

    def test_roundtrip_from_dict(self):
        pool = MHAPooling(embed_dim=8, num_heads=2)
        rebuilt = MHAPooling.from_dict(pool.to_dict())
        self.assertIsInstance(rebuilt, MHAPooling)
        self.assertEqual(rebuilt.embed_dim, pool.embed_dim)
        self.assertEqual(rebuilt.num_heads, pool.num_heads)

    def test_factory_build_roundtrip(self):
        pool = MHAPooling(embed_dim=8, num_heads=2)
        rebuilt = build(pool)
        self.assertIsInstance(rebuilt, MHAPooling)
        self.assertEqual(rebuilt.embed_dim, pool.embed_dim)


if __name__ == "__main__":
    unittest.main()
