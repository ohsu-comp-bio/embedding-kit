"""Tests for embkit.factory.layers utilities and Layer classes."""

import unittest
import numpy as np
import pandas as pd
from torch import nn
import torch

from embkit.factory import layers


class DummyConstraint:
    def __init__(self, out_features, in_features):
        self.out_features = out_features
        self.in_features = in_features

    def gen_mask(self):
        # Return a mask matching expected shape
        return np.ones((self.out_features, self.in_features), dtype=np.float32)


class TestLayerHelpers(unittest.TestCase):
    def test_idx_to_list_preserves_order(self):
        mapping = {"c": 2, "a": 0, "b": 1}
        ordered = layers.idx_to_list(mapping)
        self.assertEqual(ordered, ["a", "b", "c"])

    def test_build_features_to_group_mask_forward(self):
        feature_map = {"G1": ["f0"], "G2": ["f1"]}
        feature_idx = {"f0": 0, "f1": 1}
        group_idx = {"G1": 0, "G2": 1}
        mask = layers.build_features_to_group_mask(
            feature_map, feature_idx, group_idx, group_node_count=1, forward=True
        )
        expected = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        np.testing.assert_array_equal(mask, expected)

    def test_layer_from_dict_and_gen_layer_linear(self):
        spec = {
            "__class__": "Layer",
            "units": 3,
            "bias": True,
        }
        # classMap should have Linear registered
        layer_obj = layers.Layer.from_dict(spec)
        self.assertIsInstance(layer_obj, layers.Layer)
        # gen_layer should produce a torch.nn.Linear
        mod = layer_obj.gen_layer(in_features=2)
        self.assertIsInstance(mod, nn.Linear)
        self.assertEqual(mod.in_features, 2)
        self.assertEqual(mod.out_features, 3)

    def test_layer_gen_layer_masked_linear_with_constraint(self):
        # Create a Layer with masked_linear op and dummy constraint
        dummy = DummyConstraint(out_features=2, in_features=3)
        layer_obj = layers.Layer(
            units=2, op="masked_linear", constraint=dummy, bias=False
        )
        mod = layer_obj.gen_layer(in_features=3)
        self.assertIsInstance(mod, layers.MaskedLinear)
        # Verify mask applied matches dummy mask (all ones)
        self.assertTrue(torch.equal(mod.mask, torch.ones((2, 3), dtype=mod.mask.dtype)))

    def test_layer_gen_layer_unknown_op_raises(self):
        layer_obj = layers.Layer(units=2, op="unknown")
        with self.assertRaises(ValueError):
            layer_obj.gen_layer(in_features=3)

    def test_layerlist_build_empty_returns_linear(self):
        ll = layers.LayerList(layers=[])
        mod = ll.build(input_dim=4, output_dim=2)
        self.assertIsInstance(mod, nn.Linear)
        self.assertEqual(mod.in_features, 4)
        self.assertEqual(mod.out_features, 2)

    def test_layerlist_build_with_layers(self):
        l1 = layers.Layer(units=3, op="linear")
        l2 = layers.Layer(units=2, op="linear")
        ll = layers.LayerList(layers=[l1, l2])
        net = ll.build(input_dim=5, output_dim=1)
        # Should be a Sequential with 3 linear layers (2 from list + final output)
        self.assertIsInstance(net, nn.Sequential)
        self.assertEqual(len(net), 3)
        self.assertEqual(net[0].out_features, 3)
        self.assertEqual(net[1].out_features, 2)
        self.assertEqual(net[2].out_features, 1)


if __name__ == "__main__":
    unittest.main()
