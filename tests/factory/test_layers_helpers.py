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

    def gen_mask(self, in_features, out_features):
        # Return a mask matching expected shape
        return np.ones((out_features, in_features), dtype=np.float32)


class TestLayerHelpers(unittest.TestCase):

    def test_layer_from_dict_and_gen_layer_linear(self):
        spec = {
            "__class__": "Layer",
            "units": 3,
            "bias": True,
        }
        # classMap should have Linear registered
        layer_obj = layers.Layer.from_dict(spec)
        self.assertIsInstance(layer_obj, layers.Layer)
        # gen_layer should produce a [torch.nn.Linear, torch.nn.Relu]
        mod = layer_obj.gen_layer(in_features=2)
        self.assertIsInstance(mod[0], nn.Linear)
        self.assertEqual(mod[0].in_features, 2)
        self.assertEqual(mod[0].out_features, 3)

    def test_layer_gen_layer_masked_linear_with_constraint(self):
        # Create a Layer with masked_linear op and dummy constraint
        dummy = DummyConstraint(out_features=2, in_features=3)
        layer_obj = layers.Layer(
            units=2, op="masked_linear", constraint=dummy, bias=False
        )
        mod = layer_obj.gen_layer(in_features=3)
        self.assertIsInstance(mod[0], layers.MaskedLinear)
        # Verify mask applied matches dummy mask (all ones)
        self.assertTrue(torch.equal(mod[0].mask, torch.ones((2, 3), dtype=mod[0].mask.dtype)))

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
        print(net)
        # Should be a Sequential with 3 linear layers (2 from list + final output)
        self.assertIsInstance(net, nn.Sequential)
        self.assertEqual(len(net), 5)
        self.assertEqual(net[0].out_features, 3)
        self.assertEqual(net[2].out_features, 2)
        self.assertEqual(net[4].out_features, 1)


if __name__ == "__main__":
    unittest.main()
