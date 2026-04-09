"""Tests for mask helper functions in embkit.factory.layers."""

import unittest
import numpy as np
import pandas as pd

from embkit import pathway


class TestMaskHelpers(unittest.TestCase):
    def test_idx_to_list_orders_correctly(self):
        mapping = {"b": 0, "a": 1, "c": 2}
        ordered = pathway.idx_to_list(mapping)
        self.assertEqual(ordered, ["b", "a", "c"])  # positions 0,1,2

    def test_build_features_to_group_mask_forward(self):
        # feature_map: group -> list of features
        feature_map = {"G1": ["f0"], "G2": ["f1"]}
        feature_idx = {"f0": 0, "f1": 1}
        group_idx = {"G1": 0, "G2": 1}
        mask = pathway.build_features_to_group_mask(
            feature_map, feature_idx, group_idx, group_node_count=1, forward=True
        )
        expected = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        np.testing.assert_array_equal(mask, expected)

    def test_build_features_to_group_mask_reverse(self):
        feature_map = {"G1": ["f0"], "G2": ["f1"]}
        feature_idx = {"f0": 0, "f1": 1}
        group_idx = {"G1": 0, "G2": 1}
        mask = pathway.build_features_to_group_mask(
            feature_map, feature_idx, group_idx, group_node_count=1, forward=False
        )
        # reverse shape (in_features, out_features)
        expected = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32).T
        np.testing.assert_array_equal(mask, expected)


class TestLayerHelpers(unittest.TestCase):
    def test_idx_to_list_preserves_order(self):
        mapping = {"c": 2, "a": 0, "b": 1}
        ordered = pathway.idx_to_list(mapping)
        self.assertEqual(ordered, ["a", "b", "c"])

    def test_build_features_to_group_mask_forward(self):
        feature_map = {"G1": ["f0"], "G2": ["f1"]}
        feature_idx = {"f0": 0, "f1": 1}
        group_idx = {"G1": 0, "G2": 1}
        mask = pathway.build_features_to_group_mask(
            feature_map, feature_idx, group_idx, group_node_count=1, forward=True
        )
        expected = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        np.testing.assert_array_equal(mask, expected)

if __name__ == "__main__":
    unittest.main()
