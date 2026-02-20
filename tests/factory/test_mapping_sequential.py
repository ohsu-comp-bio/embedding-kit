"""Tests for embkit.factory.mapping utilities and Sequential round-trip."""

import unittest
import numpy as np

from embkit.factory import mapping
from torch import nn


class TestMappingSequential(unittest.TestCase):
    def test_sequential_roundtrip(self):
        # Build a Sequential with two Linear layers
        l1 = mapping.Linear(2, 3)
        l2 = mapping.Linear(3, 1)
        seq = mapping.Sequential(l1, l2)
        d = seq.to_dict()
        # Ensure dict contains class name and args
        self.assertEqual(d["__class__"], mapping.get_class_name(mapping.Sequential))
        self.assertEqual(len(d["args"]), 2)
        # Reconstruct from dict
        rebuilt = mapping.Sequential.from_dict(d)
        self.assertIsInstance(rebuilt, mapping.Sequential)
        # The rebuilt sequence should contain Linear modules with same dimensions
        self.assertIsInstance(rebuilt[0], mapping.Linear)
        self.assertEqual(rebuilt[0].in_features, 2)
        self.assertEqual(rebuilt[0].out_features, 3)
        self.assertIsInstance(rebuilt[1], mapping.Linear)
        self.assertEqual(rebuilt[1].in_features, 3)
        self.assertEqual(rebuilt[1].out_features, 1)

    def test_clean_params_removes_class(self):
        params = {"__class__": mapping.Linear.__name__, "in_features": 4, "out_features": 2}
        cleaned = mapping.clean_params(params)
        self.assertNotIn("__class__", cleaned)
        self.assertEqual(cleaned["in_features"], 4)
        self.assertEqual(cleaned["out_features"], 2)

    def test_convert_activation_unknown(self):
        self.assertIsNone(mapping.get_activation("not_a_real_activation"))


if __name__ == "__main__":
    unittest.main()
