"""Tests for embkit.factory.mapping utilities."""

import unittest
from torch import nn

from embkit.factory import mapping


class TestMappingUtilities(unittest.TestCase):
    def test_clean_params_removes_class_key(self):
        src = {"a": 1, "__class__": "Dummy"}
        cleaned = mapping.clean_params(src)
        self.assertNotIn("__class__", cleaned)
        self.assertIn("a", cleaned)
        self.assertEqual(cleaned["a"], 1)

    def test_linear_roundtrip(self):
        params = {"in_features": 2, "out_features": 3, "bias": False}
        lin = mapping.Linear(**params)
        d = lin.to_dict()
        self.assertEqual(d["__class__"], mapping.get_class_name(mapping.Linear))
        rebuilt = mapping.Linear.from_dict(d)
        self.assertEqual(rebuilt.in_features, 2)
        self.assertEqual(rebuilt.out_features, 3)
        self.assertFalse(rebuilt.bias)

    def test_convert_activation_known(self):
        self.assertIsInstance(mapping.get_activation("relu")(), nn.ReLU)
        self.assertIsInstance(mapping.get_activation("tanh")(), nn.Tanh)
        self.assertIsNone(mapping.get_activation("unknown"))


if __name__ == "__main__":
    unittest.main()
