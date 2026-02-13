import builtins
import unittest
from unittest.mock import patch

from embkit.factory.core import build
from embkit.factory.mapping import convert_activation, Linear, Sequential
from embkit.layers import MaskedLinear

class TestBuildFunction(unittest.TestCase):
   
    def test_build_from_list(self):
        # list of dicts should return nn.Sequential of layers
        desc = [
            Linear(10, 20),
            Linear(20, 1),
        ]
        seq = build(desc)
        # list should be returned directly
        self.assertIsInstance(seq, Sequential)
        self.assertEqual(len(seq), 2)
        self.assertTrue(all(isinstance(m, Linear) for m in seq))

    def test_build_from_activation_string(self):
        # ensure that a known activation string returns the class
        # convert_activation returns a nn.Module instance for known names
        result = build("relu")
        # The built-in convert_activation maps "relu" to nn.ReLU()
        from torch import nn

        self.assertIsInstance(result, nn.ReLU)

    def test_build_invalid_input(self):
        with self.assertRaises(Exception) as ctx:
            build(123)
        self.assertIn("Invalid input for build function", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
