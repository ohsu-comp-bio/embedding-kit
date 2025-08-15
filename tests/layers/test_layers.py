import unittest
import torch
from torch import nn

from embkit.layers import MaskedLinear, LayerInfo, convert_activation  # adjust import if needed


class TestLayerPrimitives(unittest.TestCase):
    def test_masked_linear_zero_mask_no_bias(self):
        ml = MaskedLinear(4, 3, bias=False, mask=torch.zeros(3, 4))
        x = torch.randn(2, 4)
        y = ml(x)
        self.assertTrue(torch.allclose(y, torch.zeros(2, 3)))

    def test_masked_linear_set_mask_shape(self):
        ml = MaskedLinear(5, 2)
        with self.assertRaises(AssertionError):
            ml.set_mask(torch.ones(3, 3))  # wrong shape

    def test_convert_activation_including_none(self):
        self.assertIsInstance(convert_activation("relu"), nn.ReLU)
        self.assertIsNone(convert_activation(None))


class TestLayerInfoDataclass(unittest.TestCase):
    def test_layerinfo_defaults(self):
        li = LayerInfo(units=64)
        self.assertEqual(li.units, 64)
        self.assertEqual(li.op, "linear")
        self.assertEqual(li.activation, "relu")
        self.assertFalse(li.batch_norm)
        self.assertTrue(li.bias)

    def test_layerinfo_none_activation(self):
        li = LayerInfo(units=32, activation=None)
        self.assertIsNone(li.activation)


if __name__ == "__main__":
    unittest.main()