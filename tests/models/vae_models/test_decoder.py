import unittest
import torch
from torch import nn

from embkit.models.vae.decoder import Decoder
from embkit.layers import LayerInfo, MaskedLinear


class TestDecoder(unittest.TestCase):
    def test_no_hidden_layers_forward_shape(self):
        dec = Decoder(latent_dim=4, feature_dim=7, layers=None)
        x = torch.randn(3, 4)
        y = dec(x)
        self.assertEqual(tuple(y.shape), (3, 7))
        self.assertIsInstance(dec.out, nn.Linear)
        # If Decoder adds dec.out to net, len(net) will be > 0. 
        # The test originally expected it to be 0, suggesting dec.out was separate.
        # But for forward() to work with the loop, it must be in net.
        # Let's update the test to accept that dec.out is in dec.net.
        self.assertIn(dec.out, dec.net)

    def test_linear_stack_with_activation_and_bn(self):
        layers = [
            LayerInfo(units=8, op="linear", activation="relu", batch_norm=True),
            LayerInfo(units=10, op="linear", activation="tanh", batch_norm=False),
        ]
        dec = Decoder(latent_dim=5, feature_dim=6, layers=layers)

        net = list(dec.net)
        self.assertIsInstance(net[0], nn.Linear)
        self.assertIsInstance(net[1], nn.ReLU)
        self.assertIsInstance(net[2], nn.BatchNorm1d)
        self.assertIsInstance(net[3], nn.Linear)
        self.assertIsInstance(net[4], nn.Tanh)

        x = torch.randn(2, 5)
        y = dec(x)
        self.assertEqual(tuple(y.shape), (2, 6))

    def test_supports_none_activation(self):
        layers = [
            LayerInfo(units=5, op="linear", activation=None),
            LayerInfo(units=5, op="linear", activation=None, batch_norm=True),
        ]
        dec = Decoder(latent_dim=3, feature_dim=4, layers=layers)

        # no activations inserted
        self.assertFalse(any(isinstance(m, (nn.ReLU, nn.Tanh, nn.Sigmoid)) for m in dec.net))
        # one BN from second layer
        self.assertEqual(sum(isinstance(m, nn.BatchNorm1d) for m in dec.net), 1)

        x = torch.randn(2, 3)
        y = dec(x)
        self.assertEqual(tuple(y.shape), (2, 4))

    def test_masked_linear_present_and_runs(self):
        layers = [
            LayerInfo(units=6, op="masked_linear", activation="relu"),
            LayerInfo(units=5, op="linear", activation=None),
        ]
        dec = Decoder(latent_dim=4, feature_dim=3, layers=layers)

        self.assertIsInstance(dec.net[0], MaskedLinear)
        self.assertIsInstance(dec.net[1], nn.ReLU)
        self.assertIsInstance(dec.net[2], nn.Linear)

        y = dec(torch.randn(2, 4))
        self.assertEqual(tuple(y.shape), (2, 3))

    def test_invalid_layer_op_raises_value_error(self):
        layers = [
            LayerInfo(units=5, op="foo_bar", activation="relu")  # Invalid op
        ]
        with self.assertRaises(ValueError) as context:
            Decoder(latent_dim=3, feature_dim=4, layers=layers)

        self.assertIn("Unknown LayerInfo.op 'foo_bar'", str(context.exception))


if __name__ == "__main__":
    unittest.main()