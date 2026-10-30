import unittest
import torch
from torch import nn

from embkit.models.vae.decoder import Decoder
from embkit.factory.layers import Layer, LayerList
from embkit.modules import MaskedLinear


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
        layers = LayerList([
            Layer(units=8, op="linear", activation="relu", batch_norm=True),
            Layer(units=10, op="linear", activation="tanh", batch_norm=False),
        ])
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
        layers = LayerList([
            Layer(units=5, op="linear", activation=None),
            Layer(units=5, op="linear", activation=None, batch_norm=True),
        ])
        dec = Decoder(latent_dim=3, feature_dim=4, layers=layers)

        # no activations inserted
        self.assertFalse(any(isinstance(m, (nn.ReLU, nn.Tanh, nn.Sigmoid)) for m in dec.net))
        # one BN from second layer
        self.assertEqual(sum(isinstance(m, nn.BatchNorm1d) for m in dec.net), 1)

        x = torch.randn(2, 3)
        y = dec(x)
        self.assertEqual(tuple(y.shape), (2, 4))

    def test_masked_linear_present_and_runs(self):
        layers = LayerList([
            Layer(units=6, op="masked_linear", activation="relu"),
            Layer(units=5, op="linear", activation=None),
        ])
        dec = Decoder(latent_dim=4, feature_dim=3, layers=layers)

        self.assertIsInstance(dec.net[0], MaskedLinear)
        self.assertIsInstance(dec.net[1], nn.ReLU)
        self.assertIsInstance(dec.net[2], nn.Linear)

        y = dec(torch.randn(2, 4))
        self.assertEqual(tuple(y.shape), (2, 3))

    def test_invalid_layer_op_raises_value_error(self):
        layers = LayerList([
            Layer(units=5, op="foo_bar", activation="relu")  # Invalid op
        ])
        with self.assertRaises(ValueError) as context:
            Decoder(latent_dim=3, feature_dim=4, layers=layers)

        self.assertIn("Unknown Layer.op 'foo_bar'", str(context.exception))

    def test_cli_style_int_layers_relu_on_every_hidden(self):
        # Mirrors commands/model.py: LayerList(sizes, output_activation=final_activation)
        dec = Decoder(latent_dim=3, feature_dim=7, layers=LayerList([4, 6], output_activation="none"))
        net = list(dec.net)
        expected = [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
        self.assertEqual(len(net), len(expected))
        for m, t in zip(net, expected):
            self.assertIsInstance(m, t)
        self.assertIs(dec.out, net[-1])

    def test_output_activation_follows_out(self):
        dec = Decoder(latent_dim=3, feature_dim=7, layers=LayerList([4], output_activation="sigmoid"))
        net = list(dec.net)
        self.assertIsInstance(net[-1], nn.Sigmoid)
        self.assertIs(dec.out, net[-2])
        self.assertEqual(dec.out.out_features, 7)

    def test_round_trip_keeps_output_activation(self):
        layers = LayerList([Layer(units=4)], output_activation="sigmoid")
        dec = Decoder(latent_dim=3, feature_dim=7, layers=layers)
        dec2 = Decoder.from_dict(dec.to_dict())
        self.assertEqual(len(dec2.net), len(dec.net))
        self.assertIsInstance(list(dec2.net)[-1], nn.Sigmoid)

    def test_from_dict_legacy_layer_list(self):
        # Decoder dicts saved before to_dict stored the full LayerList: a plain list of Layer dicts
        legacy = {"latent_dim": 3, "feature_dim": 7, "batch_norm": False,
                  "layers": [Layer(units=4).to_dict()]}
        dec = Decoder.from_dict(legacy)
        net = list(dec.net)
        expected = [nn.Linear, nn.ReLU, nn.Linear]
        self.assertEqual(len(net), len(expected))
        for m, t in zip(net, expected):
            self.assertIsInstance(m, t)


if __name__ == "__main__":
    unittest.main()