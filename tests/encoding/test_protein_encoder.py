import unittest
from unittest.mock import patch

import torch

from embkit.encoding.protein import ProteinEncoder


class DummyAlphabet:
    def __init__(self):
        self.padding_idx = 0

    def get_batch_converter(self):
        def _convert(block):
            labels = [x[0] for x in block]
            seqs = [x[1] for x in block]
            max_len = max(len(s) for s in seqs) if seqs else 0
            # +2 to simulate BOS/EOS style tokenized length.
            tokens = torch.ones((len(block), max_len + 2), dtype=torch.long)
            return labels, seqs, tokens

        return _convert


class DummyModel:
    def __init__(self):
        self.embed_dim = 16
        self.sent_to = None

    def eval(self):
        return self

    def to(self, device):
        self.sent_to = device
        return self

    def __call__(self, batch_tokens, repr_layers, return_contacts):
        layer = repr_layers[0]
        bsz, tlen = batch_tokens.shape
        reps = torch.arange(bsz * tlen * 4, dtype=torch.float32).reshape(bsz, tlen, 4)
        return {"representations": {layer: reps}}


class TestProteinEncoder(unittest.TestCase):
    def _make_pretrained_fn(self):
        def _factory():
            return DummyModel(), DummyAlphabet()

        return _factory

    @patch("embkit.encoding.protein.esm.pretrained.esm2_t6_8M_UR50D")
    def test_init_to_get_embed_dim_and_unknown_model(self, t6_mock):
        t6_mock.side_effect = self._make_pretrained_fn()

        enc = ProteinEncoder(model="t6", batch_size=2, device=torch.device("cpu"))
        self.assertEqual(enc.out_layer, 6)
        self.assertEqual(enc.get_embed_dim(), 16)
        self.assertEqual(enc.device.type, "cpu")

        enc.to(torch.device("cpu"))
        self.assertEqual(enc.model.sent_to.type, "cpu")

        with self.assertRaises(Exception):
            ProteinEncoder(model="bogus")

    @patch("embkit.encoding.protein.tqdm", side_effect=lambda x: x)
    @patch("embkit.encoding.protein.esm.pretrained.esm2_t33_650M_UR50D")
    def test_encode_modes_and_fix_len(self, t33_mock, _tqdm_mock):
        t33_mock.side_effect = self._make_pretrained_fn()
        enc = ProteinEncoder(model="t33", batch_size=2, device=None)

        data = [("p1", "AAAA"), ("p2", "AA")]

        out_sum = list(enc.encode(data, output="sum-pool", fix_len=None, verbose=False))
        self.assertEqual(len(out_sum), 2)
        self.assertEqual(out_sum[0][0], "p1")
        self.assertEqual(tuple(out_sum[0][1].shape), (4,))

        out_mean = list(enc.encode(data, output="mean-pool", fix_len=None, verbose=True))
        self.assertEqual(len(out_mean), 2)
        self.assertEqual(tuple(out_mean[1][1].shape), (4,))

        out_vec = list(enc.encode(data, output="vector", fix_len=3, verbose=False))
        self.assertEqual(tuple(out_vec[0][1].shape), (3, 4))

    @patch("embkit.encoding.protein.esm.pretrained.esm2_t12_35M_UR50D")
    def test_pad(self, t12_mock):
        t12_mock.side_effect = self._make_pretrained_fn()
        enc = ProteinEncoder(model="t12", batch_size=1)

        tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
        padded = enc.pad(tokens, 6)
        self.assertEqual(tuple(padded.shape), (1, 6))
        self.assertTrue(torch.equal(padded[0, :3], torch.tensor([1, 2, 3])))


if __name__ == "__main__":
    unittest.main()
