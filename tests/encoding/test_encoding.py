import unittest
import numpy as np
import torch

from embkit.encoding import OneHotEncoder, ProteinOneHotEncoder


class TestOneHotEncoder(unittest.TestCase):
    def test_single_and_batch_labels(self):
        labels = ['b', 'a', 'c']
        enc = OneHotEncoder(labels)

        # single
        out_a = enc('a')
        expected_a = torch.tensor([1, 0, 0], dtype=out_a.dtype)
        self.assertTrue(torch.equal(out_a, expected_a))

        # batch list of keys
        batch = enc(['a', 'c', 'b'])
        self.assertEqual(batch.shape, (3, 3))
        self.assertTrue(torch.equal(batch[0], expected_a))

        # numeric indices
        batch_idx = enc([0, 2])
        self.assertEqual(batch_idx.shape, (2, 3))

        # tensor input
        t = torch.tensor([1, 2])
        out_t = enc(t)
        self.assertEqual(out_t.shape, (2, 3))


class TestProteinOneHotEncoder(unittest.TestCase):
    def test_single_sequence(self):
        enc = ProteinOneHotEncoder(full_len=None, encode_x=True, encode_pos=False)
        seq = 'ACD'
        out = enc(seq)
        # shape (L, D)
        self.assertEqual(out.shape, (3, len(enc.alphabet)))
        # each row is one-hot
        sums = out.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums)))

    def test_batch_and_padding(self):
        enc = ProteinOneHotEncoder(full_len=None, encode_x=True, encode_pos=False)
        seqs = ['A', 'ACDE']
        out = enc(seqs)
        # batch, FL, D
        self.assertEqual(out.shape[0], 2)
        FL = out.shape[1]
        self.assertEqual(FL, 4)
        D = out.shape[2]
        self.assertEqual(D, len(enc.alphabet))

        # padding for first sequence should be 'X'
        idx_X = enc.aa_to_index['X']
        # positions 1..3 for first sequence should have 1 at idx_X
        pad = out[0, 1:, idx_X]
        self.assertTrue(torch.allclose(pad, torch.ones_like(pad)))

    def test_encode_pos_column(self):
        enc = ProteinOneHotEncoder(full_len=3, encode_x=True, encode_pos=True)
        seq = 'AC'
        out = enc(seq)
        # shape (L, D+1)
        self.assertEqual(out.shape, (3, len(enc.alphabet) + 1))
        # last column holds position (normalized by full_len)
        pos_col = out[:, -1]
        print(pos_col)
        expected = torch.tensor([0.0, 1.0 / 3.0, 0.0], dtype=pos_col.dtype)
        self.assertTrue(torch.allclose(pos_col, expected))

    def test_encode_pos_column_numpy(self):
        enc = ProteinOneHotEncoder(full_len=3, encode_x=True, encode_pos=True, backend='numpy')
        seq = 'AC'
        out = enc(seq)
        # shape (L, D+1)
        self.assertEqual(out.shape, (3, len(enc.alphabet) + 1))
        # last column holds position (normalized by full_len)
        pos_col = out[:, -1]
        print(pos_col)
        expected = np.array([0.0, 1.0 / 3.0, 0.0], dtype=pos_col.dtype)
        self.assertTrue(np.allclose(pos_col, expected))


if __name__ == '__main__':
    unittest.main()
