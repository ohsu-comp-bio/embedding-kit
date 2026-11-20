"""Unit tests for ``embkit.models.hmm``."""

import unittest

import numpy as np
import torch

from embkit.factory.core import build
from embkit.models.hmm import GaussianHMM, segment_by_order


def _three_segments(seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0, 0.2, 30), rng.normal(4, 0.2, 30), rng.normal(-3, 0.2, 30)])


class TestGaussianHMM(unittest.TestCase):
    def test_fit_recovers_segments(self):
        seq = _three_segments()
        hmm = GaussianHMM(3, 1, stickiness=0.95).fit(seq, n_iter=150)
        path = hmm.viterbi(seq).numpy()
        # each true segment is assigned one state, and the three states differ
        seg_states = [np.bincount(path[i:i + 30]).argmax() for i in (0, 30, 60)]
        self.assertEqual(len(set(seg_states)), 3)
        for i, s in zip((0, 30, 60), seg_states):
            self.assertGreaterEqual((path[i:i + 30] == s).mean(), 0.9)

    def test_posterior_rows_sum_to_one(self):
        hmm = GaussianHMM(2, 2)
        post = hmm.posterior(np.random.default_rng(0).normal(size=(10, 2)))
        self.assertEqual(post.shape, (10, 2))
        self.assertTrue(torch.allclose(post.sum(-1), torch.ones(10, dtype=torch.float64)))

    def test_batch_inputs(self):
        hmm = GaussianHMM(2, 1)
        x = torch.randn(3, 7, 1)
        self.assertEqual(hmm.log_likelihood(x).shape, (3,))
        self.assertEqual(hmm.viterbi(x).shape, (3, 7))
        self.assertEqual(hmm.posterior(x).shape, (3, 7, 2))

    def test_log_likelihood_single_state_matches_gaussian(self):
        hmm = GaussianHMM(1, 1)
        x = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
        expected = torch.distributions.Normal(0.0, 1.0).log_prob(x).sum()
        self.assertTrue(torch.allclose(hmm.log_likelihood(x), expected))

    def test_initialise_single_point(self):
        hmm = GaussianHMM(1, 2)
        hmm.initialise(np.ones((1, 2)))
        self.assertTrue(torch.allclose(hmm.means[0], torch.ones(2, dtype=torch.float64), atol=1e-2))
        self.assertTrue(torch.isfinite(hmm.logvars).all())

    def test_invalid(self):
        with self.assertRaises(ValueError):
            GaussianHMM(0, 1)
        with self.assertRaises(ValueError):
            GaussianHMM(2, 1, stickiness=1.0)
        with self.assertRaises(ValueError):
            GaussianHMM(2, 2).log_likelihood(np.zeros((5, 3)))

    def test_roundtrip(self):
        rebuilt = build(GaussianHMM(4, 2, stickiness=0.8))
        self.assertIsInstance(rebuilt, GaussianHMM)
        self.assertEqual((rebuilt.n_states, rebuilt.dim, rebuilt.stickiness), (4, 2, 0.8))


class TestSegmentByOrder(unittest.TestCase):
    def test_returns_original_order(self):
        seq = _three_segments()
        order = np.arange(90)
        perm = np.random.default_rng(1).permutation(90)
        states, probs, model = segment_by_order(torch.as_tensor(seq[perm]), order[perm], n_states=3, n_iter=150)
        self.assertEqual(states.shape, (90,))
        self.assertEqual(probs.shape, (90, 3))
        self.assertIsInstance(model, GaussianHMM)
        # samples from the same true segment share a state
        restored = np.empty(90, dtype=int)
        restored[perm] = states
        self.assertGreaterEqual((restored[:30] == np.bincount(restored[:30]).argmax()).mean(), 0.9)

    def test_order_length_mismatch(self):
        with self.assertRaises(ValueError):
            segment_by_order(np.zeros(5), np.zeros(4))


if __name__ == "__main__":
    unittest.main()
