import unittest
import numpy as np
import torch
from embkit.constraints import NetworkConstraint  # adjust import path to match your structure


class TestNetworkConstraint(unittest.TestCase):
    def setUp(self):
        self.feature_index = ["f1", "f2", "f3", "f4"]
        self.latent_index = ["z1", "z2"]
        self.latent_membership = {
            "z1": ["f1", "f2"],
            "z2": ["f3"]
        }

    def test_no_membership_active_constraint_is_all_ones(self):
        constraint = NetworkConstraint(self.feature_index, self.latent_index, None)
        expected = np.ones((2, 4), dtype=np.float32)
        np.testing.assert_array_equal(constraint._mask_np, expected)

    def test_membership_creates_correct_mask(self):
        constraint = NetworkConstraint(self.feature_index, self.latent_index, self.latent_membership)
        expected = np.array([
            [1, 1, 0, 0],  # z1 connects to f1, f2
            [0, 0, 1, 0]   # z2 connects to f3
        ], dtype=np.float32)
        np.testing.assert_array_equal(constraint._mask_np, expected)

    def test_inactive_constraint_is_all_ones(self):
        constraint = NetworkConstraint(self.feature_index, self.latent_index, self.latent_membership)
        constraint.set_active(False)
        expected = np.ones((2, 4), dtype=np.float32)
        np.testing.assert_array_equal(constraint._mask_np, expected)

    def test_update_membership_changes_mask(self):
        constraint = NetworkConstraint(self.feature_index, self.latent_index, self.latent_membership)
        new_membership = {
            "z1": ["f4"],
            "z2": ["f2", "f3"]
        }
        constraint.update_membership(new_membership)
        expected = np.array([
            [0, 0, 0, 1],  # z1 -> f4
            [0, 1, 1, 0]   # z2 -> f2, f3
        ], dtype=np.float32)
        np.testing.assert_array_equal(constraint._mask_np, expected)

    def test_as_torch_tensor(self):
        constraint = NetworkConstraint(self.feature_index, self.latent_index, self.latent_membership)
        tensor = constraint.as_torch(device=torch.device("cpu"))
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (2, 4))
        expected = torch.tensor([
            [1, 1, 0, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32)
        self.assertTrue(torch.equal(tensor, expected))

    def test_latent_not_in_membership_triggers_continue(self):
        feature_index = ["f1", "f2"]
        latent_index = ["z1", "z2"]  # z1 is included
        latent_membership = {
            "z2": ["f2"]  # z1 is missing from the mapping
        }

        constraint = NetworkConstraint(feature_index, latent_index, latent_membership)

        # z1 gets skipped entirely, so only z2 -> f2 matters
        expected = np.array([
            [0, 0],  # z1 skipped
            [0, 1]  # z2 -> f2
        ], dtype=np.float32)

        np.testing.assert_array_equal(constraint._mask_np, expected)

if __name__ == '__main__':
    unittest.main()