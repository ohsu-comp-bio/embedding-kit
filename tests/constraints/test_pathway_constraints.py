import unittest
import numpy as np

from embkit.constraints import PathwayConstraintInfo


class TestPathwayConstraintInfo(unittest.TestCase):
    def setUp(self):
        self.feature_index = ["f1", "f2", "f3", "f4"]
        self.group_index = ["z1", "z2"]
        self.feature_map = {
            "z1": ["f1", "f2"],
            "z2": ["f3"],
        }

    def test_features_to_group_mask_shape_and_values(self):
        c = PathwayConstraintInfo(
            "features-to-group",
            feature_map=self.feature_map,
            feature_index=self.feature_index,
            group_index=self.group_index,
        )
        mask = c.gen_mask(in_features=4, out_features=2)
        expected = np.array(
            [
                [1, 1, 0, 0],
                [0, 0, 1, 0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(mask, expected)

    def test_inactive_returns_ones(self):
        c = PathwayConstraintInfo(
            "features-to-group",
            feature_map=self.feature_map,
            feature_index=self.feature_index,
            group_index=self.group_index,
        )
        c.set_active(False)
        mask = c.gen_mask(in_features=4, out_features=2)
        np.testing.assert_array_equal(mask, np.ones((2, 4), dtype=np.float32))

    def test_update_membership_changes_mask(self):
        c = PathwayConstraintInfo(
            "features-to-group",
            feature_map=self.feature_map,
            feature_index=self.feature_index,
            group_index=self.group_index,
        )
        c.update_membership(
            {
                "z1": ["f4"],
                "z2": ["f2", "f3"],
            }
        )
        mask = c.gen_mask(in_features=4, out_features=2)
        expected = np.array(
            [
                [0, 0, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(mask, expected)


if __name__ == "__main__":
    unittest.main()
