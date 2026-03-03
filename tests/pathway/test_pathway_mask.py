import unittest
from pathlib import Path

import numpy as np

from embkit.pathway import build_sif_mask


class TestPathwayMask(unittest.TestCase):
    def test_build_sif_mask_filters_relation_and_indices(self):
        sif_path = Path(__file__).resolve().parents[1] / "data" / "sample_pathway.sif"

        src_index = {"TF1": 0, "TF2": 1}
        dst_index = {"G1": 0, "G2": 1, "G3": 2}

        mask = build_sif_mask(
            str(sif_path),
            src_index,
            dst_index,
            relation="controls-expression-of",
        )

        expected = np.array(
            [
                [1.0, 0.0],  # G1 <- TF1
                [0.0, 1.0],  # G2 <- TF2
                [1.0, 0.0],  # G3 <- TF1
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(mask, expected)


if __name__ == "__main__":
    unittest.main()
