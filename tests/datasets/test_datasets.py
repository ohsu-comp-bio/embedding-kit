import itertools
import unittest

import torch
from torch.utils.data import Dataset

from embkit.datasets import BalancedMixer, DatasetMask


class TinyDataset(Dataset):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.values[idx]


class TestDatasets(unittest.TestCase):
    def test_balanced_mixer_iterates_and_recycles(self):
        d1 = ["a1", "a2"]
        d2 = ["b1"]
        mix = BalancedMixer([d1, d2], seed=42)

        out = list(itertools.islice(iter(mix), 8))
        self.assertEqual(len(out), 8)
        self.assertTrue(any(v.startswith("a") for v in out))
        self.assertTrue(any(v.startswith("b") for v in out))

    def test_dataset_mask_applies_mask_and_device(self):
        base = TinyDataset(
            [
                (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([5.0, 6.0])),
                (torch.tensor([4.0, 5.0, 6.0]), torch.tensor([7.0, 8.0])),
            ]
        )
        mask = [torch.tensor([True, False, True]), torch.tensor([False, True])]

        masked = DatasetMask(base, mask, device=torch.device("cpu"))
        self.assertEqual(len(masked), 2)

        x0, y0 = masked[0]
        self.assertTrue(torch.equal(x0, torch.tensor([1.0, 3.0])))
        self.assertTrue(torch.equal(y0, torch.tensor([6.0])))
        self.assertEqual(x0.device.type, "cpu")
        self.assertEqual(y0.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
