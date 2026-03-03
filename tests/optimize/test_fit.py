import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from embkit.optimize import fit


class TupleInputDataset(Dataset):
    def __init__(self, x_left: torch.Tensor, x_right: torch.Tensor):
        self.x_left = x_left
        self.x_right = x_right
        self.y = x_left + x_right

    def __len__(self) -> int:
        return self.x_left.shape[0]

    def __getitem__(self, idx: int):
        inputs = (self.x_left[idx], self.x_right[idx])
        target = self.y[idx]
        return inputs, target


class TupleInputModel(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, inputs):
        left, right = inputs
        return (left + right) * self.scale


class TestFitTupleInputs(unittest.TestCase):
    def test_fit_handles_tuple_inputs(self):
        torch.manual_seed(7)
        x_left = torch.randn(12, 4)
        x_right = torch.randn(12, 4)
        dataset = TupleInputDataset(x_left, x_right)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        model = TupleInputModel(width=4)

        final_loss = fit(model=model, X=loader, epochs=2, progress=False)

        self.assertIsInstance(final_loss, float)
        self.assertIn("loss", model.history)
        self.assertEqual(len(model.history["loss"]), 2)


if __name__ == "__main__":
    unittest.main()
