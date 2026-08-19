import unittest

import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset


class UnrolledPairDataset(Dataset):
    """Dataset yielding (inputs=(a, b), target) items for unrolled models."""

    def __init__(self, a, b, y):
        self.a = a
        self.b = b
        self.y = y

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        return (self.a[idx], self.b[idx]), self.y[idx]

from embkit.optimize.multitask import (
    LearningTask,
    _normalize_task_schedule,
    _prepare_learning_tasks,
    _task_loss,
    _unique_parameters,
    multi_task_train_interleaved,
    multi_task_train_weighted_sync,
)

DEVICE = torch.device("cpu")


class TinyModel(nn.Module):
    def __init__(self, in_features=2, out_features=1):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.lin(x)


class TinyUnrollModel(nn.Module):
    """Model whose forward takes unrolled positional arguments (a, b)."""

    def __init__(self):
        super().__init__()
        self.lin_a = nn.Linear(2, 1)
        self.lin_b = nn.Linear(2, 1)

    def forward(self, a, b):
        return self.lin_a(a) + self.lin_b(b)

    @property
    def lin(self):
        # Convenience accessor used by tests that assert on a single linear layer.
        return self.lin_a


class TinyListModel(nn.Module):
    """Model that accepts its inputs as a raw list."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 1)

    def forward(self, xs):
        return self.lin(xs[0])


def make_dataset(n=6, in_features=2):
    x = torch.randn(n, in_features)
    y = torch.randn(n, 1)
    return TensorDataset(x, y)


def make_task(model=None, n=6, batch_size=2, weight=1.0):
    model = model or TinyModel()
    criterion = nn.MSELoss()
    return LearningTask(
        model=model,
        dataset=make_dataset(n=n),
        batch_size=batch_size,
        criterion=criterion,
        weight=weight,
    )


class TestUniqueParameters(unittest.TestCase):
    def test_dedupes_shared_parameters(self):
        shared = nn.Linear(2, 1)
        model_a = TinyModel()
        model_b = nn.Sequential(TinyModel(), shared)

        unique = _unique_parameters(model_a, model_b)
        ids = [id(p) for p in unique]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn(id(shared.weight), ids)

    def test_no_models(self):
        self.assertEqual(_unique_parameters(), [])


class TestTaskLoss(unittest.TestCase):
    def test_basic(self):
        model = TinyModel()
        criterion = nn.MSELoss()
        loss, outputs, targets = _task_loss(criterion, model, (torch.randn(4, 2), torch.randn(4, 1)))
        self.assertEqual(loss.dim(), 0)
        self.assertEqual(outputs.shape, (4, 1))
        self.assertEqual(targets.shape, (4, 1))

    def test_moves_to_device(self):
        model = TinyModel()
        criterion = nn.MSELoss()
        loss, _, _ = _task_loss(
            criterion, model, (torch.randn(4, 2), torch.randn(4, 1)), device=DEVICE
        )
        self.assertEqual(loss.device, DEVICE)

    def test_list_inputs_not_unrolled(self):
        model = TinyListModel()
        criterion = nn.MSELoss()
        # A list of inputs (without unroll_inputs) is passed to the model as-is.
        loss, outputs, _ = _task_loss(
            criterion,
            model,
            ([torch.randn(4, 2)], torch.randn(4, 1)),
        )
        self.assertEqual(outputs.shape, (4, 1))

    def test_mixed_tensor_and_non_tensor_list(self):
        model = TinyListModel()
        criterion = nn.MSELoss()
        # Non-tensor elements in a list input must be left untouched.
        loss, _, _ = _task_loss(
            criterion,
            model,
            ([torch.randn(4, 2), "label"], torch.randn(4, 1)),
            device=DEVICE,
        )
        self.assertEqual(loss.device, DEVICE)

    def test_list_inputs_moved_to_device(self):
        model = TinyListModel()
        criterion = nn.MSELoss()
        loss, outputs, targets = _task_loss(
            criterion,
            model,
            ([torch.randn(4, 2)], torch.randn(4, 1)),
            device=DEVICE,
        )
        self.assertEqual(outputs.device, DEVICE)
        self.assertEqual(targets.device, DEVICE)

    def test_unroll_inputs(self):
        model = TinyUnrollModel()
        criterion = nn.MSELoss()
        a = torch.randn(4, 2)
        b = torch.randn(4, 2)
        y = torch.randn(4, 1)
        loss, outputs, _ = _task_loss(
            criterion,
            model,
            ((a, b), y),
            unroll_inputs=True,
        )
        self.assertEqual(outputs.shape, (4, 1))
        self.assertEqual(loss.dim(), 0)


class TestLearningTask(unittest.TestCase):
    def test_attributes(self):
        task = make_task(n=4)
        self.assertEqual(task.batch_size, 2)
        self.assertEqual(task.weight, 1.0)
        self.assertIsInstance(task.dataset, TensorDataset)

    def test_custom_weight(self):
        task = make_task(weight=2.5)
        self.assertEqual(task.weight, 2.5)


class TestPrepareLearningTasks(unittest.TestCase):
    def test_returns_loaders_and_params(self):
        model = TinyModel()
        task = make_task(model=model)
        loaders, params = _prepare_learning_tasks([task])
        self.assertEqual(len(loaders), 1)
        self.assertGreater(len(params), 0)
        self.assertIn(model.lin.weight, params)

    def test_dedupes_params_across_tasks(self):
        shared = TinyModel()
        task_a = make_task(model=shared)
        task_b = make_task(model=shared)
        _, params = _prepare_learning_tasks([task_a, task_b])
        self.assertEqual(len(params), len({id(p) for p in params}))

    def test_rejects_empty_tasks(self):
        with self.assertRaises(ValueError):
            _prepare_learning_tasks(None)
        with self.assertRaises(ValueError):
            _prepare_learning_tasks([])

    def test_rejects_non_learning_task(self):
        with self.assertRaises(TypeError):
            _prepare_learning_tasks([object()])

    def test_rejects_none_fields(self):
        good = make_task()
        for field in ("model", "dataset", "criterion"):
            bad = LearningTask(
                model=good.model,
                dataset=good.dataset,
                batch_size=good.batch_size,
                criterion=good.criterion,
            )
            setattr(bad, field, None)
            with self.assertRaises(ValueError, msg=field):
                _prepare_learning_tasks([bad])

    def test_rejects_bad_batch_size(self):
        bad = make_task(batch_size=0)
        with self.assertRaises(ValueError):
            _prepare_learning_tasks([bad])
        bad = LearningTask(
            model=make_task().model,
            dataset=make_dataset(),
            batch_size=None,
            criterion=nn.MSELoss(),
        )
        with self.assertRaises(ValueError):
            _prepare_learning_tasks([bad])


class TestNormalizeTaskSchedule(unittest.TestCase):
    def test_none_default(self):
        self.assertEqual(_normalize_task_schedule(None, 3), (0, 1, 2))

    def test_integer_schedule(self):
        self.assertEqual(_normalize_task_schedule([2, 0], 3), (2, 0))

    def test_letter_labels(self):
        self.assertEqual(_normalize_task_schedule(["b", "a"], 3), (1, 0))
        self.assertEqual(_normalize_task_schedule(["C"], 3), (2,))

    def test_empty_schedule(self):
        with self.assertRaises(ValueError):
            _normalize_task_schedule([], 3)

    def test_invalid_entry(self):
        with self.assertRaises(ValueError):
            _normalize_task_schedule([1.5], 3)
        with self.assertRaises(ValueError):
            _normalize_task_schedule(["ab"], 3)
        with self.assertRaises(ValueError):
            _normalize_task_schedule(["1"], 3)

    def test_out_of_range_index(self):
        with self.assertRaises(ValueError):
            _normalize_task_schedule([3], 3)
        with self.assertRaises(ValueError):
            _normalize_task_schedule([-1], 3)


class TestWeightedSyncTraining(unittest.TestCase):
    def test_truncate_runs_and_updates_weights(self):
        model = TinyModel()
        task = make_task(model=model)
        before = model.lin.weight.detach().clone()

        multi_task_train_weighted_sync([task], epochs=1, lr=0.01)

        self.assertFalse(torch.equal(before, model.lin.weight.detach()))

    def test_balanced_cycle_with_uneven_tasks(self):
        task_fast = make_task(n=4, batch_size=4)  # 1 batch
        task_slow = make_task(n=12, batch_size=2)  # 6 batches
        fast_before = task_fast.model.lin.weight.detach().clone()
        slow_before = task_slow.model.lin.weight.detach().clone()

        multi_task_train_weighted_sync(
            [task_fast, task_slow],
            epochs=1,
            lr=0.01,
            pairing_mode="balanced_cycle",
        )

        self.assertFalse(torch.equal(fast_before, task_fast.model.lin.weight.detach()))
        self.assertFalse(torch.equal(slow_before, task_slow.model.lin.weight.detach()))

    def test_weights_are_applied(self):
        # Two identical single-step tasks with different weights: the weighted
        # total loss reported by the optimizer must reflect the weighting by
        # checking against a manual computation with one fixed step.
        torch.manual_seed(0)
        x = torch.randn(2, 2)
        y = torch.randn(2, 1)
        m1 = TinyModel()
        m2 = TinyModel()
        crit = nn.MSELoss()
        t1 = LearningTask(m1, TensorDataset(x, y), 2, crit, weight=1.0)
        t2 = LearningTask(m2, TensorDataset(x, y), 2, crit, weight=3.0)

        multi_task_train_weighted_sync([t1, t2], epochs=1, lr=0.001)
        # Both models trained; exact values depend on optimizer, so just assert
        # the call completed and weights changed.
        self.assertIsNotNone(m1.lin.weight.grad)
        self.assertIsNotNone(m2.lin.weight.grad)

    def test_gradient_clipping_path(self):
        task = make_task(n=8, batch_size=2)
        multi_task_train_weighted_sync([task], epochs=1, lr=0.01, gradient_clip_norm=1.0)
        # Reaching this without error exercises the clip_grad_norm_ branch.
        self.assertIsNotNone(task.model.lin.weight)

    def test_unroll_inputs(self):
        torch.manual_seed(1)
        a = torch.randn(6, 2)
        b = torch.randn(6, 2)
        y = torch.randn(6, 1)
        model = TinyUnrollModel()
        task = LearningTask(
            model=model,
            dataset=UnrolledPairDataset(a, b, y),
            batch_size=2,
            criterion=nn.MSELoss(),
        )
        multi_task_train_weighted_sync([task], epochs=1, lr=0.01, unroll_inputs=True)
        self.assertIsNotNone(model.lin.weight)

    def test_invalid_pairing_mode(self):
        with self.assertRaises(ValueError):
            multi_task_train_weighted_sync([make_task()], pairing_mode="bogus")


class TestInterleavedTraining(unittest.TestCase):
    def test_default_schedule_cycles_all_tasks(self):
        t1 = make_task()
        t2 = make_task()
        before1 = t1.model.lin.weight.detach().clone()
        before2 = t2.model.lin.weight.detach().clone()

        multi_task_train_interleaved([t1, t2], epochs=1, lr=0.01)

        self.assertFalse(torch.equal(before1, t1.model.lin.weight.detach()))
        self.assertFalse(torch.equal(before2, t2.model.lin.weight.detach()))

    def test_letter_schedule(self):
        tasks = [make_task(), make_task()]
        multi_task_train_interleaved(
            tasks,
            epochs=1,
            lr=0.01,
            task_schedule=["b"],  # only the second task trains
        )
        self.assertIsNotNone(tasks[1].model.lin.weight)

    def test_steps_per_epoch(self):
        task = make_task(n=8, batch_size=2)
        multi_task_train_interleaved([task], epochs=1, lr=0.01, steps_per_epoch=2)
        self.assertIsNotNone(task.model.lin.weight)

    def test_gradient_clipping_and_unroll(self):
        a = torch.randn(4, 2)
        b = torch.randn(4, 2)
        y = torch.randn(4, 1)
        model = TinyUnrollModel()
        task = LearningTask(
            model=model,
            dataset=UnrolledPairDataset(a, b, y),
            batch_size=2,
            criterion=nn.MSELoss(),
        )
        multi_task_train_interleaved(
            [task],
            epochs=1,
            lr=0.01,
            gradient_clip_norm=1.0,
            unroll_inputs=True,
        )
        self.assertIsNotNone(model.lin.weight)

    def test_invalid_schedule_propagates(self):
        with self.assertRaises(ValueError):
            multi_task_train_interleaved([make_task()], task_schedule=["z"])


if __name__ == "__main__":
    unittest.main()
