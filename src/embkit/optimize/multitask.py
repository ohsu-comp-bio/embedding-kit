
from itertools import cycle

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


def _unique_parameters(*models):
    seen = set()
    unique_params = []
    for model in models:
        for parameter in model.parameters():
            parameter_id = id(parameter)
            if parameter_id not in seen:
                seen.add(parameter_id)
                unique_params.append(parameter)
    return unique_params


def _task_loss(criterion, model, batch, device=None, unroll_inputs=False, auto_encoder=False):
    if auto_encoder:
        inputs = batch
        targets = batch
    else:
        inputs, targets = batch

    if device is not None:
        if isinstance(inputs, (tuple, list)):
            inputs = [x.to(device) if torch.is_tensor(x) else x for x in inputs]
        elif torch.is_tensor(inputs):
            inputs = inputs.to(device)
        if torch.is_tensor(targets):
            targets = targets.to(device)

    if auto_encoder:
        if isinstance(inputs, (tuple, list)) and unroll_inputs:
            inputs = inputs[0]
        res = model(inputs)
        total_loss, recon_loss, kl_loss = criterion(res.recon, inputs, res.mu, res.logvar)
        return total_loss, res.recon, inputs


    if isinstance(inputs, (tuple, list)) and unroll_inputs:
        outputs = model(*inputs)
    else:
        outputs = model(inputs)

    loss = criterion(outputs, targets)
    return loss, outputs, targets


class LearningTask:
    """Defines a learning task with model, dataset, and training configuration."""

    def __init__(self, model, dataset, batch_size, criterion, weight=1.0, auto_encoder=False, unroll_inputs=False):
        """
        Initialize a LearningTask.

        Args:
            model: The model to train.
            dataset: The dataset for training.
            batch_size: Batch size for data loading.
            criterion: Loss function.
            weight: Task weight for weighted multitask learning (default: 1.0).
        """
        self.model = model
        self.dataset = dataset
        self.criterion = criterion
        self.weight = weight
        self.batch_size = batch_size
        self.auto_encoder = auto_encoder
        self.unroll_inputs = unroll_inputs



def _prepare_learning_tasks(tasks):
    if tasks is None or len(tasks) == 0:
        raise ValueError("tasks must contain at least one LearningTask.")

    for idx, task in enumerate(tasks):
        if not isinstance(task, LearningTask):
            raise TypeError(f"tasks[{idx}] must be a LearningTask instance.")
        if task.model is None:
            raise ValueError(f"tasks[{idx}].model must not be None.")
        if task.dataset is None:
            raise ValueError(f"tasks[{idx}].dataset must not be None.")
        if task.criterion is None:
            raise ValueError(f"tasks[{idx}].criterion must not be None.")
        if task.batch_size is None or task.batch_size <= 0:
            raise ValueError(f"tasks[{idx}].batch_size must be > 0.")

    loaders = [DataLoader(task.dataset, batch_size=task.batch_size, shuffle=True) for task in tasks]
    params = _unique_parameters(*(task.model for task in tasks))
    return loaders, params


def _normalize_task_schedule(task_schedule, task_count):
    if task_schedule is None:
        normalized = list(range(task_count))
    else:
        normalized = []
        for entry in task_schedule:
            if isinstance(entry, int):
                idx = entry
            elif isinstance(entry, str) and len(entry) == 1 and entry.isalpha():
                # Supports shorthand labels like "a", "b", "c".
                idx = ord(entry.lower()) - ord("a")
            else:
                raise ValueError("task_schedule entries must be integer task indices or single-letter labels.")

            if idx < 0 or idx >= task_count:
                raise ValueError(f"task_schedule entry {entry!r} resolves to invalid index {idx} for {task_count} tasks.")
            normalized.append(idx)

    if len(normalized) == 0:
        raise ValueError("task_schedule must contain at least one task index.")
    return tuple(normalized)


def multi_task_train_weighted_sync(
    tasks,
    epochs=5,
    lr=0.001,
    pairing_mode="truncate",
    gradient_clip_norm=None,
    device=None,
    lr_gamma=0.5,
):
    """
    Weighted multitask training over an arbitrary list of LearningTask.

    pairing_mode:
      - "truncate": stop at the shortest task loader.
      - "balanced_cycle": cycle shorter loaders to run for the longest loader length.
    """
    if pairing_mode not in {"truncate", "balanced_cycle"}:
        raise ValueError("pairing_mode must be one of: 'truncate', 'balanced_cycle'.")

    loaders, trainable_params = _prepare_learning_tasks(tasks)

    optimizer = Adam(trainable_params, lr=lr)
    scheduler = None
    if lr_gamma is not None:
        scheduler = StepLR(optimizer, step_size=1, gamma=lr_gamma)

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        if pairing_mode == "truncate":
            steps = min(len(loader) for loader in loaders)
            iterators = [iter(loader) for loader in loaders]
        else:
            steps = max(len(loader) for loader in loaders)
            iterators = [cycle(loader) for loader in loaders]

        batch_bar = tqdm(range(steps), total=steps, position=1, leave=False, desc=f"Epoch {epoch}")

        for _ in batch_bar:
            optimizer.zero_grad()
            task_losses = []
            total_loss = None

            for task_idx, task in enumerate(tasks):
                batch = next(iterators[task_idx])
                loss, _, _ = _task_loss(
                    task.criterion,
                    task.model,
                    batch,
                    device=device,
                    unroll_inputs=task.unroll_inputs,
                    auto_encoder=task.auto_encoder
                )
                task_losses.append(loss)
                weighted_loss = task.weight * loss
                total_loss = weighted_loss if total_loss is None else (total_loss + weighted_loss)

            total_loss.backward()

            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip_norm)

            optimizer.step()

            postfix = {
                "total_loss": float(total_loss.detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
            }
            postfix.update({f"loss_{i}": float(loss.detach().cpu()) for i, loss in enumerate(task_losses)})
            pbar.set_postfix(**postfix)

        if scheduler is not None:
            scheduler.step()


def multi_task_train_interleaved(
    tasks,
    epochs=5,
    lr=0.001,
    task_schedule=None,
    steps_per_epoch=None,
    gradient_clip_norm=None,
    device=None,
    lr_gamma=0.5,
):
    """
    Interleaved multitask training over an arbitrary list of LearningTask.

    task_schedule: list of task indices or single-letter labels (a, b, c, ...) to define task order.
      If None, cycles through all tasks in order.
    steps_per_epoch: number of batches per epoch. If None, uses the longest loader length.
    """
    loaders, trainable_params = _prepare_learning_tasks(tasks)
    normalized_schedule = _normalize_task_schedule(task_schedule, len(tasks))

    optimizer = Adam(trainable_params, lr=lr)
    scheduler = None
    if lr_gamma is not None:
        scheduler = StepLR(optimizer, step_size=1, gamma=lr_gamma)

    if steps_per_epoch is None:
        steps_per_epoch = max(len(loader) for loader in loaders)

    schedule_cycle = cycle(normalized_schedule)

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        task_iters = [cycle(loader) for loader in loaders]

        batch_bar = tqdm(range(steps_per_epoch), position=1, leave=False, desc=f"Epoch {epoch}")
        for _ in batch_bar:
            optimizer.zero_grad()
            task_idx = next(schedule_cycle)
            task = tasks[task_idx]
            batch = next(task_iters[task_idx])

            loss, _, _ = _task_loss(
                task.criterion,
                task.model,
                batch,
                device=device,
                unroll_inputs=task.unroll_inputs,
                auto_encoder=task.auto_encoder
            )
            weighted_loss = task.weight * loss

            weighted_loss.backward()

            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip_norm)

            optimizer.step()

            pbar.set_postfix(
                task=task_idx,
                loss=float(loss.detach().cpu()),
                weighted_loss=float(weighted_loss.detach().cpu()),
                lr=optimizer.param_groups[0]["lr"],
            )
        if scheduler is not None:
            scheduler.step()
