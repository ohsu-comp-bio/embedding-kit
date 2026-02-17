
import pandas as pd
import logging
from typing import Dict, Optional, List, Union
from collections.abc import Callable

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm.autonotebook import tqdm

from .. import get_device, dataframe_loader


logger = logging.getLogger(__name__)


def fit(self, X: Union[torch.Tensor],
        y: Union[torch.Tensor], **kwargs):

    epochs: int = int(kwargs.pop("epochs", 20))
    lr: Optional[float] = kwargs.pop("lr", None)
    beta: float = float(kwargs.pop("beta", 1.0))
    optimizer: Optional[torch.optim.Optimizer] = kwargs.pop("optimizer", None)
    # loss: Optional[Callable] = kwargs.pop("loss", None)
    reset_optimizer: bool = bool(kwargs.pop("reset_optimizer", False))
    device: Optional[torch.device] = kwargs.pop("device", None)
    progress: bool = bool(kwargs.pop("progress", True))
    pin_memory: bool = bool(kwargs.pop("pin_memory", True))

    # --- setup ---
    if lr is None:
        lr = self.lr
    if device is None:
        device = get_device()

    self.to(device)
    self.train()

    # --- persistent optimizer (reuse momentum/velocity across phases) ---
    if optimizer is not None:
        self._optimizer = optimizer
    elif reset_optimizer or not hasattr(self, "_optimizer") or self._optimizer is None:
        self._optimizer = Adam(self.parameters(), lr=lr)
    else:
        # Reuse existing optimizer but refresh LR if changed
        for g in self._optimizer.param_groups:
            g["lr"] = lr
    opt = self._optimizer
    critereon = nn.MSELoss()

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=pin_memory)

    # --- epoch runner (epoch-only progress) ---
    def run_epochs(n_epochs: int, beta_value: float) -> float:
        last_loss = 0.0
        epoch_batches = 0
        epoch_bar = tqdm(range(n_epochs), disable=not progress, desc=f"β={beta_value:.2f}")
        for epoch_idx in epoch_bar:
            epoch_loss_sum = 0.0
            epoch_batches = 0

            for inputs, outputs in dataloader:
                inputs = inputs.to(device, non_blocking=pin_memory)
                outputs = outputs.to(device, non_blocking=pin_memory)

                predictions = self(inputs)
                loss = critereon(predictions, outputs)

                # Backprop
                opt.zero_grad()
                loss.backward()
                opt.step()

                # Accumulate epoch stats
                tl = float(loss.detach().cpu())
                epoch_loss_sum += tl
                epoch_batches += 1
                last_loss = tl

            # Compute epoch means
            if epoch_batches > 0:
                ep_loss = epoch_loss_sum / epoch_batches
                self.history["loss"].append(ep_loss)

                # Update the epoch progress bar once per epoch (no jitter)
                if progress:
                    epoch_bar.set_postfix(loss=f"{ep_loss:.3f}")

        return last_loss

    return run_epochs(epochs, beta)


def fit_vae(model, X: Union[pd.DataFrame, torch.Tensor, torch.utils.data.DataLoader], **kwargs):
    """
    Training loop using vae_loss(recon, x, mu, logvar).

    X: pandas.DataFrame with float features, columns must match `self.features`.
    If `beta_schedule` is provided as a list of (beta, epochs) pairs, it overrides
    the single-phase (beta, epochs) arguments and runs multiple phases while
    reusing the same optimizer/momentum.
    """

    epochs: int = int(kwargs.pop("epochs", 20))
    lr: Optional[float] = kwargs.pop("lr", None)
    beta: float = float(kwargs.pop("beta", 1.0))
    optimizer: Optional[torch.optim.Optimizer] = kwargs.pop("optimizer", None)
    loss: Optional[Callable] = kwargs.pop("loss", None)
    reset_optimizer: bool = bool(kwargs.pop("reset_optimizer", False))
    device: Optional[torch.device] = kwargs.pop("device", None)
    progress: bool = bool(kwargs.pop("progress", True))
    beta_schedule = kwargs.pop("beta_schedule", None)
    y = kwargs.pop("y", None)  # if you need it, fetch it from kwargs

    if loss is None:
        raise ValueError("loss function is required (e.g., from embkit.losses.vae_loss)")

    # --- setup ---
    if lr is None:
        lr = model.lr
    if device is None:
        device = get_device()

    if beta_schedule is not None:
        logger.info(f"Using beta_schedule: {beta_schedule}")

    # Column alignment safety check if a DataFrame is passed
    if hasattr(X, "columns") and model.features is not None:
        if list(X.columns) != list(model.features):
            raise ValueError(
                "Input DataFrame columns do not match model features.\n"
                f"Data columns: {list(X.columns)[:5]}... (n={len(X.columns)})\n"
                f"Model features: {model.features[:5]}... (n={len(model.features)})"
            )

    model.to(device)
    model.train()

    # Build dataloader once
    if isinstance(X, pd.DataFrame):
        data_loader = dataframe_loader(X, device=device)  # ensure it shuffles in training mode
    else:
        data_loader = X

    # --- persistent optimizer (reuse momentum/velocity across phases) ---
    if optimizer is not None:
        model._optimizer = optimizer
    elif reset_optimizer or not hasattr(model, "_optimizer") or model._optimizer is None:
        model._optimizer = Adam(model.parameters(), lr=lr)
    else:
        # Reuse existing optimizer but refresh LR if changed
        for g in model._optimizer.param_groups:
            g["lr"] = lr
    opt = model._optimizer

    history = {
        "loss" : [],
        "recon" : [],
        "kl" : [],
    }
    # --- epoch runner (epoch-only progress) ---
    def run_epochs(n_epochs: int, beta_value: float) -> float:
        last_loss = 0.0
        epoch_bar = tqdm(range(n_epochs), disable=not progress, desc=f"β={beta_value:.2f}", position=0)
        for epoch_idx in epoch_bar:
            epoch_loss_sum = 0.0
            epoch_recon_sum = 0.0
            epoch_kl_sum = 0.0
            epoch_batches = 0

            for (x_tensor,) in tqdm(data_loader, position=1):
                opt.zero_grad(set_to_none=True)
                x_tensor = x_tensor.to(device).float()

                # Forward
                recon, mu, logvar, _ = model(x_tensor)

                total_loss, recon_loss, kl_loss = loss(recon, x_tensor, mu, logvar, beta=beta_value)

                # Backprop
                total_loss.backward()
                opt.step()

                # Accumulate epoch stats
                tl = float(total_loss.detach().cpu())
                epoch_loss_sum += tl
                epoch_recon_sum += float(recon_loss.detach().cpu())
                epoch_kl_sum += float(kl_loss.detach().cpu())
                epoch_batches += 1
                last_loss = tl

            # Compute epoch means
            if epoch_batches > 0:
                ep_loss = epoch_loss_sum / epoch_batches
                ep_recon = epoch_recon_sum / epoch_batches
                ep_kl = epoch_kl_sum / epoch_batches
                history["loss"].append(ep_loss)
                history["recon"].append(ep_recon)
                history["kl"].append(ep_kl)

                # Update the epoch progress bar once per epoch (no jitter)
                if progress:
                    epoch_bar.set_postfix(loss=f"{ep_loss:.3f}",
                                            recon=f"{ep_recon:.3f}",
                                            kl=f"{ep_kl:.3f}")

        return last_loss

    # --- single phase or multi-phase (beta schedule) ---
    if beta_schedule is None:
        return run_epochs(epochs, beta)
    else:
        last = 0.0
        for beta_value, n_epochs in beta_schedule:
            last = run_epochs(n_epochs, beta_value)
        return last
