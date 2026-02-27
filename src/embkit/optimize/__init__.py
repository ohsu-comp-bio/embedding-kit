"""
Docstring for embkit.optimize

"""

import logging
from typing import Dict, Optional, List, Union, Tuple, Any
from collections.abc import Callable

import pandas as pd

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm.autonotebook import tqdm

from .. import get_device, dataframe_loader


logger = logging.getLogger(__name__)


def _resolve_optimizer(model, lr: float,
                       optimizer: Optional[torch.optim.Optimizer]) -> torch.optim.Optimizer:
    if optimizer is not None:
        return optimizer    
    return Adam(model.parameters(), lr=lr)


def _loader_length(loader: DataLoader) -> Optional[int]:
    try:
        return len(loader)
    except TypeError:
        return None


def _resolve_phases(epochs: int,
                    beta: float,
                    beta_schedule: Optional[List[Tuple[float, int]]]) -> List[Tuple[float, int]]:
    if beta_schedule is None:
        return [(beta, epochs)]
    return [(float(beta_value), int(n_epochs)) for beta_value, n_epochs in beta_schedule]


def _ensure_history(model, keys: List[str]) -> Dict[str, List[float]]:
    history = getattr(model, "history", None)
    if not isinstance(history, dict):
        history = {}
    for key in keys:
        history.setdefault(key, [])
    model.history = history
    return history


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().to(device="cpu"))
    return float(value)


def _move_to_device(value: Any,
                    device: torch.device,
                    non_blocking: bool = False) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device, non_blocking=non_blocking) for v in value)
    if isinstance(value, list):
        return [_move_to_device(v, device, non_blocking=non_blocking) for v in value]
    if isinstance(value, dict):
        return {k: _move_to_device(v, device, non_blocking=non_blocking) for k, v in value.items()}
    return value


def _run_training_phases(
                         model,
                         loader: DataLoader,
                         optimizer: torch.optim.Optimizer,
                         phases: List[Tuple[float, int]],
                         metric_keys: List[str],
                         step_fn: Callable[[Any, float], Dict[str, torch.Tensor]],
                         progress: bool,
                         accumulate_steps: int) -> float:
    if accumulate_steps < 1:
        raise ValueError("accumulate_steps must be >= 1")

    total_epochs = sum(n_epochs for _, n_epochs in phases)
    global_bar = tqdm(total=total_epochs,
                      disable=not progress,
                      desc="training",
                      position=0)

    last_loss = 0.0
    epoch_number = 0

    for beta_value, n_epochs in phases:
        for _ in range(n_epochs):
            epoch_number += 1
            model.train()

            metric_sums = {key: 0.0 for key in metric_keys}
            batches = 0
            optimizer.zero_grad(set_to_none=True)

            epoch_desc = f"epoch {epoch_number}/{total_epochs} β={beta_value:.3f}"
            batch_bar = tqdm(loader,
                             total=_loader_length(loader),
                             disable=not progress,
                             position=1,
                             leave=False,
                             desc=epoch_desc)

            for batch_index, batch in enumerate(batch_bar):
                metrics = step_fn(batch, beta_value)
                loss = metrics["loss"]
                (loss / accumulate_steps).backward()

                if (batch_index + 1) % accumulate_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                for key in metric_keys:
                    metric_sums[key] += _to_float(metrics[key])
                batches += 1
                last_loss = _to_float(metrics["loss"])

                if progress and batches > 0:
                    batch_bar.set_postfix({key: f"{metric_sums[key] / batches:.4f}" for key in metric_keys})

            if batches > 0 and (batches % accumulate_steps) != 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            history = _ensure_history(model, metric_keys)
            if batches > 0:
                for key in metric_keys:
                    history[key].append(metric_sums[key] / batches)
            else:
                for key in metric_keys:
                    history[key].append(float("nan"))

            if "beta" in history:
                history["beta"].append(beta_value)

            if progress:
                global_bar.update(1)
                global_bar.set_postfix({key: f"{history[key][-1]:.4f}" for key in metric_keys})

    global_bar.close()
    return last_loss


def fit(model, X: Union[torch.Tensor, Dataset, DataLoader],
    y: Optional[torch.Tensor] = None, 
        epochs: int = 20, 
        lr: Optional[float] = 1e-3, 
        beta: float = 1.0,
        optimizer: Optional[torch.optim.Optimizer] = None, 
        loss: Optional[Callable] = None, 
        device: Optional[torch.device] = None, 
        progress: bool = True, 
        pin_memory: bool = True, 
        batch_size: int = 256,
        shuffle: bool = True,
        accumulate_steps: int = 1
    ):
    """
    General training loop for a PyTorch model.

    X: Tensor, Dataset, or DataLoader. If Tensor, y must be provided.

    """

    # --- setup ---
    if lr is None:
        lr = model.lr
    if device is None:
        device = get_device()

    model.to(device)
    model.train()

    opt = _resolve_optimizer(model=model, lr=lr, optimizer=optimizer)
    if loss is None:
        criterion = nn.MSELoss()
    else:
        criterion = loss

    history = _ensure_history(model, ["loss"])
    history["loss"].clear()

    if isinstance(X, torch.Tensor):
        if y is None:
            raise ValueError("y is required when X is a tensor")
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    elif isinstance(X, pd.DataFrame):
        dataloader = dataframe_loader(X, batch_size=batch_size, shuffle=shuffle, device=device)
    elif isinstance(X, Dataset):
        dataloader = DataLoader(X, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    elif isinstance(X, DataLoader):
        dataloader = X
    else:
        raise TypeError("X must be a Tensor, Dataset, or DataLoader")

    phases = _resolve_phases(epochs=epochs, beta=beta, beta_schedule=None)

    def recognizer_step(batch, beta_value: float) -> Dict[str, torch.Tensor]:
        del beta_value
        inputs, outputs = batch
        inputs = _move_to_device(inputs, device=device, non_blocking=pin_memory)
        outputs = _move_to_device(outputs, device=device, non_blocking=pin_memory)

        predictions = model(inputs)
        loss = criterion(predictions, outputs)
        return {"loss": loss}

    return _run_training_phases(
        model=model,
        loader=dataloader,
        optimizer=opt,
        phases=phases,
        metric_keys=["loss"],
        step_fn=recognizer_step,
        progress=progress,
        accumulate_steps=accumulate_steps,
    )


def fit_vae(model, 
            X: Union[pd.DataFrame, torch.Tensor, torch.utils.data.DataLoader], 
            epochs: int = 20, 
            lr: Optional[float] = 1e-3,
            beta: float = 1.0,
            optimizer: Optional[torch.optim.Optimizer] = None,
            beta_schedule: Optional[List[Tuple[float, int]]] = None,
            loss: Optional[Callable] = None,
            device: Optional[torch.device] = None,
            progress: bool = True,
            accumulate_steps: int = 1,
            shuffle: bool = True,
            batch_size: int = 256,):
    """
    Training loop using vae_loss(recon, x, mu, logvar).

    X: pandas.DataFrame with float features, columns must match `self.features`.
    If `beta_schedule` is provided as a list of (beta, epochs) pairs, it overrides
    the single-phase (beta, epochs) arguments and runs multiple phases while
    reusing the same optimizer/momentum.
    """

    if loss is None:
        raise ValueError("loss function is required (e.g., from embkit.losses.vae_loss)")

    # --- setup ---
    if lr is None:
        lr = model.lr
    if device is None:
        device = get_device()

    if beta_schedule is not None:
        logger.info("Using beta_schedule: %s", beta_schedule)

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
        data_loader = dataframe_loader(X, batch_size=batch_size, shuffle=shuffle, device=device)
    elif isinstance(X, torch.Tensor):
        data_loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=shuffle)
    elif isinstance(X, DataLoader):
        data_loader = X
    else:
        raise TypeError("X must be DataFrame, Tensor, or DataLoader")

    opt = _resolve_optimizer(model=model, lr=lr, optimizer=optimizer)

    history = _ensure_history(model, ["loss", "recon", "kl", "beta"])
    history["loss"].clear()
    history["recon"].clear()
    history["kl"].clear()
    history["beta"].clear()

    phases = _resolve_phases(epochs=epochs, beta=beta, beta_schedule=beta_schedule)

    def vae_step(batch, beta_value: float) -> Dict[str, torch.Tensor]:
        (x_tensor,) = batch
        x_tensor = x_tensor.to(device).float()

        recon, mu, logvar, _ = model(x_tensor)
        total_loss, recon_loss, kl_loss = loss(recon, x_tensor, mu, logvar, beta=beta_value)

        return {
            "loss": total_loss,
            "recon": recon_loss,
            "kl": kl_loss,
        }

    return _run_training_phases(
        model=model,
        loader=data_loader,
        optimizer=opt,
        phases=phases,
        metric_keys=["loss", "recon", "kl"],
        step_fn=vae_step,
        progress=progress,
        accumulate_steps=accumulate_steps,
    )


def fit_net_vae(
    model: nn.Module,
    X: Union[pd.DataFrame, torch.Tensor],
    *,
    latent_dim: Optional[int] = None,
    latent_index: Optional[List[str]] = None,
    latent_groups: Optional[Dict[str, List[str]]] = None,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    epochs: int = 80,
    phases: Optional[List[int]] = None,
    device: Optional[str] = None,
    grouping_fn: Optional[Callable[[Any, List[str]], Dict[str, List[str]]]] = None,
) -> None:
    """Train a NetVAE with optional alternating constraint phases."""

    import numpy as np
    from ..constraints import NetworkConstraint
    from ..losses import net_vae_loss
    from ..models.vae.encoder import Encoder
    from ..models.vae.base_vae import BaseVAE

    if isinstance(X, torch.Tensor):
        if not getattr(model, "features", None):
            raise ValueError("Tensor input requires model.features to be defined.")
        df = pd.DataFrame(X.detach().cpu().numpy(), columns=model.features)
    else:
        df = X

    if latent_index is None:
        if latent_dim is None:
            raise ValueError("Provide latent_dim or latent_index.")
        latent_index = [f"z{i}" for i in range(latent_dim)]
    else:
        latent_index = list(latent_index)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    torch_device = torch.device(device)

    constraint = NetworkConstraint(list(df.columns), latent_index, latent_groups)
    if getattr(model, "encoder", None) is None or getattr(model, "decoder", None) is None:
        feature_dim = len(df.columns)
        model.encoder = Encoder(feature_dim=feature_dim, latent_dim=len(latent_index), constraint=constraint)
        model.decoder = BaseVAE.build_decoder(feature_dim=feature_dim, latent_dim=len(latent_index))
    else:
        model.encoder.constraint = constraint

    model.latent_index = list(latent_index)

    model.to(torch_device)
    x_tensor = torch.tensor(df.values, dtype=torch.float32, device=torch_device)
    data_loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.history = {"loss": [], "reconstruction_loss": [], "kl_loss": []}

    def refresh_mask():
        if model.encoder is None:
            raise RuntimeError("Encoder must be initialized before refreshing mask.")
        model.encoder.refresh_mask(torch_device)

    def start_constrained_phase():
        if grouping_fn is not None and model.encoder is not None:
            with torch.no_grad():
                weights = model.encoder.pathway.linear.weight.detach().cpu().numpy()
                new_groups = grouping_fn(weights, list(df.columns))
                constraint.update_membership(new_groups)
        constraint.set_active(True)
        refresh_mask()

    def start_unconstrained_phase():
        constraint.set_active(False)
        refresh_mask()

    constraint.set_active(False)
    refresh_mask()

    total_epochs = sum(phases) if phases else epochs
    boundaries = np.cumsum(phases).tolist() if phases else []

    for epoch in range(total_epochs):
        if boundaries and epoch in boundaries:
            boundary_index = boundaries.index(epoch)
            if boundary_index % 2 == 0:
                start_constrained_phase()
            else:
                start_unconstrained_phase()

        model.train()
        epoch_tot = epoch_rec = epoch_kl = 0.0
        n_batches = 0

        for (batch_x,) in data_loader:
            optimizer.zero_grad()
            total_loss, recon_loss, kl_loss = net_vae_loss(model, batch_x)
            total_loss.backward()
            optimizer.step()
            epoch_tot += float(total_loss.item())
            epoch_rec += float(recon_loss.item())
            epoch_kl += float(kl_loss.item())
            n_batches += 1

        def _append_history(key: str, value: float) -> None:
            model.history[key].append(value)

        batch_count = max(1, n_batches)
        _append_history("loss", epoch_tot / batch_count)
        _append_history("reconstruction_loss", epoch_rec / batch_count)
        _append_history("kl_loss", epoch_kl / batch_count)

        if epoch % 2 == 0:
            print(f"Epoch {epoch + 1}/{total_epochs} | ")
            print(
                f"loss={model.history['loss'][-1]:.4f} | "
                f"recon={model.history['reconstruction_loss'][-1]:.4f} | "
                f"kl={model.history['kl_loss'][-1]:.4f}"
            )
            logger.info(
                "Epoch %d/%d | loss=%.4f | recon=%.4f | kl=%.4f",
                epoch + 1,
                total_epochs,
                model.history["loss"][-1],
                model.history["reconstruction_loss"][-1],
                model.history["kl_loss"][-1],
            )

    model.latent_groups = constraint.latent_membership

    model.eval()
    with torch.no_grad():
        if model.encoder is None or model.decoder is None:
            raise RuntimeError("Encoder and decoder must be initialized before evaluation.")
        mu, _, _ = model.encoder(x_tensor)
        recon = model.decoder(mu).cpu().numpy()

    normal_pred = pd.DataFrame(recon, index=df.index, columns=df.columns)
    resid = normal_pred - df
    model.normal_stats = pd.DataFrame({"mean": resid.mean(), "std": resid.std(ddof=0)})


def fit_alt(model, loader, lr:float = 1e-5, epochs=32, accumulate_steps=8):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.history = {"loss": []}

    def step_fn(batch, beta_value: float) -> Dict[str, torch.Tensor]:
        del beta_value
        x_tensor, y_tensor = batch
        predictions = model(x_tensor)
        loss = criterion(predictions, y_tensor)
        return {"loss": loss}

    _run_training_phases(
        model=model,
        loader=loader,
        optimizer=optimizer,
        phases=[(1.0, int(epochs))],
        metric_keys=["loss"],
        step_fn=step_fn,
        progress=True,
        accumulate_steps=int(accumulate_steps),
    )

    return [{"epoch": i + 1, "loss": loss} for i, loss in enumerate(model.history["loss"])]