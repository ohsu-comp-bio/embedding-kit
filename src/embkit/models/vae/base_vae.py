"""
Base VAE class
"""
from typing import Type, Any, List, Optional, Dict, overload, TypeVar, Union
from abc import ABC, abstractmethod
from pathlib import Path
import json
import logging
import pandas as pd
from torch import nn
import torch
from .encoder import Encoder
from .decoder import Decoder
from ...layers import LayerInfo

logger = logging.getLogger(__name__)
T = TypeVar("T")


class BaseVAE(nn.Module, ABC):
    """
    Minimal VAE wrapper to hold encoder/decoder and provide forward().
    Allows late-binding of encoder/decoder by subclasses.
    """

    @staticmethod
    def build_encoder(feature_dim: int, latent_dim: int,
                      layers: List[LayerInfo] = None,
                      constraint=None,
                      batch_norm: bool = False) -> Encoder:
        return Encoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=layers,
            constraint=constraint,
            batch_norm=batch_norm
        )

    @staticmethod
    def build_decoder(feature_dim: int, latent_dim: int, layers: List[LayerInfo] = None,
                      constraint=None, batch_norm: bool = False) -> Decoder:
        return Decoder(latent_dim, feature_dim, layers=layers,
                       constraint=constraint)

    @overload
    @staticmethod
    def open_model(
            path: str,
            device: Optional[str] = ...,
            model_cls: None = ...,
            model_kwargs: Optional[Dict[str, Any]] = ...
    ) -> "NetVae":
        ...

    # Overload 2: model_cls provided -> returns that class
    @overload
    @staticmethod
    def open_model(
            path: str,
            device: Optional[str] = ...,
            model_cls: Type[T] = ...,
            model_kwargs: Optional[Dict[str, Any]] = ...
    ) -> T:
        ...

    @staticmethod
    def open_model(
            path: str,
            device: Optional[str] = None,
            model_cls: Optional[Type[T]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None
    ) -> T:
        """
        Load a model saved by NetVae.save (PyTorch-compatible).
        You can select the container class via `model_cls` (default: NetVae).
        The container is expected to accept at least: (features, encoder=..., decoder=...).

        Works with or without net.groups.tsv:
          - If groups exist: load names + groups
          - Else: try latent.index
          - Else: infer latent_dim from encoder weights and synthesize names
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)

        model_kwargs = dict(model_kwargs or {})

        # --- Features (required) ---
        index_path = Path(path, "index")
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Missing features index file: {index_path}")
        with open(index_path, "rt", encoding="ascii") as handle:
            features = json.load(handle)
        feature_dim = len(features)

        # --- Determine latent names/dim ---
        groups_path = Path(path, "net.groups.tsv")
        latent_idx_path = Path(path, "latent.index")
        groups: Optional[Dict[str, List[str]]] = None
        latent_index: Optional[List[str]] = None
        latent_dim: Optional[int] = None

        if Path(groups_path).exists():
            latent_index = []
            groups = {}
            with open(groups_path, "rt", encoding="ascii") as fh:
                for line in fh:
                    row = line.rstrip().split("\t")
                    if not row or row[0] == "":
                        continue
                    latent_index.append(row[0])
                    groups[row[0]] = row[1:]
            latent_dim = len(latent_index)
        elif Path(latent_idx_path).exists():
            with open(latent_idx_path, "rt", encoding="ascii") as fh:
                latent_index = [ln.strip() for ln in fh if ln.strip() != ""]
            if latent_index:
                latent_dim = len(latent_index)
            else:
                latent_index = None  # fall through to weight inference

        if latent_dim is None:
            # --- Infer from encoder weights ---
            enc_state_path = Path(path, "model.enc.pt")
            if not Path(enc_state_path).exists():
                raise FileNotFoundError(f"Missing encoder weights: {enc_state_path}")
            enc_state = torch.load(enc_state_path, map_location=device)

            if "pathway.linear.weight" in enc_state:
                latent_dim = int(enc_state["pathway.linear.weight"].shape[0])
            elif "z_mean.weight" in enc_state:
                latent_dim = int(enc_state["z_mean.weight"].shape[0])
            else:
                cand = None
                for k, v in enc_state.items():
                    if isinstance(v, torch.Tensor) and v.ndim == 2:
                        cand = int(v.shape[0])
                        break
                if cand is None:
                    raise ValueError(
                        "Could not infer latent_dim from encoder state. "
                        "Expected 'pathway.linear.weight' or 'z_mean.weight' in state dict."
                    )
                latent_dim = cand

            if latent_index is None:
                latent_index = [f"z{i}" for i in range(latent_dim)]

        # --- Build fresh modules and load weights ---
        enc = BaseVAE.build_encoder(feature_dim, latent_dim, constraint=None)
        dec = BaseVAE.build_decoder(feature_dim, latent_dim)

        enc.load_state_dict(torch.load(Path(path, "model.enc.pt"), map_location=device))
        dec.load_state_dict(torch.load(Path(path, "model.dec.pt"), map_location=device))
        enc.to(device)
        dec.to(device)

        # --- Instantiate user-provided container ---
        try:
            out = model_cls(features, encoder=enc, decoder=dec, **model_kwargs)
        except TypeError as e:
            raise TypeError(
                f"{model_cls.__name__} could not be constructed with "
                f"(features, encoder=..., decoder=..., **model_kwargs). "
                f"Error: {e}"
            ) from e

        # Optional standard fields if the container supports them
        # (we won't crash if the attributes don't exist)
        if hasattr(out, "latent_groups"):
            out.latent_groups = groups
        if hasattr(out, "latent_index"):
            out.latent_index = latent_index

        stats_path = Path(path, "training.stats.tsv")
        if Path(stats_path).exists():
            try:
                if hasattr(out, "normal_stats"):
                    out.normal_stats = pd.read_csv(stats_path, sep="\t", index_col=0)
            except Exception:
                logger.warning(f"Could not read {stats_path}")
                pass

        return out

    def __init__(self, features: List[str], encoder: Optional[Encoder] = None, decoder: Optional[Decoder] = None,
                 **kwargs):
        super().__init__()
        self.features = list(features)
        self.encoder: Optional[Encoder] = encoder
        self.decoder: Optional[Decoder] = decoder
        self.extra_args = kwargs

    def forward(self, x: torch.Tensor):
        if self.encoder is None or self.decoder is None:
            raise RuntimeError(
                "VAE encoder/decoder not initialized. Set self.encoder/self.decoder before calling forward().")
        mu, logvar, z = self.encoder(x)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, torch.Tensor, torch.utils.data.DataLoader], **kwargs):
        """
        Train the VAE on input data.

        Subclasses must override this to implement their training loop.
        `X` may be a pandas.DataFrame or a torch.Tensor; additional training
        parameters should be provided via **kwargs (e.g., epochs, lr, device).
        """
        raise NotImplementedError("Subclasses must implement fit() in VAE subclasses.")

    def save(self, path: str, normal_df: Optional[pd.DataFrame] = None):
        """Save VAE model with associated elements (PyTorch-native)."""
        if not Path(path).exists():
            Path(path).mkdir(parents=True, exist_ok=True)

        # Save encoder/decoder state dicts
        torch.save(self.encoder.state_dict(), Path(path, "model.enc.pt"))
        torch.save(self.decoder.state_dict(), Path(path, "model.dec.pt"))

        # Save feature index
        with open(Path(path, "index"), "wt", encoding="ascii") as handle:
            handle.write(json.dumps(list(self.features)))

        # Save latent groups (if present)
        if self.latent_index is not None and self.latent_groups is not None:
            with open(Path(path, "net.groups.tsv"), "wt", encoding="ascii") as handle:
                for g in self.latent_index:
                    n = self.latent_groups.get(g, [])
                    handle.write("\t".join([str(g), *n]) + "\n")

        # Save history
        with open(Path(path, "stats.json"), "wt", encoding="ascii") as handle:
            handle.write(json.dumps({"loss_history": self.history["loss"] if self.history else []}))

        # Optionally save training stats like original
        if normal_df is not None and self.normal_stats is not None:
            self.normal_stats.to_csv(Path(path, "training.stats.tsv"), sep="\t")
