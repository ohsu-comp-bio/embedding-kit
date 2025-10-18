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
                      layers: Optional[List[LayerInfo]] = None,
                      batch_norm: bool = False) -> Encoder:
        return Encoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=layers,
            batch_norm=batch_norm
        )

    @staticmethod
    def build_decoder(feature_dim: int, latent_dim: int,
                      layers: Optional[List[LayerInfo]] = None,
                      batch_norm: bool = False) -> Decoder:
        return Decoder(
            latent_dim=latent_dim,
            feature_dim=feature_dim,
            layers=layers,
            batch_norm=batch_norm
        )

    @overload
    @staticmethod
    def open_model(
        path: str,
        device: Optional[str] = ...,
        model_cls: Type[T] = ...,
        model_kwargs: Optional[Dict[str, Any]] = ...
    ) -> T: ...

    @staticmethod
    def open_model(
        path: str,
        device: Optional[str] = None,
        model_cls: Optional[Type[T]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None
    ) -> T:
        """
        Load a model saved by BaseVAE.save.
        Rebuilds the EXACT architecture when model.arch.json is present.
        Uses latent index files only for names, never for shapes.
        """
        if model_cls is None:
            # Default to BaseVAE container unless the caller passes a subclass.
            model_cls = BaseVAE  # type: ignore

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)

        model_kwargs = dict(model_kwargs or {})

        # --- Features (required) ---
        index_path = Path(path, "index")
        if not index_path.exists():
            raise FileNotFoundError(f"Missing features index file: {index_path}")
        with open(index_path, "rt", encoding="ascii") as handle:
            features = json.load(handle)
        feature_dim = len(features)

        # --- Preferred: read exact architecture ---
        arch_path = Path(path, "model.arch.json")
        enc_layers: Optional[List[LayerInfo]] = None
        dec_layers: Optional[List[LayerInfo]] = None
        batch_norm = False

        if arch_path.exists():
            with open(arch_path, "rt", encoding="utf-8") as fh:
                arch = json.load(fh)
            latent_dim = int(arch["latent_dim"])
            enc_layers = [LayerInfo.from_dict(d) for d in arch.get("encoder_layers", [])]
            dec_layers = [LayerInfo.from_dict(d) for d in arch.get("decoder_layers", [])]
            batch_norm = bool(arch.get("batch_norm", False))
        else:
            # --- Fallback: infer latent_dim from encoder weights (never from names) ---
            enc_state_path = Path(path, "model.enc.pt")
            if not enc_state_path.exists():
                raise FileNotFoundError(f"Missing encoder weights: {enc_state_path}")
            enc_state = torch.load(enc_state_path, map_location=device)

            if "z_mean.weight" in enc_state:
                latent_dim = int(enc_state["z_mean.weight"].shape[0])
            elif "pathway.linear.weight" in enc_state:
                latent_dim = int(enc_state["pathway.linear.weight"].shape[0])
            else:
                two_d = [v.shape for k, v in enc_state.items()
                         if isinstance(v, torch.Tensor) and v.ndim == 2]
                if not two_d:
                    raise ValueError("Could not infer latent_dim from encoder state dict.")
                latent_dim = int(two_d[-1][0])  # prefer last 2D weight (closest to heads)

        # --- Latent naming (names only; ignore if size mismatches) ---
        groups_path = Path(path, "net.groups.tsv")
        latent_idx_path = Path(path, "latent.index")
        groups: Optional[Dict[str, List[str]]] = None
        latent_index: Optional[List[str]] = None

        if groups_path.exists():
            latent_index = []
            groups = {}
            with open(groups_path, "rt", encoding="ascii") as fh:
                for line in fh:
                    row = line.rstrip().split("\t")
                    if not row or row[0] == "":
                        continue
                    latent_index.append(row[0])
                    groups[row[0]] = row[1:]
            if len(latent_index) != int(latent_dim):
                logger.warning("Ignoring net.groups.tsv because size != latent_dim (%d != %d)",
                               len(latent_index), latent_dim)
                latent_index, groups = None, None

        elif latent_idx_path.exists():
            with open(latent_idx_path, "rt", encoding="ascii") as fh:
                li = [ln.strip() for ln in fh if ln.strip()]
            if len(li) == int(latent_dim):
                latent_index = li
            else:
                logger.warning("Ignoring latent.index because size != latent_dim (%d != %d)",
                               len(li), latent_dim)
                latent_index = None

        # --- Build EXACT modules and load weights (strict) ---
        enc = BaseVAE.build_encoder(feature_dim, latent_dim, layers=enc_layers, batch_norm=batch_norm)
        dec = BaseVAE.build_decoder(feature_dim, latent_dim, layers=dec_layers, batch_norm=False)

        enc.load_state_dict(torch.load(Path(path, "model.enc.pt"), map_location=device), strict=True)
        dec.load_state_dict(torch.load(Path(path, "model.dec.pt"), map_location=device), strict=True)
        enc.to(device)
        dec.to(device)

        # --- Instantiate the container ---
        try:
            out = model_cls(features, encoder=enc, decoder=dec, **model_kwargs)  # type: ignore
        except TypeError as e:
            raise TypeError(
                f"{getattr(model_cls, '__name__', str(model_cls))} could not be constructed with "
                f"(features, encoder=..., decoder=..., **model_kwargs). Error: {e}"
            ) from e

        # Optional fields if supported by container
        if hasattr(out, "latent_groups"):
            out.latent_groups = groups  # type: ignore[attr-defined]
        if hasattr(out, "latent_index"):
            out.latent_index = latent_index  # type: ignore[attr-defined]

        stats_path = Path(path, "training.stats.tsv")
        if stats_path.exists():
            try:
                if hasattr(out, "normal_stats"):
                    out.normal_stats = pd.read_csv(stats_path, sep="\t", index_col=0)  # type: ignore[attr-defined]
            except Exception:
                logger.warning(f"Could not read {stats_path}")

        return out

    def __init__(self, features: List[str], encoder: Optional[Encoder] = None, decoder: Optional[Decoder] = None,
                 **kwargs):
        super().__init__()
        self.features = list(features)
        self.encoder: Optional[Encoder] = encoder
        self.decoder: Optional[Decoder] = decoder
        self.extra_args = kwargs  # for subclasses to stash configs

    def forward(self, x: torch.Tensor):
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("VAE encoder/decoder not initialized.")
        mu, logvar, z = self.encoder(x)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, torch.Tensor, torch.utils.data.DataLoader], **kwargs):
        raise NotImplementedError("Subclasses must implement fit().")

    def save(self, path: str, normal_df: Optional[pd.DataFrame] = None):
        """Save VAE model with associated elements."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        # Weights
        torch.save(self.encoder.state_dict(), p / "model.enc.pt")  # type: ignore[arg-type]
        torch.save(self.decoder.state_dict(), p / "model.dec.pt")  # type: ignore[arg-type]

        # Features
        (p / "index").write_text(json.dumps(list(self.features)), encoding="ascii")

        # Optional latent names/groups
        if getattr(self, "latent_index", None) is not None and getattr(self, "latent_groups", None) is not None:
            with open(p / "net.groups.tsv", "wt", encoding="ascii") as fh:
                for g in self.latent_index:  # type: ignore[attr-defined]
                    fh.write("\t".join([str(g), *self.latent_groups.get(g, [])]) + "\n")  # type: ignore[attr-defined]

        # History
        (p / "stats.json").write_text(
            json.dumps({"loss_history": getattr(self, "history", {}).get("loss", [])}), encoding="ascii"
        )

        # Training stats (save provided df if given; else save internal normal_stats if present)
        if normal_df is not None:
            normal_df.to_csv(p / "training.stats.tsv", sep="\t")
        elif getattr(self, "normal_stats", None) is not None:
            self.normal_stats.to_csv(p / "training.stats.tsv", sep="\t")  # type: ignore[attr-defined]

        # --- Save architecture (authoritative for reload) ---
        try:
            latent_dim = (getattr(self.encoder, "latent_dim", None)  # type: ignore[attr-defined]
                          or getattr(self.decoder, "latent_dim", None))  # type: ignore[attr-defined]
            arch = {
                "feature_dim": len(self.features),
                "latent_dim": int(latent_dim) if latent_dim is not None else None,
                "encoder_layers": [li.to_dict() for li in getattr(self, "_encoder_layers_cfg", [])],
                "decoder_layers": [li.to_dict() for li in getattr(self, "_decoder_layers_cfg", [])],
                "batch_norm": bool(getattr(self, "_batch_norm", False)),
            }
            (p / "model.arch.json").write_text(json.dumps(arch, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not write model.arch.json: {e}")