"""
NetVAE implementation
"""
import logging

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import torch
import numpy as np
from ...modules import MaskedLinear

from .base_vae import BaseVAE
from ... import factory
from ...pathway import build_feature_map_indices
from ...constraints import PathwayConstraintInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# NetVae (training with optional alternating constraint)
# ---------------------------------------------------------

@factory.nn_module
class NetVAE(BaseVAE):
    """
    NetVAE

    A VAE model with group based constraint. Designed to work with 
    transcription factor network groups. All elements controlled by a common
    transcription factor a pooled into a single embedding variable. All other connections
    in from the input layer are forced to be zero
    """

    def __init__(
            self,
            features: List[str],
            latent_groups: Dict[str, List[str]],
            latent_index: Optional[List[str]] = None,
            group_layer_size: Optional[List[int]] = None,
            batch_norm: bool = False,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        if not latent_groups:
            raise ValueError("latent_groups cannot be empty for NetVAE.")

        if group_layer_size is None:
            group_layer_size = [1, 1]
        group_layer_size = [int(v) for v in group_layer_size]
        if any(v <= 0 for v in group_layer_size):
            raise ValueError(f"group_layer_size must contain positive integers, got {group_layer_size}.")

        if latent_index is None:
            _, group_idx = build_feature_map_indices(latent_groups)
            latent_index = list(group_idx)
        else:
            latent_index = list(latent_index)
        if len(latent_index) == 0:
            raise ValueError("latent_index cannot be empty.")

        feature_list = list(features)
        latent_size = len(latent_index)

        # Encoder: features -> groups*s0 -> groups*s1 -> ... -> groups*sN
        enc_layers = [
            factory.Layer(
                units=latent_size * group_layer_size[0],
                op="masked_linear",
                constraint=PathwayConstraintInfo(
                    "features-to-group",
                    feature_map=latent_groups,
                    feature_index=feature_list,
                    group_index=latent_index,
                    out_group_scaling=group_layer_size[0],
                ),
            )
        ]
        for in_scale, out_scale in zip(group_layer_size[:-1], group_layer_size[1:]):
            enc_layers.append(
                factory.Layer(
                    units=latent_size * out_scale,
                    op="masked_linear",
                    constraint=PathwayConstraintInfo(
                        "group-to-group",
                        feature_map=latent_groups,
                        feature_index=feature_list,
                        group_index=latent_index,
                        in_group_scaling=in_scale,
                        out_group_scaling=out_scale,
                    ),
                )
            )

        # Decoder: groups*sN -> ... -> groups*s1 -> groups*s0 -> features
        dec_layers = []
        for in_scale, out_scale in zip(reversed(group_layer_size[1:]), reversed(group_layer_size[:-1])):
            dec_layers.append(
                factory.Layer(
                    units=latent_size * out_scale,
                    op="masked_linear",
                    constraint=PathwayConstraintInfo(
                        "group-to-group",
                        feature_map=latent_groups,
                        feature_index=feature_list,
                        group_index=latent_index,
                        in_group_scaling=in_scale,
                        out_group_scaling=out_scale,
                    ),
                )
            )
        dec_layers.append(
            factory.Layer(
                units=len(feature_list),
                op="masked_linear",
                constraint=PathwayConstraintInfo(
                    "group-to-features",
                    feature_map=latent_groups,
                    feature_index=feature_list,
                    group_index=latent_index,
                    in_group_scaling=group_layer_size[0],
                ),
                activation="none",
            )
        )

        encoder = self.build_encoder(
            feature_dim=len(feature_list),
            latent_dim=latent_size,
            layers=factory.LayerList(enc_layers),
            batch_norm=batch_norm,
            device=device,
            dtype=dtype,
        )
        decoder = self.build_decoder(
            feature_dim=len(feature_list),
            latent_dim=latent_size,
            layers=factory.LayerList(dec_layers),
            device=device,
            dtype=dtype,
        )

        super().__init__(features=feature_list, encoder=encoder, decoder=decoder)
        self.latent_groups: Dict[str, List[str]] = latent_groups
        self.latent_index: List[str] = latent_index
        self.group_layer_size: List[int] = list(group_layer_size)
        self.history: Dict[str, List[float]] = {}
        self.normal_stats: Optional[pd.DataFrame] = None

    def _iter_pathway_constraints(self):
        modules = []
        if self.encoder is not None:
            modules.extend(self.encoder.net)
        if self.decoder is not None:
            modules.extend(self.decoder.net)
        for module in modules:
            if isinstance(module, MaskedLinear):
                constraint_info = getattr(module, "constraint_info", None)
                if constraint_info is not None:
                    yield module, constraint_info

    def set_constraint_active(self, active: bool) -> None:
        for _, constraint_info in self._iter_pathway_constraints():
            if hasattr(constraint_info, "set_active"):
                constraint_info.set_active(active)

    def update_membership(self, latent_groups: Dict[str, List[str]]) -> None:
        self.latent_groups = latent_groups
        for _, constraint_info in self._iter_pathway_constraints():
            if hasattr(constraint_info, "update_membership"):
                constraint_info.update_membership(latent_groups)

    def refresh_masks(self, device: Optional[torch.device] = None) -> None:
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
                
        if self.encoder is not None:
            self.encoder.refresh_mask(device)
        if self.decoder is not None:
            for module in self.decoder.net:
                if isinstance(module, MaskedLinear):
                    constraint_info = getattr(module, "constraint_info", None)
                    if constraint_info is not None:
                        m = constraint_info.gen_mask(module.linear.in_features, module.linear.out_features)
                        module.set_mask(torch.as_tensor(m, dtype=module.mask.dtype, device=device))
        
        # Ensure underlying weights are also zeroed
        with torch.no_grad():
            for module, _ in self._iter_pathway_constraints():
                module.linear.weight.mul_(module.mask)

    def get_constraint_projection_weights(self) -> Optional[np.ndarray]:
        if self.encoder is None:
            return None
        for module in self.encoder.net:
            if isinstance(module, MaskedLinear):
                return module.linear.weight.detach().cpu().numpy()
        return None

    def verify_integrity(self) -> Dict[str, Any]:
        """
        In NetVAE, perform a layer-by-layer audit of the effective weights
        to ensure sparsity constraints are being enforced.
        Also perform a leakage test to ensure weights outside the
        mask are strictly zero.
        """
        # Snapshot raw constrained weights before parent checks. Parent deep checks
        # run forwards that enforce masks in-place, which would otherwise hide leakage.
        raw_snapshots = []
        for module, constraint_info in self._iter_pathway_constraints():
            raw_snapshots.append(
                (
                    module,
                    constraint_info,
                    module.linear.weight.detach().clone(),
                    module.mask.detach().clone(),
                )
            )

        report = super().verify_integrity()

        layer_audit = []
        for module, constraint_info, weights, mask in raw_snapshots:

            # Leakage test
            # Check weights that should be masked out (weights * (1-mask))
            leakage_weights = weights * (1.0 - mask)
            leakage_sum = float(torch.sum(torch.abs(leakage_weights)))

            # Effective weights (inside mask)
            effective_weights = weights * mask

            num_params = weights.numel()
            raw_nonzero = int(torch.count_nonzero(weights))
            eff_nonzero = int(torch.count_nonzero(effective_weights))

            is_healthy = (eff_nonzero > 0)
            if leakage_sum > 1e-7: # Numerical tolerance for float32
                is_healthy = False
                report["healthy"] = False
                report["issues"].append(
                    f"Weight leakage detected in pathway layer '{constraint_info.op}' "
                    f"(leakage_sum={leakage_sum:.2e}). Weight updates bypassed the mask."
                )

            layer_audit.append({
                "layer": constraint_info.op,
                "in_features": int(module.linear.in_features),
                "out_features": int(module.linear.out_features),
                "raw_sparsity": 1.0 - (raw_nonzero / num_params),
                "effective_sparsity": 1.0 - (eff_nonzero / num_params),
                "leakage_sum": leakage_sum,
                "is_healthy": is_healthy
            })

            if eff_nonzero == 0:
                report["healthy"] = False
                report["issues"].append(f"Zero effective weights in pathway layer: {constraint_info.op}")

        report["sparsity_audit"] = layer_audit
        return report

    def to_dict(self) -> Dict[str, Any]:
        return {
            "features": self.features,
            "latent_groups": self.latent_groups,
            "latent_index": self.latent_index,
            "group_layer_size": self.group_layer_size,
            "history": getattr(self, "history", {}) or {}
        }

    @classmethod
    def from_dict(cls, d):
        if "group_layer_scaling" in d and "group_layer_size" not in d:
            raise ValueError(
                "NetVAE config uses deprecated key 'group_layer_scaling'. "
                "Use 'group_layer_size' instead."
            )

        features = d.get("features")
        if features is None:
            fmap = d.get("latent_groups") or {}
            feature_set = set()
            for members in fmap.values():
                feature_set.update(members)
            features = sorted(feature_set)

        model = NetVAE(
            features=features,
            latent_groups=d.get("latent_groups"),
            latent_index=d.get("latent_index"),
            group_layer_size=d.get("group_layer_size"),
        )
        model.history = d.get("history") or {}
        return model
