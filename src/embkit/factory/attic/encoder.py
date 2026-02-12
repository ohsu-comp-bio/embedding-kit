"""Factory encoder builder – constructs an Encoder from a spec.

The spec format mirrors the previous JSON format used throughout the codebase:

    {
        "type": "encoder",
        "params": {
            "feature_dim": 123,
            "latent_dim": 64,
            "layers": [ {layer dict}, ... ],
            "batch_norm": false,
            "default_activation": "relu",
            "make_latent_heads": true,
            "sampling": false,
            "constraint": null,
            "device": null
        }
    }
"""

from typing import Any, Dict, List

from ..models.vae.encoder import Encoder
from ..layers.layer_info import LayerInfo


def build_encoder(spec: Dict[str, Any], device=None) -> Encoder:
    """Build an :class:`Encoder` from a JSON spec.

    Parameters
    ----------
    spec: dict
        Must contain a ``"params"`` dict with the arguments required by
        :class:`embkit.models.vae.encoder.Encoder`.
    device: torch.device, optional
        Passed through to the underlying ``Encoder``.
    """
    params = spec.get("params", {})
    # Convert layer dicts to LayerInfo objects if needed
    layers = params.get("layers")
    if layers is not None:
        layer_objs: List[LayerInfo] = []
        for li in layers:
            # Allow dicts or already‑instantiated LayerInfo objects
            if isinstance(li, LayerInfo):
                layer_objs.append(li)
            else:
                layer_objs.append(LayerInfo.from_dict(li))
        params["layers"] = layer_objs
    return Encoder(**params, device=device)
