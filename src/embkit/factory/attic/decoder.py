"""Factory decoder builder – constructs a Decoder from a spec.

Spec format mirrors the JSON used previously:

    {
        "type": "decoder",
        "params": {
            "latent_dim": 64,
            "feature_dim": 123,
            "layers": [ {layer dict}, ... ],
            "batch_norm": false,
            "default_activation": "relu",
            "device": null
        }
    }
"""

from typing import Any, Dict, List

from ..models.vae.decoder import Decoder
from ..layers.layer_info import LayerInfo


def build_decoder(spec: Dict[str, Any], device=None) -> Decoder:
    """Build a :class:`Decoder` from a JSON spec.

    Parameters
    ----------
    spec: dict
        Must contain a ``"params"`` dictionary with arguments required by
        :class:`embkit.models.vae.decoder.Decoder`.
    device: torch.device, optional
        Forwarded to the ``Decoder`` constructor.
    """
    params = spec.get("params", {})
    layers = params.get("layers")
    if layers is not None:
        layer_objs: List[LayerInfo] = []
        for li in layers:
            if isinstance(li, LayerInfo):
                layer_objs.append(li)
            else:
                layer_objs.append(LayerInfo.from_dict(li))
        params["layers"] = layer_objs
    return Decoder(**params, device=device)
