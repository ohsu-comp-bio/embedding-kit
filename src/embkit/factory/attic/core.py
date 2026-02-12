"""Factory core – dispatcher for building objects from JSON specs.

Supported spec types:
- "layer": builds a single ``Layer`` and returns the instantiated ``nn.Module``.
- "encoder": builds an ``Encoder`` model.
- "decoder": builds a ``Decoder`` model.
"""

from typing import Any, Dict
from .layer_info import LayerInfo

def build(spec: Dict[str, Any], device=None, **extra) -> Any:
    """Build an object from a specification dictionary.

    Parameters
    ----------
    spec: dict
        Must contain a ``"type"`` key indicating what to build and a ``"params"``
        dictionary with the constructor arguments.
    device: torch.device, optional
        Passed through to the underlying builders.
    extra: dict
        Additional keyword arguments forwarded to the builder (currently unused).
    """
    typ = spec.get("type")
    if not typ:
        raise ValueError("Spec must contain a 'type' field")
    params = spec.get("params", {})

    if typ == "layer":
        from .layer import Layer

        # ``in_features`` must be provided in the params for a Layer
        layer = Layer(**params)
        return layer.gen_layer(device=device)
    elif typ == "encoder":
        from .encoder import build_encoder

        return build_encoder(spec, device=device)
    elif typ == "decoder":
        from .decoder import build_decoder

        return build_decoder(spec, device=device)
    else:
        raise ValueError(f"Unsupported spec type: {typ!r}")


def build_layers(sizes, activation="relu", end_activation="relu"):
    """
    Simple layer builder

    end_activation: if the last layer it an output, allow activation to be set to None
    """
    out = []
    for i, s in enumerate(sizes):
        if i < len(sizes) - 1:
            l = LayerInfo(s, activation=activation)
        else:
            # print(f"end activation{end_activation}")
            l = LayerInfo(s, activation=end_activation)
        out.append(l)
    return out
