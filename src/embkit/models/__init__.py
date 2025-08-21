from typing import TYPE_CHECKING

if TYPE_CHECKING: # pragma: no cover
    from .vae_model.vae import VAE as _VAE  # pragma: no cover
    from .vae_model.net_vae import NetVae as _NetVae  # pragma: no cover

def __getattr__(name: str):  # pragma: no cover
    if name == "VAE":  # pragma: no cover
        from .vae_model.vae import VAE  # pragma: no cover
        return VAE  # pragma: no cover
    elif name == "NetVae":  # pragma: no cover
        from .vae_model.net_vae import NetVae  # pragma: no cover
        return NetVae  # pragma: no cover
    raise AttributeError(f"module {__name__} has no attribute {name}")  # pragma: no cover