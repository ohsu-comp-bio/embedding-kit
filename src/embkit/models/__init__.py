from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vae_model.vae import VAE as _VAE
    from .vae_model.net_vae import NetVae as _NetVae

def __getattr__(name: str):
    if name == "VAE":
        from .vae_model.vae import VAE
        return VAE
    elif name == "NetVae":
        from .vae_model.net_vae import NetVae
        return NetVae
    raise AttributeError(f"module {__name__} has no attribute {name}")