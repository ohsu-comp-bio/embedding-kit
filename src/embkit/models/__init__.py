
"""
Models
"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    # Only for type checkers; doesn't run at runtime
    from .vae_model.vae import VAE as _VAE
    from .vae_model.net_vae import NetVae as _NetVae

def __getattr__(name):
    if name == "VAE":
        from .vae_model.vae import VAE
        return VAE
    if name == "NetVae":
        from .vae_model.net_vae import NetVae
        return NetVae
    raise AttributeError(name)