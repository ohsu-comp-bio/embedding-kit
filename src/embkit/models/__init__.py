
"""

This module provides access to various models used in the embkit package.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING: # pragma: no cover
    from .vae.vae import VAE as _VAE  # pragma: no cover
    from .vae.net_vae import NetVAE as _NetVAE  # pragma: no cover

def __getattr__(name: str):  # pragma: no cover
    if name == "VAE":  # pragma: no cover
        from .vae.vae import VAE  # pragma: no cover
        return VAE  # pragma: no cover
    elif name == "NetVAE":  # pragma: no cover
        from .vae.net_vae import NetVAE  # pragma: no cover
        return NetVAE  # pragma: no cover
    raise AttributeError(f"module {__name__} has no attribute {name}")  # pragma: no cover