"""Factory package – single source for building layers, encoders and decoders from JSON specs.

Exports:
    - build(spec, device=None, **extra) – dispatcher
    - Layer – layer description class (formerly LayerInfo)
    - build_encoder(spec, device=None)
    - build_decoder(spec, device=None)
    - get_activation – helper to map string names to torch modules
"""

from .core import build, save, load

from .mapping import Linear

from .layers import Layer, LayerList

from .registery import nn_module, nn_module_named, get_class_name
