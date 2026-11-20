# Convolutional Backbone and Decoder

`ConvBackbone` plugs into `VAEEncoder` as its `backbone`; `ConvDecoder` maps latents back to flat image tiles or 1-D signals.

Import from `embkit.modules.conv` (it registers both classes with the factory, so import it before `factory.load` of a saved conv model).

::: embkit.modules.conv
    options:
      show_source: false
      filters:
        - "!^_"
