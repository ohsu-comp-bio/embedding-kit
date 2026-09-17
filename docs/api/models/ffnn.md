# FFNN

A multi-layer feed-forward neural network for supervised regression/classification tasks. Built from a `LayerList` config and serialized through the factory, so it saves/loads just like the VAEs.

```python
from embkit.models.ffnn import FFNN
from embkit.factory.layers import Layer, LayerList
from embkit import optimize

net = FFNN(input_dim=100, output_dim=1, layers=LayerList([64, 32]))
optimize.fit(net, X=X, y=y, epochs=50, lr=1e-3)
```

## API

::: embkit.models.ffnn.FFNN
    options:
      show_source: false
      filters:
        - "!^_"
