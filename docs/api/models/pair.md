# PairPredictor

Predicts a scalar interaction score from a pair of items plus a context. Item and context embeddings are produced by provided encoder modules, concatenated, and passed through a small MLP.

```python
from embkit.models.pair import PairPredictor
from embkit.models.ffnn import FFNN

item_enc = FFNN(input_dim=100, output_dim=32)
ctx_enc  = FFNN(input_dim=200, output_dim=32)

model = PairPredictor(
    item_module=item_enc,
    context_module=ctx_enc,
    item_dim=32,
    context_dim=32,
    learning_dim=128,
)
score = model([item1, item2, context])   # (batch, 1)
```

## API

::: embkit.models.pair.PairPredictor
    options:
      show_source: false
      filters:
        - "!^_"
