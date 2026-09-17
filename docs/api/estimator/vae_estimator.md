# Estimator · VAE

`VAEEstimator` is a scikit-learn style wrapper around [BaseVAE](../models/vae.md) for quick experiments:

```python
import pandas as pd
from embkit.estimator import VAEEstimator

est = VAEEstimator(latent_dim=32, epochs=20, batch_size=64, device="cpu")
est.fit(df_norm)            # trains a BaseVAE and stores it on .model

score = est.score(df_norm)  # negative MSE of the reconstruction
vae = est.model             # the underlying BaseVAE — use it for encode(), saving, etc.
embeddings = vae.encode(torch_tensor)
```

`est.history` holds the per-epoch training loss.

## API

::: embkit.estimator
    options:
      show_source: false
      filters:
        - "!^_"
