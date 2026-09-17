# Pathway

Pathway and feature-group helpers used for pathway-aware model construction (e.g. [NetVAE](models/net_vae.md)) and for building the binary masks that `MaskedLinear` layers enforce.

Module path: `embkit.pathway`

These functions operate on a **feature map** — a dict of `{group: [feature, ...]}` — and the row/column indices of an expression matrix.

> **Status note** — the downstream NetVAE consumer is in maintenance mode; the team's next architecture is a mixture-of-experts model. The pathway/mask machinery itself is a generic building block and remains supported.

## Typical flow

```python
import pandas as pd
from embkit.pathway import (
    extract_sif_interactions,
    feature_map_link_filter,
    build_mask,
)

# 1) Parse a SIF pathway file into a feature map
feature_map = extract_sif_interactions("pathway.sif")

# 2) Drop groups that only have a handful of links
feature_map = feature_map_link_filter(feature_map, min_group_size=2)

# 3) Build a (groups x features) binary mask for a MaskedLinear layer
mask = build_mask(
    feature_map,
    src_index=list(df.columns),   # features
    dst_index=group_names,        # group nodes
)
```

## API

::: embkit.pathway
    options:
      show_source: false
      filters:
        - "!^_"
