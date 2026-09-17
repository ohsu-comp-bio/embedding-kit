# Utilities

Standalone, file-oriented utilities for large matrices (back the `embkit matrix pca` command and clustering workflows).

| Name | Purpose |
|------|---------|
| `run_pca` | Reduce a TSV matrix to N principal components, written to a new TSV |
| `run_kmeans` | K-means clustering over a TSV matrix with chunked input for large data |

```python
from embkit.utilities import run_pca, run_kmeans

pca_df = run_pca("rna.tsv", pca_size=50, output_file="rna.pca50.tsv")

centroids = run_kmeans("embeddings.tsv", k=8, output_file="centroids.tsv")
```

## API

::: embkit.utilities
    options:
      show_source: false
      filters:
        - "!^_"
