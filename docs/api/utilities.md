# Utilities API

Dimensionality-reduction and clustering helpers that operate on the large-CSV
files produced by the embedding pipeline. Both read a TSV file of embeddings via
`LargeCsvReader`, keep the row labels, and write results with the labels
restored. Clustering and PCA are implemented with scikit-learn.

## run_kmeans

::: embkit.utilities.kmeans.run_kmeans
    options:
      show_source: false

## run_pca

::: embkit.utilities.pca.run_pca
    options:
      show_source: false
