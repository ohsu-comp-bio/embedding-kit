# Preprocessing API

Normalization utilities for biological expression data, assay-specific count transforms (RNA, ADT, ATAC), stratified splitting and H&E stain normalization.

## ExpMinMaxScaler

::: embkit.preprocessing.normalize.ExpMinMaxScaler
    options:
      show_source: false
      members:
        - fit
        - transform
        - inverse_transform

## quantile_max_norm

::: embkit.preprocessing.normalize.quantile_max_norm
    options:
      show_source: false

## exp_max_norm

::: embkit.preprocessing.normalize.exp_max_norm
    options:
      show_source: false

## get_dataset_nonzero_mask

::: embkit.preprocessing.normalize.get_dataset_nonzero_mask
    options:
      show_source: false

## library_size_norm

::: embkit.preprocessing.transform.library_size_norm
    options:
      show_source: false

## clr_norm

::: embkit.preprocessing.transform.clr_norm
    options:
      show_source: false

## TfidfLSI

::: embkit.preprocessing.transform.TfidfLSI
    options:
      show_source: false
      members:
        - fit
        - transform

## split_by_label

::: embkit.preprocessing.split.split_by_label
    options:
      show_source: false

## MacenkoNormalizer

::: embkit.preprocessing.stain.MacenkoNormalizer
    options:
      show_source: false
      members:
        - estimate
        - fit
        - transform

## rgb_to_od / od_to_rgb

::: embkit.preprocessing.stain.rgb_to_od
    options:
      show_source: false

::: embkit.preprocessing.stain.od_to_rgb
    options:
      show_source: false
