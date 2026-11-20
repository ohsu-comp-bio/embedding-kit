# Datasets API

Hashable dataset adapters for large membership testing.

`HashedDataset` is a `torch.utils.data.Dataset` that exposes an abstract
`hash_item` hook so items can be reduced to a hash. `DatasetBloomFilter` builds
(or loads) a Bloom filter over those hashes, giving fast approximate membership
tests (`item in dataset`) for very large datasets that do not fit in memory. The
filter is backed by `rbloom`, which supports saving and reloading it to disk.

## DatasetBloomFilter

::: embkit.datasets.hashed.DatasetBloomFilter
    options:
      show_source: false
      members:
        - __init__
        - has
        - save_bloom
      merge_init_into_class: true

## HashedDataset

::: embkit.datasets.hashed.HashedDataset
    options:
      show_source: false
      filters:
        - "!^_"

## hash_sha256

::: embkit.datasets.hashed.hash_sha256
    options:
      show_source: false
