# Encoding

Feature encoders used to turn raw biological features into numeric inputs for the models.

| Name | Purpose |
|------|---------|
| `ProteinEncoder` | Encode FASTA protein sequences with ESM2 (backs `embkit protein encode`) |
| `OneHotEncoder` | Map categorical values to one-hot vectors |
| `ProteinOneHotEncoder` | One-hot + positional encoding of amino-acid sequences |
| `PreEncoded` | Load pre-computed embeddings from disk |

```python
from embkit.encoding.protein import ProteinEncoder

enc = ProteinEncoder(model="t33", batch_size=100)
embeddings = enc.encode(["MKTLLL...", "MSDLS..."], output="mean-pool")
```

> Note: `ProteinEncoder` lives in the `embkit.encoding.protein` submodule (not re-exported from `embkit.encoding`).

For VCF-based variant-count features, see the genome helpers in `embkit.encoding.genome` (`vcf_to_dataframe`, `vectorize_variant_count`).

## API

::: embkit.encoding
    options:
      show_source: false
      filters:
        - "!^_"
