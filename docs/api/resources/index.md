# Resources

Downloaders for external biological datasets. Each class downloads (and, where needed, unpacks) a dataset into a local folder and exposes the resulting file path. Backs the `embkit resources` and `embkit cbio` CLI commands.

| Class | Dataset |
|-------|---------|
| `GTEx` | GTEx RNA-seq (gene TPM or transcript TPM) |
| `Hugo` | HGNC / HUGO gene symbols |
| `SIF` | SIF (Simple Interaction Format) pathway file |
| `CBIOPortal` | cBioPortal molecular profiles (per study) |

```python
from embkit.resources import GTEx, Hugo, SIF, CBIOPortal

gtex = GTEx(data_type="gene_tpm", save_path="data/gtex", download=True)
hugo = Hugo(save_path="data/hugo", download=True)
sif  = SIF(save_path="data/pathway", download=True)

portal = CBIOPortal(study_id="brca_tcga", save_path="data/cbio", download=True)
portal.download()
portal.unpack()
print(portal.unpacked_file_path)
```

## API

::: embkit.resources
    options:
      show_source: false
      filters:
        - "!^_"
