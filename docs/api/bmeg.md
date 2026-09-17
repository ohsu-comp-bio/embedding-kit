# BMEG

Utilities for interacting with BMEG-backed resources in Embedding Kit.

Module path: `embkit.bmeg`

> **Note** — this is a Python API only; there is no `embkit` CLI command for BMEG. The `cbio` CLI commands are a separate feature (see [cBioPortal](commands/cbio.md)).

The BMEG helpers depend on the optional `gripql` package.

## connect

Open a BMEG graph connection.

```python
from embkit.bmeg import connect

conn = connect(
    url="https://bmeg.io/grip",
    graph="rc6",
    cred_file="./bmeg_credentials.json",
)
```

## get_gene_map

Build an Ensembl Gene ID → Hugo Symbol mapping from a connected graph.

```python
from embkit.bmeg import connect, get_gene_map

gene_map = get_gene_map(connect(cred_file="./bmeg_credentials.json"))
```

## API

::: embkit.bmeg
    options:
      show_source: false
      filters:
        - "!^_"
