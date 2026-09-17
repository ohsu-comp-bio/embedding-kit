# Commands · cbio

The `cbio` command group wraps cBioPortal access. It is a `click` group exposed via the CLI — see the full option reference in the [CLI Reference](../../cli.md#cbio).

| Sub-command | Purpose |
|-------------|---------|
| `embkit cbio studies` | List available cBioPortal studies (ID and name) |
| `embkit cbio download --study_id ID` | Download and unpack a study's molecular profile data |

## Python API

The underlying client is `embkit.resources.CBIOPortal`:

```python
from embkit.resources import CBIOPortal
from embkit.c_bio import CBIOAPI

# List studies
print([s["studyId"] for s in (CBIOAPI().list_studies() or [])])

# Download + unpack a specific study
portal = CBIOPortal(study_id="brca_tcga", save_path="data/cbio", download=True)
portal.download()
portal.unpack()
print(portal.unpacked_file_path)
```

::: embkit.c_bio.api.CBIOAPI
    options:
      show_source: false
      filters:
        - "!^_"
