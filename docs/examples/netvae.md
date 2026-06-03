# NetVAE Example

This example shows how to train a pathway-constrained VAE (`NetVAE`) on gene expression data. Instead of learning a fully unconstrained latent space, each latent dimension corresponds to a biological pathway, constraining which input features contribute to each latent node.

## Prerequisites

- `embkit` installed
- A tab-separated gene expression matrix (rows = samples, columns = genes)
- A pathway file in **SIF** (Simple Interaction Format): three columns `source \t interaction_type \t target`

A minimal SIF file looks like:

```
TP53    controls-expression-of   CDKN1A
MYC     controls-expression-of   CDK4
MYC     controls-expression-of   E2F1
BRCA1   controls-expression-of   CCND1
```

---

## CLI usage

The simplest way to train a NetVAE is via the CLI:

```bash
embkit model train-netvae rna.tsv pathway.sif \
    --epochs 80 \
    --normalize expMinMax \
    --out netvae.model
```

The command:
1. Reads and normalizes the expression matrix
2. Parses the SIF file to extract feature→pathway group membership
3. Intersects the matrix features with the pathway genes
4. Builds encoder/decoder layers with `MaskedLinear` (connections not in the pathway are zeroed)
5. Trains and saves the model

### Optional: verify constraint integrity

After training, you can run an explicit integrity audit:

```bash
embkit model verify netvae.model --ci
```

For strict identity checks against expected model shape and features:

```bash
embkit model verify netvae.model \
    --strict \
    --expected-feature-count 15425 \
    --expected-latent-dim 2061 \
    --expected-features-file expected_features.txt \
    --fail-on-unhealthy
```

---

## Python API

### 1) Load and normalize data

```python
import pandas as pd
from embkit.preprocessing import ExpMinMaxScaler

df = pd.read_csv("rna.tsv", sep="\t", index_col=0)

norm = ExpMinMaxScaler()
norm.fit(df)
df_norm = pd.DataFrame(norm.transform(df), index=df.index, columns=df.columns)
```

### 2) Parse pathways and intersect with features

```python
from embkit.pathway import extract_sif_interactions, feature_map_intersect, build_feature_map_indices

feature_map = extract_sif_interactions("pathway.sif")
feature_map = feature_map_intersect(feature_map, df_norm.columns)
feature_idx, group_idx = build_feature_map_indices(feature_map)

# Keep only genes that appear in the pathway file
df_norm = df_norm[feature_idx]

group_count   = len(group_idx)
feature_count = len(feature_idx)
print(f"Features: {feature_count}, Pathway groups: {group_count}")
```

### 3) Build masked layers

Each `Layer` uses `op="masked_linear"` with `PathwayConstraintInfo` that describes the connection pattern:

```python
from embkit.factory.layers import Layer
from embkit.constraints import PathwayConstraintInfo
from embkit import dataframe_loader

# How many nodes per group at each encoder depth
gcounts = [5, 2, 1]

enc_layers = [
    Layer(group_count * gcounts[0], op="masked_linear",
          constraint=PathwayConstraintInfo("features-to-group", feature_map)),
    Layer(group_count * gcounts[1], op="masked_linear",
          constraint=PathwayConstraintInfo("group-to-group", feature_map,
                                           in_group_scaling=gcounts[0], out_group_scaling=gcounts[1])),
    Layer(group_count, op="masked_linear",
          constraint=PathwayConstraintInfo("group-to-group", feature_map,
                                           in_group_scaling=gcounts[1], out_group_scaling=gcounts[2])),
]

dec_layers = [
    Layer(group_count * gcounts[1], op="masked_linear",
          constraint=PathwayConstraintInfo("group-to-group", feature_map,
                                           in_group_scaling=gcounts[2], out_group_scaling=gcounts[1])),
    Layer(group_count * gcounts[0], op="masked_linear",
          constraint=PathwayConstraintInfo("group-to-group", feature_map,
                                           in_group_scaling=gcounts[1], out_group_scaling=gcounts[0])),
    Layer(feature_count, op="masked_linear",
          constraint=PathwayConstraintInfo("group-to-features", feature_map, in_group_scaling=gcounts[0]),
          activation="none"),
]
```

**Constraint types**

| Type | Description |
|------|-------------|
| `"features-to-group"` | Input features → pathway group nodes |
| `"group-to-group"` | Pathway group → same pathway group (depth-to-depth) |
| `"group-to-features"` | Pathway group nodes → output features |

### 4) Train

This Python API example uses the standard `VAE` class with pathway-constrained masked layers — not the `NetVAE` class. This gives you finer control over layer architecture (multiple depths per group). The `NetVAE` class wraps this pattern and builds the masked stack from `latent_groups` + `group_layer_size`.

```python
from embkit.models.vae.vae import VAE
from embkit.losses import bce_with_logits
from embkit import optimize

dataloader = dataframe_loader(df_norm, batch_size=256)

vae = VAE(
    features=list(df_norm.columns),
    latent_dim=group_count,
    encoder_layers=enc_layers,
    decoder_layers=dec_layers,
)

schedule = [(0.0, 20), (0.1, 20), (0.3, 40)]

optimize.fit_vae(
    vae,
    X=dataloader,
    beta_schedule=schedule,
    lr=1e-4,
    loss=bce_with_logits,
)
```

### 5) Save and encode

```python
from embkit.factory import save, load
from embkit import dataframe_tensor
import pandas as pd
import torch

save(vae, "netvae.model")

# Encode samples — each dimension corresponds to a pathway group
vae.eval()
with torch.no_grad():
    z = vae.encode(dataframe_tensor(df_norm))

embedding_df = pd.DataFrame(z.cpu().numpy(), index=df_norm.index)
embedding_df.to_csv("netvae_embeddings.tsv", sep="\t")
```

The embedding has one column per pathway group, making it directly interpretable in downstream analyses.

---

## Notes

- The intersection step (`feature_map_intersect`) intersects genes/features (columns) with the pathway file and drops genes that are not present there, subsetting the matrix columns accordingly. Check `isect` before training to see how many genes are retained.
- `gcounts` controls the number of hidden nodes per group at each depth. More nodes = more expressiveness per group but larger model.
- The `VAE` built above supports `factory.save` / `factory.load` serialization. The `NetVAE` class (used by the CLI) also supports serialization via `factory.save` / `factory.load`.
- Constraint class naming is canonicalized to `PathwayConstraintInfo`. The legacy alias `PathwayControlConstraint` has been removed.
