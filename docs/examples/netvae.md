# NetVAE Example

This example shows how to train a pathway-constrained VAE (`NetVAE`) on gene expression data. Instead of learning a fully unconstrained latent space, each latent dimension corresponds to a biological pathway, constraining which input features contribute to each latent node.

## Prerequisites

- `embkit` installed
- A tab-separated gene expression matrix (rows = samples, columns = genes)
- A pathway file in **SIF** (Simple Interaction Format): three columns `source \t interaction_type \t target`

A minimal SIF file looks like:

```
TP53    activates   CDKN1A
MYC     activates   CDK4
MYC     activates   E2F1
BRCA1   inhibits    CCND1
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
from embkit.pathway import extract_pathway_interactions, feature_map_intersect, FeatureGroups

feature_map = extract_pathway_interactions("pathway.sif")
feature_map, isect = feature_map_intersect(feature_map, df_norm.columns)

# Keep only genes that appear in the pathway file
df_norm = df_norm[isect]

fmap = FeatureGroups(feature_map)
group_count   = len(fmap)
feature_count = len(isect)
print(f"Features: {feature_count}, Pathway groups: {group_count}")
```

### 3) Build masked layers

Each `Layer` uses `op="masked_linear"` with a `ConstraintInfo` that describes the connection pattern:

```python
from embkit.factory.layers import Layer, LayerList, ConstraintInfo
from embkit import dataframe_loader

# How many nodes per group at each encoder depth
gcounts = [5, 2, 1]

enc_layers = [
    Layer(group_count * gcounts[0], op="masked_linear",
          constraint=ConstraintInfo("features-to-group", fmap, out_group_count=gcounts[0])),
    Layer(group_count * gcounts[1], op="masked_linear",
          constraint=ConstraintInfo("group-to-group", fmap,
                                    in_group_count=gcounts[0], out_group_count=gcounts[1])),
    Layer(group_count, op="masked_linear",
          constraint=ConstraintInfo("group-to-group", fmap,
                                    in_group_count=gcounts[1], out_group_count=gcounts[2])),
]

dec_layers = [
    Layer(group_count * gcounts[1], op="masked_linear",
          constraint=ConstraintInfo("group-to-group", fmap,
                                    in_group_count=gcounts[2], out_group_count=gcounts[1])),
    Layer(group_count * gcounts[0], op="masked_linear",
          constraint=ConstraintInfo("group-to-group", fmap,
                                    in_group_count=gcounts[1], out_group_count=gcounts[0])),
    Layer(feature_count, op="masked_linear",
          constraint=ConstraintInfo("group-to-features", fmap, in_group_count=gcounts[0]),
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

- The intersection step (`feature_map_intersect`) silently drops samples whose genes are absent from the pathway file. Check `isect` before training to see how many genes are retained.
- `gcounts` controls the number of hidden nodes per group at each depth. More nodes = more expressiveness per group but larger model.
- NetVAE does not currently support `factory.save` / `factory.load` serialization (the masked layer configuration is not yet registered with the factory). Save the model's `state_dict` manually if needed, and rebuild the architecture before loading weights.
