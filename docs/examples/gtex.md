# GTEx Example

This walkthrough shows how to build embeddings from GTEx gene expression data using Embedding Kit from download through training.

## Prerequisites

- Python environment with `embkit` installed
- Sufficient disk space (~1 GB) and memory for the GTEx matrix (17,000+ samples, 50,000+ genes)

## 1) Download GTEx data

`GTEx` is a `Resource` that downloads gene TPM data from the GTEx portal on first use and caches it under `~/.embkit/`.

```python
from embkit.resources import GTEx

# Downloads GTEx v10 gene TPM on first call; uses cached file afterwards
gtex = GTEx(data_type="gene_tpm")  # also accepts "transcript_tpm"
print(gtex)  # shows save path and download status
```

The file is a gzip-compressed GCT format. Load it with pandas:

```python
import pandas as pd

# GCT files have two header rows to skip; the gene id is in column 0
gtex_df = pd.read_csv(str(gtex), sep="\t", index_col=0, skiprows=2)

# Drop the Description column if present
if "Description" in gtex_df.columns:
    gtex_df = gtex_df.drop(columns=["Description"])

# Transpose so rows = samples, columns = genes (Ensembl IDs)
gtex_df = gtex_df.T

print(gtex_df.shape)  # (n_samples, n_genes)
print(gtex_df.head())
```

## 2) Map Ensembl IDs to gene symbols and filter to protein-coding genes

```python
from embkit.resources import Hugo
import pandas as pd

# Downloads HUGO gene nomenclature file on first call
hugo = Hugo()
hugo_df = pd.read_csv(str(hugo), sep="\t", low_memory=False)

gene_names = (
    hugo_df[hugo_df["locus_group"] == "protein-coding gene"]
    [["symbol", "ensembl_gene_id"]]
    .dropna()
    .set_index("ensembl_gene_id")["symbol"]
    .to_dict()
)

# Strip version suffix from Ensembl IDs (e.g. ENSG00000000003.14 -> ENSG00000000003)
gtex_df.columns = [c.split(".")[0] for c in gtex_df.columns]

# Keep only protein-coding genes with a known symbol
gtex_df = gtex_df.rename(columns=gene_names)
gtex_df = gtex_df.loc[:, ~gtex_df.columns.str.startswith("ENSG")]

print(f"Protein-coding features: {gtex_df.shape[1]}")
```

## 3) Normalize

`ExpMinMaxScaler` applies a log2(x+1) transform then min-max scales to [0, 1]. This is appropriate for TPM data where values span several orders of magnitude.

```python
from embkit.preprocessing import ExpMinMaxScaler
import pandas as pd

norm = ExpMinMaxScaler()
norm.fit(gtex_df)
df_norm = pd.DataFrame(
    norm.transform(gtex_df),
    index=gtex_df.index,
    columns=gtex_df.columns,
)

print(df_norm.min().min(), df_norm.max().max())  # should be ~0.0 and ~1.0
```

## 4) Train a VAE in Python

```python
from embkit import dataframe_loader
from embkit.losses import bce_with_logits
from embkit.models.vae import VAE
from embkit.factory.layers import Layer
from embkit import optimize

dataloader = dataframe_loader(df_norm, batch_size=512)

encoder_layers = [
    Layer(2048, activation="relu"),
    Layer(1024, activation="relu"),
    Layer(512,  activation="relu"),
]
decoder_layers = [
    Layer(1024, activation="relu"),
    Layer(2048, activation="relu"),
]

# KL annealing schedule: (beta, n_epochs)
schedule = [(0.0, 20), (0.1, 20), (0.3, 40), (0.4, 40)]

vae = VAE(
    features=list(df_norm.columns),
    latent_dim=128,
    encoder_layers=encoder_layers,
    decoder_layers=decoder_layers,
)

optimize.fit_vae(vae, X=dataloader, beta_schedule=schedule, lr=1e-3, loss=bce_with_logits)
```

Progress bars show per-epoch loss, reconstruction loss, KL divergence, and the current beta value.

## 5) Save and reload the model

```python
from embkit.factory import save, load

save(vae, "gtex.vae.model")

# Reload on any machine — architecture is stored inside the file
vae2 = load("gtex.vae.model")
print(vae2.latent_dim)  # 128
```

## 6) Encode samples into latent space

```python
from embkit import dataframe_tensor
import torch
import pandas as pd

vae.eval()
with torch.no_grad():
    z = vae.encode(dataframe_tensor(df_norm))

embedding_df = pd.DataFrame(
    z.cpu().numpy(),
    index=df_norm.index,
)
print(embedding_df.shape)  # (n_samples, 128)
embedding_df.to_csv("gtex.embedding.tsv", sep="\t")
```

## 7) Quick reconstruction check

```python
import numpy as np

vae.eval()
with torch.no_grad():
    recon_logits, mu, logvar, z = vae(dataframe_tensor(df_norm))
    recon = torch.sigmoid(recon_logits)

recon_df = pd.DataFrame(recon.cpu().numpy(), index=df_norm.index, columns=df_norm.columns)

gene = df_norm.columns[0]
y    = df_norm[gene].values
yhat = recon_df[gene].values
print("gene corr:", float(np.corrcoef(y, yhat)[0, 1]))
print("global MAE:", float(np.mean(np.abs(recon_df.values - df_norm.values))))
```

## Optional: CLI-driven training

If your data is already in TSV format (rows = samples, columns = features), you can skip Python entirely:

```bash
# Normalize first
embkit matrix normalize gtex.raw.tsv --out gtex.normalized.tsv

# Train
embkit model train-vae gtex.normalized.tsv \
    --epochs 120 \
    --latent 128 \
    --encode-layers 2048,1024,512 \
    --decode-layers 1024,2048 \
    --loss bce-logit \
    --schedule "20:0,20:0.1,40:0.3,40:0.4" \
    --out gtex.vae.model

# Encode
embkit model encode gtex.normalized.tsv gtex.vae.model --out gtex.embedding.tsv
```
