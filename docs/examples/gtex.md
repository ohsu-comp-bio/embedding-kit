# GTEx Example

This walkthrough shows one way to build embeddings from GTEx expression data using Embedding Kit.

## Prerequisites

- Python environment with `embkit` installed
- Enough memory for large expression matrices (GTEx files can be large)

## 1) Load GTEx data

```python
from embkit.resources import GTEx
from embkit.preprocessing import load_gct

gtex = GTEx()
gtex_df = load_gct(gtex)

print(gtex_df.shape)
print(gtex_df.head())
```

## 2) Prepare and normalize features

```python
from embkit.resources import Hugo
from embkit.preprocessing import load_raw_hugo, ExpMinMaxScaler
import pandas as pd

hugo = Hugo()
hugo_df = load_raw_hugo(hugo)

gene_names = (
    hugo_df[hugo_df["locus_group"] == "protein-coding gene"]
    [["symbol", "ensembl_gene_id"]]
    .set_index("ensembl_gene_id")["symbol"]
    .to_dict()
)

df_select = gtex_df.rename(columns=lambda x: gene_names.get(x.split(".")[0], x))
df_select = df_select.loc[:, df_select.columns.map(lambda x: not x.startswith("ENSG0"))]

norm = ExpMinMaxScaler()
norm.fit(df_select)
df_norm = pd.DataFrame(norm.transform(df_select), index=df_select.index, columns=df_select.columns)

print(df_norm.shape)
```

## 3) Train a VAE in Python

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
    Layer(128, activation="relu"),
]
decoder_layers = [
    Layer(1024, activation="relu"),
    Layer(2048, activation="relu"),
]

schedule = [(0.0, 20), (0.1, 20), (0.3, 40), (0.4, 40)]

vae = VAE(df_norm.columns, latent_dim=128, decoder_layers=decoder_layers, encoder_layers=encoder_layers)
optimize.fit_vae(vae, X=dataloader, beta_schedule=schedule, lr=1e-3, loss=bce_with_logits)
```

## 4) Reconstruct and run quick checks

```python
from embkit import dataframe_tensor
import torch
import numpy as np
import pandas as pd

vae.eval()
with torch.no_grad():
    x_t = dataframe_tensor(df_norm)
    recon_logits, mu, logvar, z = vae(x_t)
    recon = torch.sigmoid(recon_logits)

recon_df = pd.DataFrame(recon.cpu().numpy(), index=df_norm.index, columns=df_norm.columns)

gene = "OR4F5"
y = df_norm[gene].values
yhat = recon_df[gene].values

print("gene corr:", float(np.corrcoef(y, yhat)[0, 1]))
print("gene MAE:", float(np.mean(np.abs(y - yhat))))
print("global MAE:", float(np.mean(np.abs(recon_df.values - df_norm.values))))
```

## Optional: CLI-driven training

If your input is already in TSV matrix format, you can train directly from the CLI:

```bash
embkit model train-vae ./gtex.normalized.tsv --epochs 120 --latent 128 --out gtex.vae.model
embkit model encode ./gtex.normalized.tsv gtex.vae.model --out gtex.embedding.tsv
```
