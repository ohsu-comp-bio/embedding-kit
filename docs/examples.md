

## Loading GTEx data

```python
from embkit.datasets import GTEx
from embkit.preprocessing import load_gct

gtex_df = load_gct(g.unpacked_file_path)
```




## GTEx embedding

```python
from embkit import dataframe_loader, dataframe_tensor
from embkit.preprocessing import load_gct
from embkit.datasets import GTEx, Hugo
from embkit.models.vae import VAE
from embkit.preprocessing import ExpMinMaxScaler
from embkit import bmeg

import pandas as pd
import numpy as np

g=GTEx()
hugo = Hugo()

hugo_df = pd.read_csv(hugo.unpacked_file_path, sep="\t", index_col=0)
df = load_gct( g.unpacked_file_path )

hugo_df["ensembl_gene_id"]
# select protein coding genes and build a dict that translates from ensembl gene id to Hugo name
gene_names = hugo_df[ hugo_df["locus_group"] == "protein-coding gene" ][ ["symbol", "ensembl_gene_id"]  ].set_index("ensembl_gene_id")["symbol"].to_dict()

# rename columns using HUGO names
df_select = df.rename(columns=lambda x: gene_names.get(x.split(".")[0], x))
# remove columns that aren't renamed (including all non-protein coding columns)
df_select = df_select.loc[:, df_select.columns.map(lambda x:not x.startswith("ENSG0")) ]

norm = ExpMinMaxScaler()
norm.fit(df_select)
df_norm = pd.DataFrame( norm.transform(df_select), index=df_select.index, columns=df_select.columns)

dataloader = dataframe_loader(df_norm, batch_size=512)

vae = VAE(df_norm.columns, latent_dim=512)
vae.fit(dataloader, epochs=128)

df_enc = vae.encoder(dataframe_tensor(df_norm))

out = pd.DataFrame( vae.decoder(df_enc[0]).detach().cpu(), index=df_norm.index, columns=df_norm.columns)

```
