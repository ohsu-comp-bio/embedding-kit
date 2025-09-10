

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
from embkit.layers import LayerInfo
import torch
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
dec_layers = [LayerInfo(1024, activation="relu"), LayerInfo(2048, activation="relu")]
enc_layers = [LayerInfo(2048, activation="relu"), LayerInfo(1024, activation="relu"), LayerInfo(128, activation="relu")]

schedule = [(0.0, 20), (0.1, 20), (0.3, 40), (0.4, 40)]
vae = VAE(df_norm.columns, latent_dim=128, decoder_layers=dec_layers, encoder_layers=enc_layers)
vae.fit(dataloader, beta_schedule=schedule, lr=1e-3)

vae.eval()
with torch.no_grad():
    x_t = dataframe_tensor(df_norm)
    recon_logits, mu, logvar, z = vae(x_t)
    recon = torch.sigmoid(recon_logits)           

out = pd.DataFrame(recon.cpu().numpy(), index=df_norm.index, columns=df_norm.columns)

# Example metrics
gene = "OR4F5"
y = df_norm[gene].values
yhat = out[gene].values

corr = np.corrcoef(y, yhat)[0,1]
mae_gene = np.mean(np.abs(y - yhat))
print(gene, "corr:", float(corr), "MAE:", float(mae_gene))

# global calibration
print("input mean/std:", float(df_norm.values.mean()), float(df_norm.values.std()))
print("recon mean/std:", float(out.values.mean()), float(out.values.std()))
print("global MAE:", float((out.values - df_norm.values).mean()))

```
