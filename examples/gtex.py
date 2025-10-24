from embkit import dataframe_loader, dataframe_tensor
from embkit.preprocessing import load_gct, ExpMinMaxScaler
from embkit.datasets import GTEx, Hugo
from embkit.models.vae import VAE
from embkit.layers import LayerInfo

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np


# ---------- regression VAE loss (MSE + β·KL) ----------
def vae_mse_loss(recon, x, mu, logvar, beta=1.0, reduction="mean"):
    # recon loss
    recon_loss = F.mse_loss(recon, x, reduction=reduction)
    # KL( q(z|x) || N(0, I) )
    # mean over batch for stability; fit() averages per-epoch across batches
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl
    return total, recon_loss, kl


# ---------- data ----------
g = GTEx()
hugo = Hugo()

print("Loading data...")
# Use dataset file paths
hugo_df = pd.read_csv(hugo.unpacked_file_path, sep="\t", index_col=0)
df = load_gct(g.unpacked_file_path)

# Map Ensembl -> HUGO (protein-coding only)
gene_names = (
    hugo_df[hugo_df["locus_group"] == "protein-coding gene"]
    [["symbol", "ensembl_gene_id"]]
    .set_index("ensembl_gene_id")["symbol"]
    .to_dict()
)
df_select = df.rename(columns=lambda x: gene_names.get(x.split(".")[0], x))
df_select = df_select.loc[:, df_select.columns.map(lambda x: not x.startswith("ENSG0"))]

# Normalize to [0,1]
norm = ExpMinMaxScaler()
norm.fit(df_select)
df_norm = pd.DataFrame(norm.transform(df_select), index=df_select.index, columns=df_select.columns)

# ---------- model ----------
dataloader = dataframe_loader(df_norm, batch_size=512)
dec_layers = [LayerInfo(1024, activation="relu"), LayerInfo(2048, activation="relu"),
              LayerInfo(len(df_norm.columns), activation=None)]
enc_layers = [LayerInfo(2048, activation="relu"), LayerInfo(1024, activation="relu"), LayerInfo(128, activation="relu")]

schedule = [(0.0, 20), (0.1, 20), (0.3, 40), (0.4, 40)]
vae = VAE(df_norm.columns, latent_dim=128, decoder_layers=dec_layers, encoder_layers=enc_layers)

# Train with regression loss (choose one)
vae.fit(X=dataloader, beta_schedule=schedule, lr=1e-3, loss=vae_mse_loss)

# ---------- inference----------
vae.eval()
with torch.no_grad():
    x_t = dataframe_tensor(df_norm)
    recon, mu, logvar, z = vae(x_t)
    out = pd.DataFrame(recon.cpu().numpy(), index=df_norm.index, columns=df_norm.columns)

# ---------- metrics ----------
gene = "OR4F5"
y = df_norm[gene].values
yhat = out[gene].values
corr = np.corrcoef(y, yhat)[0, 1]
mae_gene = np.mean(np.abs(y - yhat))
print(gene, "corr:", float(corr), "MAE:", float(mae_gene))

print("input mean/std:", float(df_norm.values.mean()), float(df_norm.values.std()))
print("recon mean/std:", float(out.values.mean()), float(out.values.std()))
print("global MAE:", float(np.mean(np.abs(out.values - df_norm.values))))
