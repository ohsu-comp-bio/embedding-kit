"""Train a VAE on a gene-expression matrix, encode samples, and score the reconstruction.

This is the core "build and test an embedding model" workflow a computational-biology
researcher reaches for first: turn a sample-by-gene expression matrix into a learned
latent representation, then check that the model actually captured the data.

What this script does, end to end:
    1. Load a small expression matrix (rows = samples, columns = genes) from ``examples/data``.
    2. Normalize with ``ExpMinMaxScaler`` (log2(x+1) then min-max to [0, 1]) — the right
       transform for TPM/count data that spans several orders of magnitude.
    3. Build a ``BaseVAE`` and train it with ``optimize.fit_vae`` under a beta-KL
       annealing schedule.  ``MSEVAELoss`` is used because the decoder has a linear
       (non-sigmoid) output head, so the reconstruction is directly comparable to the
       [0, 1]-normalized input.
    4. ``factory.save`` / ``factory.load`` round-trip to prove the model is portable.
    5. ``encode`` samples into latent space and save the embedding matrix.
    6. Score the reconstruction (per-gene Pearson r and global MAE) so you can see the
       model "working" before any downstream analysis.

Real-data variant
-----------------
The bundled data file is intentionally tiny so the script runs anywhere, in seconds, with
no network access. To run this on real GTEx data instead, replace step 1 with:

    from embkit.resources import GTEx, Hugo
    from embkit.files.loaders import load_gct
    import pandas as pd

    gtex = GTEx()                       # downloads GTEx v10 gene TPM on first use, caches under ~/.embkit
    hugo = Hugo()                       # downloads the HUGO gene-nomenclature file
    df = load_gct(gtex.unpacked_file_path)
    df = df.T                           # rows = samples, columns = Ensembl gene IDs

    # map Ensembl -> HUGO symbol, keep protein-coding genes only
    hugo_df = pd.read_csv(hugo.unpacked_file_path, sep="\t", low_memory=False)
    gene_names = (hugo_df[hugo_df["locus_group"] == "protein-coding gene"]
                  [["symbol", "ensembl_gene_id"]].dropna()
                  .set_index("ensembl_gene_id")["symbol"].to_dict())
    df.columns = [c.split(".")[0] for c in df.columns]
    df = df.rename(columns=gene_names)
    df = df.loc[:, ~df.columns.str.startswith("ENSG")]

and then continue exactly as below. The rest of the pipeline is identical.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

from embkit import dataframe_loader, dataframe_tensor, get_device
from embkit.preprocessing import ExpMinMaxScaler
from embkit.models.vae import BaseVAE
from embkit.factory.layers import Layer
from embkit.losses import MSEVAELoss
from embkit.optimize import fit_vae
from embkit.factory import save, load


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_expression_matrix(path: str) -> pd.DataFrame:
    """Load a TSV expression matrix (rows = samples, columns = genes)."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    df = df.astype("float32")
    print(f"Loaded expression matrix: {df.shape[0]} samples x {df.shape[1]} genes")
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """log2(x+1) then min-max scale to [0, 1] — appropriate for TPM/count data."""
    scaler = ExpMinMaxScaler()
    scaler.fit(df)
    return pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)


def build_model(features, latent_dim=16):
    """
    Build a VAE whose encoder/decoder widths scale with the feature count, so the same
    code works on the bundled 30-gene matrix and on a 20k-gene matrix.

    Note: ``BaseVAE`` is the concrete class that builds the encoder and decoder from the
    layer specs. The base ``VAE`` class only pairs a pre-built encoder+decoder together.
    """
    hidden = max(16, min(512, len(features)))
    encoder_layers = [
        Layer(hidden, activation="relu"),
        Layer(max(8, hidden // 2), activation="relu"),
    ]
    decoder_layers = [
        Layer(max(8, hidden // 2), activation="relu"),
        Layer(hidden, activation="relu"),
        Layer(len(features), activation=None),  # linear output head
    ]
    return BaseVAE(
        features=features,
        latent_dim=latent_dim,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
    )


def score_reconstruction(vae, df: pd.DataFrame) -> None:
    """Reconstruct the (normalized) input and report per-gene and global quality."""
    vae.eval()
    with torch.no_grad():
        recon, mu, logvar, z = vae(dataframe_tensor(df))
        recon_df = pd.DataFrame(
            recon.cpu().numpy(), index=df.index, columns=df.columns
        )

    # Per-gene Pearson correlation (top and bottom by |r|)
    corrs = []
    for gene in df.columns:
        y = df[gene].values
        yhat = recon_df[gene].values
        if np.std(y) < 1e-8 or np.std(yhat) < 1e-8:
            continue
        corrs.append((float(np.corrcoef(y, yhat)[0, 1]), gene))
    corrs.sort(key=lambda t: abs(t[0]), reverse=True)
    print("\n=== Reconstruction quality ===")
    print(f"global MAE        : {np.mean(np.abs(recon_df.values - df.values)):.4f}")
    print(f"global Pearson r  : {np.mean([c for c, _ in corrs]):.4f}")
    print("best 3 genes (|r|):")
    for c, g in corrs[:3]:
        print(f"    {g:<10} r={c:+.4f}")
    print("worst 3 genes (|r|):")
    for c, g in corrs[-3:]:
        print(f"    {g:<10} r={c:+.4f}")


def main() -> int:
    print(f"Device: {get_device()}")
    device = get_device()

    # 1) Load + 2) Normalize
    raw_path = os.path.join(DATA_DIR, "expression.tsv")
    df = load_expression_matrix(raw_path)
    df_norm = normalize(df)
    print(f"Normalized range: [{df_norm.min().min():.3f}, {df_norm.max().max():.3f}]")

    # 3) Build + train
    vae = build_model(list(df_norm.columns), latent_dim=16)
    print(f"\nTraining VAE (latent_dim=16, features={len(df_norm.columns)})...")
    history = fit_vae(
        vae,
        X=df_norm,                       # fit_vae builds its own dataloader from a DataFrame
        epochs=80,
        lr=1e-3,
        beta_schedule=[(0.0, 20), (0.1, 20), (0.3, 20), (0.4, 20)],
        loss=MSEVAELoss(),
        device=device,
        batch_size=8,
        progress=True,
    )
    print(f"\nFinal per-epoch loss: {history['loss'][-1]:.4f}")

    # 4) Save + reload (proves the model is portable across machines)
    model_path = os.path.join(DATA_DIR, "vae.model")
    save(vae, model_path)
    vae_loaded = load(model_path)
    print(f"Saved model to {model_path}; reload OK (latent_dim={vae_loaded.extra_args.get('latent_dim', 16)})")

    # 5) Encode samples into latent space
    vae_loaded.eval()
    with torch.no_grad():
        z = vae_loaded.encode(dataframe_tensor(df_norm, device=device))
    emb = pd.DataFrame(z.cpu().numpy(), index=df_norm.index)
    emb_path = os.path.join(DATA_DIR, "vae_embedding.tsv")
    # write id \t dim0 \t dim1 ... with no header — same shape as `embkit protein encode`
    emb.to_csv(emb_path, sep="\t", header=False)
    print(f"Encoded embedding shape: {emb.shape} -> {emb_path}")

    # 6) Score the reconstruction
    score_reconstruction(vae_loaded, df_norm)

    print("\nDone. Embedding matrix is ready for downstream analysis "
          "(see analyze_embeddings.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
