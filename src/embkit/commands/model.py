import click
import pandas as pd
import numpy as np
import torch
from ..layers import LayerInfo
from ..models.vae.vae import VAE
from ..models.hla2vec import fit_hla2vec, load_bigmhc

model = click.Group(name="model", help="Model commands.")

@model.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str))
@click.option("--latent", "-l", type=int, default=256, show_default=True, help="Latent dimension size.")
@click.option("--epochs", "-e", type=int, default=20, show_default=True, help="Training epochs.")
def train(input_path: str, latent: int, epochs: int):
    """Train VAE model from a TSV file."""
    df = pd.read_csv(input_path, sep="\t", index_col=0)

    encoder_layers: list[LayerInfo] = [
        LayerInfo(units=400, activation="relu", batch_norm=True),
        LayerInfo(units=200, activation="relu", batch_norm=True),
        LayerInfo(units=latent, activation=None, batch_norm=False, bias=False)
    ]

    vae = VAE(features=df.columns, latent_dim=latent, encoder_layers=encoder_layers)

    vae.fit(df, epochs=epochs)
    click.echo("Training complete.")


@model.command()
@click.argument("h5", type=str)
@click.option("--epochs", "-e", type=int, default=5, show_default=True, help="Training epochs")
@click.option("--batch-size", type=int, default=1024, show_default=True, help="Batch Size")
@click.option("--learning-rate", type=float, default=1e-3, show_default=True, help="Learning Rate")
@click.option("--seed", type=int, default=42, show_default=True, help="Batch Size")
@click.option("--embedding-dim", type=int, default=64)
def train_hla2vec(h5:str, epochs:int, batch_size:int, learning_rate:float, seed:int, embedding_dim:int):

    #TODO: this should be put into a library function
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    data = load_bigmhc(h5)

    model, history = fit_hla2vec(
        **data,
        emb_dim=embedding_dim,
        batch_size=batch_size,
        epochs=epochs,
        lr=learning_rate,
        seed=seed,
    )

    print("Final epoch:", history[-1])
    torch.save(model.state_dict(), "hla2vec_demo.pt")
    print("Saved weights -> hla2vec_demo.pt")