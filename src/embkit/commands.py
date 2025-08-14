import click
import pandas as pd
from ..models import VAE

@click.group()
def model():
    """
    model - model grouping
    """
    pass


@model.command()
@click.argument("input")
@click.option("--latent", default=256)
@click.option("--epochs", default=20)
def train(input, latent, epochs):
    """
    Train VAE model
    """

    df = pd.read_csv(input, sep="\t", index_col=0)

    vae = VAE(input_dim=df.shape[1], hidden_dim=400, latent_dim=latent)

    vae.fit(df, epochs=epochs)

