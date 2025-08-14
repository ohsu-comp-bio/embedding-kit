import click
import pandas as pd
from ..vae import VAE


model = click.Group(name="model", help="Model commands.")

@model.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str))
@click.option("--latent", "-l", type=int, default=256, show_default=True, help="Latent dimension size.")
@click.option("--epochs", "-e", type=int, default=20, show_default=True, help="Training epochs.")
def train(input_path: str, latent: int, epochs: int):
    """Train VAE model from a TSV file."""
    df = pd.read_csv(input_path, sep="\t", index_col=0)
    vae = VAE(input_dim=df.shape[1], hidden_dim=400, latent_dim=latent)
    vae.fit(df, epochs=epochs)
    click.echo("Training complete.")