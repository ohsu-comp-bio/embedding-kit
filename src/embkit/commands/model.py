import click
import pandas as pd
from ..layers import LayerInfo
from ..models.vae.vae import VAE

from ..pathway import extract_pathway_interactions, FeatureGroups


vae_model = click.Group(name="vae-model", help="VAE Model commands.")
@vae_model.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str))
@click.option("--latent", "-l", type=int, default=256, show_default=True, help="Latent dimension size.")
@click.option("--epochs", "-e", type=int, default=20, show_default=True, help="Training epochs.")
def vae_train(input_path: str, latent: int, epochs: int):
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



netvae_model = click.Group(name="netvae-model", help="VAE Model commands.")
@netvae_model.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str))
@click.argument("pathway_sif", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str))
def netvae_train(input_path: str, pathway_sif:str, epochs: int):
    """Train VAE model from a TSV file."""
    df = pd.read_csv(input_path, sep="\t", index_col=0)

    df_sif = extract_pathway_interactions(pathway_sif)


    encoder_layers: list[LayerInfo] = [
        LayerInfo(units=400, activation="relu", batch_norm=True),
        LayerInfo(units=200, activation="relu", batch_norm=True),
        LayerInfo(units=latent, activation=None, batch_norm=False, bias=False)
    ]

    vae = VAE(features=df.columns, latent_dim=latent, encoder_layers=encoder_layers)

    vae.fit(df, epochs=epochs)
    click.echo("Training complete.")
