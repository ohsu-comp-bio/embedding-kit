
import click
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from .. import dataframe_loader, dataframe_tensor
from ..layers import LayerInfo, build_layers, ConstraintInfo
from ..models.vae.vae import VAE, BaseVAE
from ..preprocessing import ExpMinMaxScaler
from ..losses import bce_with_logits, bce, mse
from ..pathway import extract_pathway_interactions, feature_map_intersect, FeatureGroups

model = click.Group(name="model", help="VAE Model commands.")

@model.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str))
@click.option("--latent", "-l", type=int, default=256, show_default=True, help="Latent dimension size.")
@click.option("--epochs", "-e", type=int, default=20, show_default=True, help="Training epochs.")
@click.option("--encode-layers", type=str, default="400,200")
@click.option("--decode-layers", type=str, default="200,400")
@click.option("--normalize", "-n", type=str, default="none")
@click.option("--learning-rate", "-r", type=float, default=0.0001)
@click.option("--out", "-o", type=str, default=None)
@click.option("--loss", type=click.Choice(["mse", "bce", "bce-logit"]), default="bce-logit")
def train_vae(input_path: str, latent: int, 
              epochs: int, out: str, normalize:str, 
              encode_layers:str, decode_layers:str,
              learning_rate: float,
              loss: str):
    """Train VAE model from a TSV file."""
    df = pd.read_csv(input_path, sep="\t", index_col=0)

    if normalize == "expMinMax":
        norm = ExpMinMaxScaler()
        norm.fit(df)
        df = pd.DataFrame( norm.transform(df), index=df.index, columns=df.columns)
    elif normalize == "minMax":
        norm = MinMaxScaler()
        norm.fit(df)
        df = pd.DataFrame( norm.transform(df), index=df.index, columns=df.columns)


    feature_count = len(df.columns)

    layer_sizes = list( int(i) for i in encode_layers.split(",") )
    layer_sizes.append(latent)
    enc_layers = build_layers( layer_sizes )

    layer_sizes = list( int(i) for i in decode_layers.split(",") )
    layer_sizes.append(feature_count)
    dec_layers = build_layers( layer_sizes, end_activation="none" )

    schedule = [(0.0, 20), (0.1, 20), (0.3, 40), (0.4, 40)]
    #schedule = [(0.0, 20)]
    vae = VAE(features=df.columns,
              latent_dim=latent,
              encoder_layers=enc_layers,
              decoder_layers=dec_layers)

    loss_func = bce_with_logits
    if loss == "mse":
        loss_func = mse
    elif loss == "bce":
        loss_func = bce

    vae.fit(df, epochs=epochs,
            beta_schedule=schedule, lr=learning_rate, loss=loss_func)
    click.echo("Training complete.")

    vae.save(out, df)
    click.echo(f"Model saved, to {out}")


@model.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str))
@click.argument("pathway_sif", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str))
@click.option("--epochs", "-e", type=int, default=20, show_default=True, help="Training epochs.")
@click.option("--encode-layers", type=str, default="400,200")
@click.option("--decode-layers", type=str, default="200,400")
@click.option("--normalize", "-n", type=str, default="none")
@click.option("--learning-rate", "-r", type=float, default=0.0001)
@click.option("--out", "-o", type=str, default=None)
@click.option("--loss", type=click.Choice(["mse", "bce", "bce-logit"]), default="bce-logit")
def train_netvae(input_path: str, pathway_sif:str, out:str,
                encode_layers:str, decode_layers:str,
                epochs: int, normalize: str,
                learning_rate: float,
                loss:str):
    """Train VAE model from a TSV file."""
    df = pd.read_csv(input_path, sep="\t", index_col=0)

    feature_map = extract_pathway_interactions(pathway_sif)

    feature_map, isect = feature_map_intersect(feature_map, df.columns)

    df = df[isect]
    if normalize == "expMinMax":
        norm = ExpMinMaxScaler()
        norm.fit(df)
        df = pd.DataFrame( norm.transform(df), index=df.index, columns=df.columns)

    batch_size=256
    dataloader = dataframe_loader(df, batch_size=batch_size)

    fmap = FeatureGroups(feature_map)
    group_count = len(fmap)
    feature_count = len(isect)

    print(f"Feature count {feature_count} latent_size: {group_count}")

    gcounts = [5,2,1]

    enc_layers = [
        LayerInfo(group_count*gcounts[0], op="masked_linear", constraint=ConstraintInfo("features-to-group", fmap, out_group_count=gcounts[0])),
        LayerInfo(group_count*gcounts[1], op="masked_linear", constraint=ConstraintInfo("group-to-group", fmap, in_group_count=gcounts[0], out_group_count=gcounts[1])),
        LayerInfo(group_count, op="masked_linear", constraint=ConstraintInfo("group-to-group", fmap, in_group_count=gcounts[1], out_group_count=gcounts[2]))
    ]

    dec_layers = [
        LayerInfo(group_count*gcounts[1], op="masked_linear", constraint=ConstraintInfo("group-to-group", fmap, in_group_count=gcounts[2], out_group_count=gcounts[1])),
        LayerInfo(group_count*gcounts[0], op="masked_linear", constraint=ConstraintInfo("group-to-group", fmap, in_group_count=gcounts[1], out_group_count=gcounts[0])),
        LayerInfo(feature_count, op="masked_linear", constraint=ConstraintInfo("group-to-features", fmap, in_group_count=gcounts[0]), activation="none")
    ]

    loss_func = bce_with_logits
    if loss == "mse":
        loss_func = mse
    elif loss == "bce":
        loss_func = bce


    #schedule = [(0.0, 20), (0.1, 20), (0.3, 40), (0.4, 40)]
    schedule = [(0.0, 20)]
    vae = VAE(df.columns, latent_dim=group_count, encoder_layers=enc_layers, decoder_layers=dec_layers)
    vae.fit(X=dataloader, beta_schedule=schedule, lr=learning_rate, loss=loss_func)

    click.echo("Training complete.")

    vae.save(out, df)

@model.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str))
@click.argument("model_path", type=click.Path(exists=True, dir_okay=True, readable=True, path_type=str))
def encode(input_path: str, model_path:str):
    m = VAE.open_model(path=model_path)

    print(m)
    #df = pd.read_csv(input_path, sep="\t", index_col=0)


