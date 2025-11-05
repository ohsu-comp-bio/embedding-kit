
import click
import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler

from .. import dataframe_loader, dataframe_tensor, get_device
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
@click.option("--schedule", "-s", type=str, default=None, help="20:0,20:0.1,40:.3,40:.4")
@click.option("--loss", type=click.Choice(["mse", "bce", "bce-logit"]), default="bce-logit")
@click.option("--save-stats", is_flag=True)
def train_vae(input_path: str, latent: int, 
              epochs: int, out: str, normalize:str,
              encode_layers:str, decode_layers:str,
              learning_rate: float,
              loss: str,
              schedule:str, 
              save_stats: bool
              ):
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

    beta_schedule = None
    if schedule is not None:
        beta_schedule = [] #[(0.0, 20), (0.1, 20), (0.3, 40), (0.4, 40)]
        for b in schedule.split(","):
            e, b = b.split(":")
            beta_schedule.append( (float(b), int(e)) )
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
            beta_schedule=beta_schedule, lr=learning_rate, loss=loss_func)
    click.echo("Training complete.")

    if out is None:
        click.echo(f"No output path provided, using default naming.")
        out = f"vae_latent{latent}_epochs{epochs}.model"

    if save_stats:
        vae.save(out, df)
    else:
        vae.save(out)

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
@click.option("--save-stats", is_flag=True)
def train_netvae(input_path: str, pathway_sif:str, out:str,
                encode_layers:str, decode_layers:str,
                epochs: int, normalize: str,
                learning_rate: float,
                loss:str,
                save_stats:bool):
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

    click.echo(f"Feature count {feature_count} latent_size: {group_count}")

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

    if save_stats:
        vae.save(out, df)
    else:
        vae.save(out)

@model.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str))
@click.argument("model_path", type=click.Path(exists=True, dir_okay=True, readable=True, path_type=str))
@click.option("--normalize", "-n", type=str, default="none")
@click.option("--out", "-o", type=str, default="embedding.tsv")
def encode(input_path: str, model_path:str, normalize:str, out:str):
    m = VAE.open_model(path=model_path)
    df = pd.read_csv(input_path, sep="\t", index_col=0)

    # print(m)
    # df = pd.read_csv(input_path, sep="\t", index_col=0)

    df = df[m.features]
    if normalize == "expMinMax":
        norm = ExpMinMaxScaler()
        norm.fit(df)
        df = pd.DataFrame( norm.transform(df), index=df.index, columns=df.columns)

    df_tensor = dataframe_tensor(df).to(get_device())
    m.to(get_device())
    result = m.encoder(df_tensor)
    
    martix = result[2].detach().cpu().numpy()
    out_df = pd.DataFrame(martix, index=df.index)
    out_df.to_csv(out, sep="\t")

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
