
import click
import pandas as pd
import json

from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import DataLoader

from .. import dataframe_loader, dataframe_tensor, get_device, dataframe_dataset
from ..files import H5Reader
from ..factory import save, load
from ..factory.layers import Layer, LayerList
from ..optimize import fit_vae
from ..models.vae.vae import VAE
from ..models.vae.net_vae import NetVAE
from ..preprocessing import ExpMinMaxScaler, get_dataset_nonzero_mask
from ..datasets import DatasetMask
from ..losses import bce_with_logits, bce, mse
from ..pathway import extract_sif_interactions, feature_map_intersect, build_feature_map_indices

model = click.Group(name="model", help="VAE Model commands.")

@model.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str))
@click.option("--group", "-g", default=None, help="HD5F group name")
@click.option("--latent", "-l", type=int, default=256, show_default=True, help="Latent dimension size.")
@click.option("--epochs", "-e", type=int, default=20, show_default=True, help="Training epochs.")
@click.option("--batch-size", "-b", type=int, default=256)
@click.option("--encode-layers", type=str, default="400,200")
@click.option("--decode-layers", type=str, default="200,400")
@click.option("--normalize", "-n", type=click.Choice(["none", "expMinMax", "minMax"]), default="none")
@click.option("--final-activation", default="none", type=click.Choice(["none", "sigmoid", "relu"]))
@click.option("--learning-rate", "-r", type=float, default=0.0001)
@click.option("--out", "-o", type=str, default=None)
@click.option("--schedule", "-s", type=str, default=None, help="20:0,20:0.1,40:.3,40:.4")
@click.option("--loss", type=click.Choice(["mse", "bce", "bce-logit"]), default="bce-logit")
@click.option("--save-stats", is_flag=True)
@click.option("--zero-mask", default=None, type=float)
@click.option("--seed", default=42, type=int)
@click.option("--bfloat16", is_flag=True)
def train_vae(input_path: str,
              group: str,
              latent: int,
              epochs: int,
              batch_size: int,
              out: str, normalize:str,
              encode_layers:str, decode_layers:str,
              learning_rate: float,
              final_activation: str,
              loss: str,
              schedule:str,
              zero_mask: float,
              save_stats: bool,
              seed: int,
              bfloat16: bool
              ):
    """
    Train VAE model from a TSV file.
    """
    device = get_device()
    dtype = torch.float32
    if bfloat16:
        dtype = torch.bfloat16

    torch.manual_seed(seed)
    df = None
    if group is not None:
        if normalize != "none":
            raise click.BadParameter(
                "Normalization for HDF5 input is not supported in train-vae. "
                "Use '--normalize none' for HDF5 or provide TSV input for normalization."
            )
        dataset = H5Reader(input_path, group)
        if zero_mask is not None:
            dataset_mask = get_dataset_nonzero_mask(dataset, zero_mask)
            mask = dataset_mask[0].cpu().numpy()
            features = dataset.columns[mask]
            dataset = DatasetMask(dataset, dataset_mask, device)
        else:
            dataset.to(device)
            features = dataset.columns
    
    else:
        df = pd.read_csv(input_path, sep="\t", index_col=0)

        if normalize == "expMinMax":
            norm = ExpMinMaxScaler()
            norm.fit(df)
            df = pd.DataFrame( norm.transform(df), index=df.index, columns=df.columns)
        elif normalize == "minMax":
            norm = MinMaxScaler()
            norm.fit(df)
            df = pd.DataFrame( norm.transform(df), index=df.index, columns=df.columns)
        features = df.columns
        dataset = dataframe_dataset(df, device=device, dtype=dtype)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    

    layer_sizes = list( int(i) for i in encode_layers.split(",") )
    enc_layers_list = LayerList( layer_sizes )

    layer_sizes = list( int(i) for i in decode_layers.split(",") )
    dec_layers_list = LayerList( layer_sizes, end_activation=final_activation )

    beta_schedule = None
    if schedule is not None:
        beta_schedule = []
        for b in schedule.split(","):
            e, b = b.split(":")
            beta_schedule.append( (float(b), int(e)) )
    vae = VAE(features=features,
              latent_dim=latent,
              encoder_layers=enc_layers_list,
              decoder_layers=dec_layers_list,
              device=device, dtype=dtype)

    loss_func = bce_with_logits
    if loss == "mse":
        loss_func = mse
    elif loss == "bce":
        loss_func = bce

    fit_vae(vae, dataloader, epochs=epochs,
            beta_schedule=beta_schedule, lr=learning_rate, loss=loss_func)
    click.echo("Training complete.")

    if out is None:
        click.echo(f"No output path provided, using default naming.")
        out = f"vae_latent{latent}_epochs{epochs}.model"

    save(vae, out)
    click.echo(f"Model saved, to {out}")

    if save_stats and df is not None:
        stats = pd.DataFrame({"mean": df.mean(), "std": df.std(ddof=0)})
        stats_path = f"{out}.stats.tsv"
        stats.to_csv(stats_path, sep="\t")
        click.echo(f"Stats saved, to {stats_path}")


@model.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str))
@click.argument("pathway_sif", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str))
@click.option("--epochs", "-e", type=int, default=20, show_default=True, help="Training epochs.")
@click.option("--normalize", "-n", type=str, default="none")
@click.option("--learning-rate", "-r", type=float, default=0.0001)
@click.option("--out", "-o", type=str, default=None)
@click.option("--loss", type=click.Choice(["mse", "bce", "bce-logit"]), default="bce-logit")
@click.option("--group-layer-size", default="5,2,1", show_default=True,
              help="Comma-separated per-group widths for NetVAE masked layers.")
@click.option("--save-stats", is_flag=True)
def train_netvae(input_path: str, pathway_sif:str, out:str,
                epochs: int, normalize: str,
                learning_rate: float,
                loss:str,
                group_layer_size: str,
                save_stats:bool):
    """Train VAE model from a TSV file."""
    df = pd.read_csv(input_path, sep="\t", index_col=0)

    feature_map = extract_sif_interactions(pathway_sif)
    feature_map = feature_map_intersect(feature_map, df.columns)
    feature_idx, group_idx = build_feature_map_indices(feature_map)

    df = df[feature_idx]
    if normalize == "expMinMax":
        norm = ExpMinMaxScaler()
        norm.fit(df)
        df = pd.DataFrame( norm.transform(df), index=df.index, columns=df.columns)

    batch_size=256
    dataloader = dataframe_loader(df, batch_size=batch_size)

    group_count = len(group_idx)
    feature_count = len(feature_idx)

    click.echo(f"Feature count {feature_count} latent_size: {group_count}")

    gcounts = [int(v.strip()) for v in group_layer_size.split(",") if v.strip()]
    if not gcounts or any(v <= 0 for v in gcounts):
        raise click.BadParameter("--group-layer-size must contain one or more positive integers.")

    loss_func = bce_with_logits
    if loss == "mse":
        loss_func = mse
    elif loss == "bce":
        loss_func = bce

    schedule = [(0.0, epochs)]
    vae = NetVAE(list(df.columns), latent_groups=feature_map, group_layer_size=gcounts)
    fit_vae(vae, X=dataloader, beta_schedule=schedule, lr=learning_rate, loss=loss_func)

    click.echo("Training complete.")

    if out is None:
        click.echo("No output path provided, using default naming.")
        out = f"netvae_latent{group_count}_epochs{epochs}.model"

    save(vae, out)
    click.echo(f"Model saved, to {out}")

    if save_stats:
        stats = pd.DataFrame({"mean": df.mean(), "std": df.std(ddof=0)})
        stats_path = f"{out}.stats.tsv"
        stats.to_csv(stats_path, sep="\t")
        click.echo(f"Stats saved, to {stats_path}")

@model.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str))
@click.argument("model_path", type=click.Path(exists=True, dir_okay=True, readable=True, path_type=str))
@click.option("--normalize", "-n", type=str, default="none")
@click.option("--out", "-o", type=str, default="embedding.tsv")
def encode(input_path: str, model_path:str, normalize:str, out:str):

    m = load(model_path)
    df = pd.read_csv(input_path, sep="\t", index_col=0)

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
@click.argument("model_path", type=click.Path(exists=True, dir_okay=True, readable=True, path_type=str))
@click.option("--json", "as_json", is_flag=True, help="Emit a machine-readable JSON report.")
@click.option("--ci", is_flag=True, help="CI mode: same as '--json --fail-on-unhealthy'.")
@click.option("--fail-on-unhealthy/--no-fail-on-unhealthy", default=False, show_default=True,
              help="Return non-zero exit code when integrity checks fail.")
@click.option("--strict", is_flag=True, help="Enable strict identity checks against expected model shape.")
@click.option("--expected-feature-count", type=int, default=None,
              help="Expected number of input features in strict mode.")
@click.option("--expected-latent-dim", type=int, default=None,
              help="Expected latent dimension in strict mode.")
@click.option("--expected-features-file",
              type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
              default=None,
              help="Path to newline-delimited expected feature names in strict mode.")
def verify(
    model_path: str,
    as_json: bool,
    ci: bool,
    fail_on_unhealthy: bool,
    strict: bool,
    expected_feature_count: int | None,
    expected_latent_dim: int | None,
    expected_features_file: str | None,
):
    """
    Verify model integrity and architecture sanity.
    """
    from ..factory.core import run_model_verification
    from .. import get_device

    try:
        if ci:
            as_json = True
            fail_on_unhealthy = True

        if not as_json:
            click.secho(f"Starting integrity verification: {model_path}...", fg="cyan", bold=True)
        report = run_model_verification(model_path, device=get_device())
        report.setdefault("issues", [])

        expected_features = None
        if expected_features_file is not None:
            with open(expected_features_file, encoding="utf-8") as handle:
                expected_features = [line.strip() for line in handle if line.strip()]

        if strict:
            strict_issues = []
            actual_feature_names = report.get("feature_names")
            actual_feature_count = report.get("features_count")

            if expected_features is not None:
                if actual_feature_names is None:
                    strict_issues.append(
                        "Strict check failed: model report does not expose feature names."
                    )
                elif list(actual_feature_names) != list(expected_features):
                    strict_issues.append(
                        "Strict check failed: feature names/order do not match expected list."
                    )

            if expected_feature_count is not None:
                if actual_feature_count is None:
                    strict_issues.append(
                        "Strict check failed: model report does not expose feature count."
                    )
                elif int(actual_feature_count) != int(expected_feature_count):
                    strict_issues.append(
                        f"Strict check failed: expected feature count {expected_feature_count}, got {actual_feature_count}."
                    )

            if expected_latent_dim is not None:
                actual_latent_dim = None
                deep_audit = report.get("deep_audit")
                if isinstance(deep_audit, dict):
                    actual_latent_dim = deep_audit.get("latent_dim")
                if actual_latent_dim is None:
                    actual_latent_dim = report.get("declared_latent_dim")
                if actual_latent_dim is None:
                    strict_issues.append(
                        "Strict check failed: model report does not expose latent dim."
                    )
                elif int(actual_latent_dim) != int(expected_latent_dim):
                    strict_issues.append(
                        f"Strict check failed: expected latent dim {expected_latent_dim}, got {actual_latent_dim}."
                    )

            report["strict_mode"] = True
            report["strict_issues"] = strict_issues
            if strict_issues:
                report["healthy"] = False
                report["issues"].extend(strict_issues)
        else:
            report["strict_mode"] = False
            report["strict_issues"] = []

        if as_json:
            click.echo(json.dumps(report, indent=2, sort_keys=True))
        else:
            if report["healthy"]:
                click.secho("PASS: model integrity checks passed.", fg="green", bold=True)
            else:
                click.secho("FAIL: model integrity checks failed.", fg="red", bold=True)

            click.echo(f"Model Type: {report['model_type']}")
            click.echo("\n--- Integrity Diagnostics ---")
            param_issues = [i for i in report["issues"] if any(k in i for k in ["parameter", "NaN", "weight norm"])]
            click.echo(f"  {'PASS' if not param_issues else 'FAIL'} Weight Health: {report.get('weight_norm_max', 0.0):.2f} (max norm)")

            if "history_summary" in report:
                hist = report["history_summary"]
                click.echo(f"  PASS Training Trace: {hist['epochs']} epochs, loss improvement {hist['improvement']:.4f}")
            else:
                click.secho("  WARN Training Trace: Missing history; cannot assess learning trend.", fg="yellow")

            for issue in report["issues"]:
                color = "red" if not report["healthy"] else "yellow"
                click.secho(f"  - {issue}", fg=color)

            if "sparsity_audit" in report:
                click.echo("\n--- Sparsity & Leakage Audit ---")
                for layer in report["sparsity_audit"]:
                    status = "PASS" if layer["is_healthy"] else "FAIL"
                    leak_str = f"Leakage Sum: {layer['leakage_sum']:.2e}" if layer['leakage_sum'] > 1e-12 else "Zero Leakage"
                    click.echo(f"  {status} {layer['layer']}:")
                    click.echo(f"     Mask Sparsity: {layer['effective_sparsity']:.2%}")
                    click.echo(f"     Raw Sparsity:  {layer['raw_sparsity']:.2%}")
                    click.echo(f"     {leak_str}")

            if "deep_audit" in report:
                audit = report["deep_audit"]
                click.echo("\n--- Latent Manifold Audit ---")
                click.echo(f"  Latent Units: {audit.get('latent_dim', 'unknown')}")
                click.echo(f"  Dead Units: {audit.get('dead_units', 'unknown')}")
                if "reconstruction_mse" in audit:
                    click.echo(f"  Reconstruction MSE: {audit['reconstruction_mse']:.4f}")

            if "rna_diagnostics" in report:
                diag = report["rna_diagnostics"]
                click.echo("\n--- RNA Specific Diagnostics ---")
                click.echo(f"  Non-negative heads: {'PASS' if diag['is_non_negative'] else 'FAIL'}")
                click.echo(f"  Min mu: {diag['min_mu']:.4f}, Min logvar: {diag['min_logvar']:.4f}")

        if fail_on_unhealthy and not report["healthy"]:
            raise click.ClickException("Model integrity check failed.")

    except click.ClickException:
        raise
    except Exception as e:
        click.secho(f"Error during verification: {e}", fg="red")
        raise click.Abort()
