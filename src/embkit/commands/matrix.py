"""
embkit matrix — preprocessing of expression/count matrices.

The ``matrix`` command group prepares the tab-separated matrices that become the
input to an embedding model. Its subcommands implement the "normalize" and "pca"
major operations: rescaling raw feature values to a comparable ``[0, 1]`` range,
and projecting a matrix onto its leading principal components. These are the
data-preparation steps you run before (and independently of) training.
"""
import click
import pandas as pd
from pathlib import Path
from ..utilities.pca import run_pca


matrix = click.Group(name="matrix", help="Model commands.")

@matrix.command()
@click.argument("srcs", nargs=-1)
@click.option("--out", "-o", required=True)
@click.option("--features", default=None, type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str))
@click.option("--col-quantile", "col_quantile", default=False, is_flag=True)
@click.option("--quantile-max", "quantile_max", default=0.9, type=float)
@click.option("--precision", "-p", type=int, default=5)
def normalize(srcs, out, features, col_quantile, quantile_max, precision):
    """
    embkit matrix normalize — rescale one or more matrices for embedding training.

    Reads tab-separated expression/count matrices, optionally subsets them to a
    list of features, concatenates them, and writes a single normalized matrix in
    which each sample's values fall in ``[0, 1]``. This is the preprocessing step
    that puts the input on a comparable scale before a VAE / NetVAE is trained.

    By default each *sample* (row) is divided by its own ``quantile_max``
    quantile; passing ``--col-quantile`` divides each *feature* (column) by its
    quantile instead.

    Args:
        srcs: One or more input TSV matrices (index column required).
        out: Path to write the normalized TSV.
        features: Optional file listing feature names to keep.
        col_quantile: Normalize per feature (column) rather than per sample.
        quantile_max: The quantile used as the divisor (default 0.9).
        precision: Number of decimals to round to in the output.
    """

    features_list = None
    if features is not None:
        with open(features, encoding="ascii") as handle:
            features_list = list(line.rstrip() for line in handle)

    dfs = []
    for i in srcs:
        df = pd.read_csv(i, sep="\t", index_col=0)
        if features_list is not None:
            df = df[features_list]
        dfs.append(df)
    if len(dfs) == 0:
        click.echo("No matrices defined")
        return
    df = pd.concat( dfs )

    if not col_quantile:
        normDF = (df.transpose() / df.quantile(quantile_max, axis=1)).transpose().clip(upper=1.0, lower=0.0).fillna(0.0)
    else:
        normDF = (df / df.quantile(quantile_max)).clip(upper=1.0, lower=0.0).fillna(0.0)
    normDF.round(decimals=precision).to_csv(out, sep="\t")

@matrix.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str))
@click.option("--pca-size", required=True, type=int, help="Number of principal components.")
@click.option("--out", "-o", default=None, type=str, help="Output TSV path.")
def pca(input_path, pca_size, out):
    """
    embkit matrix pca — project a matrix onto its leading principal components.

    Runs principal component analysis on a tab-separated matrix and writes the
    first ``pca_size`` components to a TSV. Useful for reducing dimensionality
    before downstream analysis or visualization (e.g. inspecting whether a
    normalized matrix separates samples as expected).

    If no ``--out`` is given, the output defaults to ``<input_stem>.pca.tsv``.

    Args:
        input_path: Input TSV matrix.
        pca_size: Number of principal components to retain (must be > 0).
        out: Output TSV path (optional; defaults to ``<input_stem>.pca.tsv``).
    """
    if pca_size <= 0:
        raise click.BadParameter("--pca-size must be a positive integer.")

    if out is None:
        out = f"{Path(input_path).stem}.pca.tsv"
        click.echo(f"No output path provided, using default naming: {out}")

    run_pca(input_file=input_path, pca_size=pca_size, output_file=out)
    click.echo(f"PCA saved to {out}")
