import click
import pandas as pd

matrix = click.Group(name="matrix", help="Model commands.")

@matrix.command()
@click.argument("srcs", nargs=-1)
@click.option("--out", "-o", required=True)
@click.option("--features", default=None, type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str))
@click.option("--col-quantile", "col_quantile", default=False, is_flag=True)
@click.option("--quantile-max", "quantile_max", default=0.9, type=float)
@click.option("--precision", "-p", type=int, default=5)
def normalize(srcs, out, features, col_quantile, quantile_max, precision):

    features_list = None
    if features is not None:
        features = []
        with open(features, encoding="ascii") as handle:
            features_list = list(line.rstrip() for line in handle)

    dfs = []
    for i in srcs:
        df = pd.read_csv(i, sep="\t", index_col=0)
        if features is not None:
            df = df[features]
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
