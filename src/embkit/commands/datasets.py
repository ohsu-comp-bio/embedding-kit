import click
import pandas as pd
from ..preprocessing import load_raw_hugo, load_gct
from ..datasets import GTEx, Hugo, SIF

datasets = click.Group(name="datasets", help="Datasets commands.")


@datasets.command()
@click.option("--data_type", "-t", required=False, type=str, help="Name of the dataset to download.")
@click.option("--output_folder", "-f", required=True, type=str, help="The folder to download the dataset into.")
def gtex(data_type: str, output_folder: str):
    """Download GTEx dataset."""

    if data_type not in GTEx.NAMES:
        click.echo(f"Dataset name '{data_type}' is not recognized. Available options are: {list(GTEx.NAMES.keys())}")
        return

    GTEx(data_type=data_type, save_path=output_folder)



@datasets.command()
@click.option("--output_folder", "-f", required=True, type=str, help="The folder to download the dataset into.")
def sif(output_folder: str):
    """Download SIF dataset."""
    SIF(save_path=output_folder)



@datasets.command()
@click.option("--output_folder", "-f", required=True, type=str, help="The folder to download the dataset into.")
@click.option("--gtex_conversion", "-gtex", required=False, is_flag=True, type=bool, default=False,
              help="Whether to convert the hugo data to a gtex version.")
def hugo(gtex_conversion: bool, output_folder: str):
    """Download Hugo dataset."""

    click.echo(f"Downloading hugo dataset into '{output_folder}'")
    hugo_data: Hugo = Hugo(save_path=output_folder)

    if gtex_conversion:
        click.echo("Converting hugo dataset using GTEx dataset...")
        click.echo("Downloading required dataset (GTEx)...")
        gtex_data: GTEx = GTEx(save_path=output_folder)
        click.echo("Loading GTEx dataset...")
        gtex_df: pd.DataFrame = load_gct(gtex_data)
        click.echo("Loading Hugo dataset...")
        hugo_df: pd.DataFrame = load_raw_hugo(hugo_data)

        # select protein coding genes and build a dict that translates from ensembl gene id to Hugo name
        gene_names = hugo_df[hugo_df["locus_group"] == "protein-coding gene"][["symbol", "ensembl_gene_id"]].set_index(
            "ensembl_gene_id")["symbol"].to_dict()

        # rename columns using HUGO names
        df_select = gtex_df.rename(columns=lambda x: gene_names.get(x.split(".")[0], x))
        # remove columns that aren't renamed (including all non-protein coding columns)
        df_select = df_select.loc[:, df_select.columns.map(lambda x: not x.startswith("ENSG0"))]
        output_path = f"{hugo_data.save_path}/gtex.hugo.tsv"
        click.echo(f"Saving converted dataset to '{output_path}...'")
        df_select.to_csv(output_path, sep="\t")
        click.echo("Done.")
